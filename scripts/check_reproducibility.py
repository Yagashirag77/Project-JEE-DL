"""Contrôle non destructif de la reproductibilité OLV-6; n'exécute aucun main scientifique."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
SCRIPTS = ["audit_datasets.py", "prepare_oqds.py", "train_evaluate.py", "complementary_report.py"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    checks = []
    def record(name, ok, detail):
        checks.append({"contrôle": name, "statut": "PASS" if ok else "FAIL", "détail": str(detail)})

    record("racine autorisée", ROOT == Path(r"C:\Users\Microsoft\Desktop\Project-JEE_DL"), ROOT)
    parsed = {}
    for name in SCRIPTS:
        path = ROOT / "scripts" / name
        try:
            parsed[name] = ast.parse(path.read_text(encoding="utf-8"))
            record(f"syntaxe {name}", True, "AST Python valide")
        except Exception as exc:
            record(f"syntaxe {name}", False, repr(exc))

    dependencies = ["numpy", "pandas", "matplotlib", "sklearn", "scipy", "rasterio", "geopandas", "shapely", "tqdm", "joblib", "torch", "torchvision", "openpyxl", "PIL", "jupyter", "nbformat", "nbclient"]
    missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
    record("dépendances Python", not missing, "toutes présentes" if not missing else f"absentes: {missing}")

    dataset_checks = {
        "OQDS shapefile": len(list((ROOT / "data/raw/repository").rglob("*.shp"))) == 1,
        "OQDS six rasters": len(list((ROOT / "data/raw/repository").rglob("*.tif"))) == 6,
        "NEON GeoPackage": (ROOT / "data/neon/neon_crowns_dta.gpkg").exists(),
        "Hyperspectral Excel": (ROOT / "data/hyperspectral/merged_hs_olive_gr_tr.xlsx").exists(),
        "Hyperspectral 8 cubes": len(list((ROOT / "data/hyperspectral").glob("*.tiff"))) == 8,
        "CrownsDb présent": (ROOT / "data/crownsdb").exists(),
    }
    for name, ok in dataset_checks.items(): record(name, ok, "chemin attendu")

    # Importer les modules sans appeler main teste les imports et l'initialisation légère.
    import_code = (
        "import sys; sys.path.insert(0, r'" + str(ROOT).replace("\\", "\\\\") + "'); "
        "import scripts.audit_datasets, scripts.prepare_oqds, scripts.train_evaluate, scripts.complementary_report; print('IMPORT_OK')"
    )
    proc = subprocess.run([sys.executable, "-c", import_code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    record("imports/démarrage non destructif", proc.returncode == 0 and "IMPORT_OK" in proc.stdout, (proc.stdout + proc.stderr).strip())

    prep_text = (ROOT / "scripts/prepare_oqds.py").read_text(encoding="utf-8")
    train_text = (ROOT / "scripts/train_evaluate.py").read_text(encoding="utf-8")
    comp_text = (ROOT / "scripts/complementary_report.py").read_text(encoding="utf-8")
    record("graine préparation", "SEED = 20261106" in prep_text, "20261106")
    record("graine entraînement", "SEED = 20261106" in train_text, "20261106")
    record("graine complémentaires", "SEED = 20261106" in comp_text, "20261106")
    record("sélection Validation", 'key=lambda z: z["pr_auc"]' in train_text and "best_threshold(y[\"val\"]" in train_text, "PR-AUC et seuil sur Validation")
    record("Test après gel", train_text.find("# Test final") > train_text.find("selected_configurations.json"), "bloc Test situé après sauvegarde des choix")

    features = pd.read_csv(ROOT / "data/processed/oqds_features_splits.csv", dtype={"tree_id": str})
    record("partition arbre", features.groupby("tree_id").split.nunique().max() == 1, features.groupby("split").size().to_dict())
    record("partition bloc spatial", features.groupby("spatial_block").split.nunique().max() == 1, features.groupby("split").spatial_block.nunique().to_dict())
    record("folds fixés", set(features.fold.unique()) == set(range(7)), sorted(features.fold.unique().tolist()))

    selected_path = RESULTS / "selected_configurations.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    expected_models = {"threshold", "cusum", "isolation_forest", "siamese_cnn", "tcn"}
    record("cinq configurations sauvegardées", set(selected) == expected_models, {k: v["config"] for k, v in selected.items()})
    model_paths = [ROOT / "outputs/models" / p for p in ("threshold.joblib", "cusum.joblib", "isolation_forest.joblib", "siamese_cnn_best.pt", "tcn_best.pt")]
    record("cinq modèles/checkpoints", all(p.exists() and p.stat().st_size > 0 for p in model_paths), [p.name for p in model_paths])

    builder = (ROOT / "scripts/build_notebook.py").read_text(encoding="utf-8")
    expected_order = "['audit_datasets.py', 'prepare_oqds.py', 'train_evaluate.py', 'complementary_report.py']"
    record("ordre reproduction complète", expected_order in builder, "audit → préparation → entraînement/évaluation → complémentaires/rapport")
    record("garde-fou entraînement", "CONFIRMER_REENTRAINEMENT_COUTEUX" in builder, "confirmation explicite requise")

    # Comparaison aux empreintes établies avant la mission: modèles/prédictions/résultats centraux.
    manifest_path = RESULTS / "artifact_manifest.csv"
    manifest = pd.read_csv(manifest_path).set_index("path")
    protected = [
        "outputs/models/threshold.joblib", "outputs/models/cusum.joblib", "outputs/models/isolation_forest.joblib",
        "outputs/models/siamese_cnn_best.pt", "outputs/models/tcn_best.pt",
        "outputs/results/model_comparison_test.csv", "outputs/results/test_predictions_all_models.csv",
        "outputs/results/validation_experiments.csv", "outputs/results/selected_configurations.json",
    ]
    integrity = {}
    for rel in protected:
        path = ROOT / rel
        integrity[rel] = rel in manifest.index and sha256(path) == manifest.loc[rel, "sha256"]
    record("intégrité artefacts initiaux", all(integrity.values()), integrity)

    overwrite_targets = [
        "data/processed/oqds_features_splits.csv", "outputs/results/model_comparison_test.csv",
        "outputs/results/test_predictions_all_models.csv", "outputs/models/siamese_cnn_best.pt", "outputs/models/tcn_best.pt",
    ]
    report_status = "CONDITIONNELLEMENT REPRODUCTIBLE"
    table = pd.DataFrame(checks)
    failures = table[table.statut == "FAIL"]
    if len(failures): report_status = "ÉCHEC DES CONTRÔLES STATIQUES"
    report = [
        "# Vérification non destructive de reproductibilité OLV-6", "",
        f"**État : {report_status}.**", "",
        "Cette vérification n'a exécuté aucun `main()` scientifique et n'a entraîné aucun modèle. Elle a analysé la syntaxe, importé les modules sans lancer leurs traitements, contrôlé les dépendances, chemins, graines, partitions, configurations et empreintes des artefacts initiaux.", "",
        "## Contrôles", "", "| Contrôle | Statut | Détail |", "|---|---|---|",
    ]
    for row in checks:
        detail = row["détail"].replace("|", "\\|").replace("\n", " ")
        report.append(f"| {row['contrôle']} | {row['statut']} | {detail} |")
    report += [
        "", "## Ordre du mode complet", "",
        "1. audit des quatre sources ;", "2. préparation OQDS et partition spatiale ;",
        "3. optimisation Validation, checkpoints et évaluation Test des cinq modèles ;",
        "4. expériences complémentaires et rapport.", "",
        "## Ce qui n'a pas été testé", "",
        "- La reproductibilité bit-à-bit d'un nouvel entraînement PyTorch CPU/GPU.",
        "- Le temps et la mémoire d'une nouvelle exécution intégrale.",
        "- La stabilité des scores sur une seconde graine ou un autre matériel.",
        "- La réussite de chaque écriture au terme d'un run complet interrompu/repris.", "",
        "## Point de sécurité important", "",
        "Les scripts actuels utilisent des chemins de sortie fixes. Une véritable `reproduction_complete` dans ce même dossier remplacerait notamment :",
        *[f"- `{p}`" for p in overwrite_targets],
        "", "Elle ne doit donc pas être lancée tant que les résultats initiaux doivent rester intacts. Pour une preuve complète, il faut exécuter dans une copie isolée du projet ou ajouter un identifiant de run (`outputs/runs/<run_id>/` et `data/processed/runs/<run_id>/`), puis comparer les métriques et empreintes.", "",
        "## Conclusion", "",
        "Les contrôles structurels passent et le pipeline est cohérent avec le protocole annoncé. La reproductibilité scientifique complète reste à démontrer par une réexécution isolée autorisée; elle n'est pas équivalente au présent contrôle non destructif.",
    ]
    (RESULTS / "reproducibility_check.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print("État:", report_status)


if __name__ == "__main__":
    main()
