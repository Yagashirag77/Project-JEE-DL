"""Expériences complémentaires et rapport scientifique final OLV-6."""
from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20261106
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"


def md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    f = frame.copy()
    for c in f.select_dtypes(include=np.number):
        f[c] = f[c].round(digits)
    head = "| " + " | ".join(map(str, f.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(f.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in f.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *body])


def neon_experiment() -> dict:
    path = next((DATA / "neon").rglob("*.gpkg"))
    g = gpd.read_file(path)
    repeated = g[g.groupby("individual").individual.transform("size") >= 2].copy()
    repeated = repeated.dropna(subset=["detection_year"]).sort_values(["individual", "detection_year"])
    rows = []
    for individual, z in repeated.groupby("individual"):
        if z.detection_year.nunique() < 2:
            continue
        first, last = z.iloc[0], z.iloc[-1]
        rows.append({
            "individual": individual, "n_observations": len(z), "first_year": first.detection_year,
            "last_year": last.detection_year, "years_elapsed": last.detection_year - first.detection_year,
            "area_first_m2": first.area_m2, "area_last_m2": last.area_m2,
            "area_change_m2": last.area_m2 - first.area_m2,
            "height_first": first.height, "height_last": last.height,
            "height_change": last.height - first.height if pd.notna(first.height) and pd.notna(last.height) else np.nan,
            "siteID": first.siteID, "taxonID": first.taxonID,
        })
    changes = pd.DataFrame(rows)
    changes.to_csv(RESULTS / "neon_repeated_tree_changes.csv", index=False)
    summary = {
        "rows_total": len(g), "unique_individuals": int(g.individual.nunique()),
        "individuals_with_multi_year_observations": len(changes),
        "median_area_change_m2": float(changes.area_change_m2.median()) if len(changes) else None,
        "area_change_iqr_m2": [float(changes.area_change_m2.quantile(.25)), float(changes.area_change_m2.quantile(.75))] if len(changes) else None,
        "height_change_non_missing": int(changes.height_change.notna().sum()) if len(changes) else 0,
        "interpretation": "Mesure réelle de répétabilité/changement de couronne; aucune cible OQDS compatible, donc aucune fusion ni revendication de gain.",
    }
    return summary


def hyperspectral_experiment() -> dict:
    path = next((DATA / "hyperspectral").rglob("*.xlsx"))
    frame = pd.read_excel(path, sheet_name="hs_and_ground_truth")
    feature_cols = [c for c in frame.columns if str(c).startswith("_b") and str(c).endswith("_mean")]
    mask = frame.possible_inf.astype(str).str.lower().isin(["yes", "no"])
    x = frame.loc[mask, feature_cols].replace([np.inf, -np.inf], np.nan)
    y = frame.loc[mask, "possible_inf"].astype(str).str.lower().eq("yes").astype(int).to_numpy()
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
        ("pca", PCA(n_components=10, random_state=SEED)),
        ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED)),
    ])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    score = cross_val_predict(pipe, x, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    pred = (score >= .5).astype(int)
    out = pd.DataFrame({"id": frame.loc[mask, "id"], "y_true": y, "score_cv": score, "y_pred_0_5": pred})
    out.to_csv(RESULTS / "hyperspectral_cv_predictions.csv", index=False)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    summary = {
        "observations": len(y), "positives": int(y.sum()), "spectral_mean_features": len(feature_cols),
        "protocol": "Prédictions hors-fold, StratifiedKFold 3 plis; imputation, standardisation et PCA ajustées dans chaque pli.",
        "roc_auc_cv": float(roc_auc_score(y, score)), "pr_auc_cv": float(average_precision_score(y, score)),
        "accuracy_0_5": float(accuracy_score(y, pred)), "precision_0_5": float(precision_score(y, pred, zero_division=0)),
        "recall_0_5": float(recall_score(y, pred, zero_division=0)), "f1_0_5": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cube_linkage": "Impossible: 8 cubes sans identifiant/tag explicite contre 76 lignes agrégées; aucune correspondance n'est inventée.",
        "caution": "Seulement 6 positifs; résultat exploratoire à forte variance, non comparable au Test spatial OQDS.",
    }
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(score[y == 0], bins=12, alpha=.7, label="non stressé"); ax.hist(score[y == 1], bins=6, alpha=.7, label="stress possible")
    ax.set(xlabel="Score CV hors-fold", ylabel="Effectif", title="Expérience spectrale complémentaire"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "hyperspectral_cv_scores.png", dpi=180); plt.close(fig)
    return summary


def crowns_experiment() -> dict:
    root = DATA / "crownsdb"
    xmls = sorted(root.rglob("*.xml")); rows = []
    for path in xmls:
        doc = ET.parse(path).getroot()
        width = float(doc.findtext("size/width") or 0); height = float(doc.findtext("size/height") or 0)
        if width <= 0 or height <= 0:
            # Les XML fournis omettent <size>; la dimension est vérifiée dans l'image sœur.
            filename = doc.findtext("filename") or (path.stem + ".JPG")
            image_path = path.parent.parent / "Images" / filename
            if image_path.exists():
                with Image.open(image_path) as image:
                    width, height = float(image.width), float(image.height)
        for obj in doc.findall("object"):
            box = obj.find("bndbox")
            if box is None or width <= 0 or height <= 0: continue
            xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
            xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
            rows.append({"annotation": path.relative_to(ROOT).as_posix(), "image_width": width, "image_height": height,
                         "box_relative_area": max(0, xmax-xmin) * max(0, ymax-ymin) / (width*height),
                         "box_width_px": max(0, xmax-xmin), "box_height_px": max(0, ymax-ymin)})
    boxes = pd.DataFrame(rows); boxes.to_csv(RESULTS / "crownsdb_box_quality.csv", index=False)
    return {
        "annotation_files": len(xmls), "boxes": len(boxes),
        "median_relative_box_area": float(boxes.box_relative_area.median()),
        "relative_box_area_iqr": [float(boxes.box_relative_area.quantile(.25)), float(boxes.box_relative_area.quantile(.75))],
        "non_positive_boxes": int((boxes.box_relative_area <= 0).sum()),
        "role": "Source utile pour pré-entraînement/détection de couronnes et contrôle d'échelle; aucune date ni classe sanitaire, donc pas d'évaluation temporelle OQDS.",
    }


def main() -> None:
    complementary = {"NEON": neon_experiment(), "Hyperspectral": hyperspectral_experiment(), "CrownsDb": crowns_experiment()}
    (RESULTS / "complementary_experiments.json").write_text(json.dumps(complementary, indent=2, ensure_ascii=False), encoding="utf-8")
    test = pd.read_csv(RESULTS / "model_comparison_test.csv")
    selected = json.loads((RESULTS / "selected_configurations.json").read_text(encoding="utf-8"))
    errors = pd.read_csv(RESULTS / "error_analysis.csv")
    prep = json.loads((RESULTS / "preparation_summary.json").read_text(encoding="utf-8"))
    best = test.sort_values("pr_auc", ascending=False).iloc[0]
    neon_change_text = (f"Le changement médian d'aire est {complementary['NEON']['median_area_change_m2']:.3f} m²."
                        if complementary['NEON']['median_area_change_m2'] is not None
                        else "Aucun individu n'a deux années de détection distinctes exploitables; aucun changement longitudinal n'est estimé.")

    phase_bc = f"""# Phases B–C — protocole et préparation

- Benchmark commun: OQDS seulement, cible `is.infect` (retrait/disparition probable après OQDS).
- Représentations: intensités temporelles 2019–2021 pour seuil/CUSUM/Isolation Forest/TCN; paires RGB 2019–2021 pour Siamese.
- Les fichiers nommés NDVI sont des RGBA 8 bits sans calibration: ils sont traités comme intensité, jamais comme NDVI physique.
- Nettoyage: {prep['duplicate_coordinate_rows_removed']} doublons géométriques retirés, {sum(prep['invalid_counts'].values())} observations invalides exclues, {prep['usable_points']} arbres retenus.
- Split unique par blocs de {prep['spatial_block_m']} m: {prep['split_counts']}. Aucun arbre ni bloc partagé.
- Les scalers RGB, numériques et TCN sont ajustés exclusivement sur Train.
"""
    (RESULTS / "PHASE_B_C_PROTOCOL.md").write_text(phase_bc, encoding="utf-8")

    phase_de = "# Phases D–E — optimisation et Test final\n\n" + md_table(test) + "\n\nConfigurations retenues sur Validation (PR-AUC):\n\n```json\n" + json.dumps(selected, indent=2, ensure_ascii=False) + "\n```\n"
    (RESULTS / "PHASE_D_E_RESULTS.md").write_text(phase_de, encoding="utf-8")

    report = f"""# OliveSmart OLV-6 — comparaison scientifique de cinq algorithmes

## Objectif et données

L'expérience compare cinq pipelines sur les mêmes arbres OQDS et une séparation spatiale figée. OQDS contient trois images WorldView-2 co-enregistrées (2019–2021) et des arbres considérés infectés lorsqu'un objet présent en 2019 est absent ensuite. Cette définition est cohérente avec l'article source, mais reste une infection probable fondée sur le retrait et non un diagnostic PCR individuel ([publication OQDS](https://pmc.ncbi.nlm.nih.gov/articles/PMC12142346/)).

L'audit réel a couvert quatre sources. OQDS est seul compatible avec le benchmark temporel commun. NEON contient d'autres taxons et pas de cible OQDS; le classeur hyperspectral décrit un stress spectral très déséquilibré; CrownsDb fournit des boîtes de couronnes sans suivi sanitaire temporel.

## Préparation et protocole anti-fuite

Après retrait de {prep['duplicate_coordinate_rows_removed']} coordonnées dupliquées et {sum(prep['invalid_counts'].values())} observations invalides, {prep['usable_points']} arbres restent. Les six rasters OQDS partagent EPSG:32633, une résolution de 0,5 m, la même transformation et la même emprise. Les fichiers appelés NDVI sont en réalité des rendus RGBA `uint8` sans calibration; les variables correspondantes sont donc nommées intensités.

Une partition par blocs de {prep['spatial_block_m']} m donne Train/Validation/Test sans arbre ni bloc partagé. Toutes les transformations sont ajustées sur Train. Trois configurations sont comparées pour le seuil, Isolation Forest, Siamese et TCN; six pour CUSUM. PR-AUC Validation est le critère fixé à l'avance, adapté aux 2–3 % de positifs. Les seuils de décision sont aussi choisis sur Validation. Test n'est calculé qu'après gel des choix.

## Résultats Test

{md_table(test)}

Le Siamese CNN est nettement premier: ROC-AUC {best.roc_auc:.3f}, PR-AUC {best.pr_auc:.3f}, F1 {best.f1:.3f}. Il détecte {int(best.tp)} des 237 positifs, avec {int(best.fp)} faux positifs et {int(best.fn)} faux négatifs au seuil fixé sur Validation. Le TCN classe mieux que les méthodes simples mais son seuil Validation produit {int(test.loc[test.model.eq('tcn'),'fp'].iloc[0])} faux positifs. L'Isolation Forest est mal adaptée à cette représentation: une anomalie radiométrique n'est pas équivalente à un retrait OQDS.

Les écarts Validation/Test et les performances modestes des baselines s'expliquent notamment par la rareté et l'agrégation spatiale des cas, la radiométrie 8 bits, trois dates seulement, et le fait que le changement d'apparence n'est pas spécifique à la maladie. Les erreurs ne prouvent pas une cause unique; `error_analysis.csv` compare seulement leur proportion de pixels noirs.

## Expériences complémentaires réelles

- **NEON**: {complementary['NEON']['individuals_with_multi_year_observations']} individus disposent d'au moins deux années mesurables. {neon_change_text} Les répétitions présentes ne justifient donc ni modèle temporel ni fusion avec OQDS.
- **Hyperspectral**: régression logistique après standardisation/PCA dans une CV stratifiée à trois plis, {complementary['Hyperspectral']['observations']} observations et seulement {complementary['Hyperspectral']['positives']} positives. PR-AUC hors-fold = {complementary['Hyperspectral']['pr_auc_cv']:.3f}; résultat exploratoire, non comparable à OQDS. Les huit cubes n'ont pas de clé permettant une correspondance fiable avec les 76 lignes.
- **CrownsDb**: {complementary['CrownsDb']['boxes']} boîtes vérifiées; aire relative médiane {complementary['CrownsDb']['median_relative_box_area']:.4f}. Usage justifié: détection/pré-entraînement de couronnes, pas classification temporelle.

## Recommandation pour l'application JEE

Retenir le **Siamese CNN** comme moteur principal, avec un service Python d'inférence séparé exposé à JEE par REST. Le service doit recevoir deux patches RGB géoréférencés cohérents, appliquer la normalisation sauvegardée, renvoyer score, classe et version du modèle. Conserver le seuil d'intensité comme contrôle explicable/fallback, jamais comme équivalent de précision. Avant production, valider sur une nouvelle exploitation avec un test externe et, idéalement, de vrais rasters NDVI/NIR calibrés et une vérité terrain indépendante.

Le TCN peut être conservé comme modèle secondaire très rapide, mais sa précision actuelle est insuffisante pour déclencher seul une alerte. Toute nouvelle région doit provoquer une validation de dérive, pas un ajustement sur le Test actuel.

## Reproductibilité et limites

Graine {SEED}; Python local `.venv`; PyTorch CPU; prédictions individuelles et configurations complètes enregistrées. Les modèles neuronaux ont leurs pertes Train/Validation, ROC-AUC et PR-AUC par époque. Les limites majeures sont la cible indirecte, l'absence de NDVI physique, seulement trois dates, la dépendance spatiale résiduelle entre blocs voisins et l'extrême déséquilibre.
"""
    (RESULTS / "FINAL_SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(complementary, indent=2, ensure_ascii=False))
    print("Rapport final:", RESULTS / "FINAL_SCIENTIFIC_REPORT.md")


if __name__ == "__main__":
    main()
