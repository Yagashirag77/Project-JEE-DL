"""Contrôles finaux indépendants et manifeste des artefacts."""
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    notebook = nbformat.read(ROOT / "OLV6_Comparaison_Complete.ipynb", as_version=4)
    errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert not errors, errors
    assert all(c.execution_count is not None for c in code_cells), "Cellule de code non exécutée"

    features = pd.read_csv(ROOT / "data" / "processed" / "oqds_features_splits.csv")
    assert features.groupby("tree_id").split.nunique().max() == 1
    assert features.groupby("spatial_block").split.nunique().max() == 1
    expected_ids = set(features.loc[features.split == "test", "tree_id"].astype(str))

    comparison = pd.read_csv(RESULTS / "model_comparison_test.csv").set_index("model")
    predictions = pd.read_csv(RESULTS / "test_predictions_all_models.csv", dtype={"tree_id": str})
    checks = []
    for model, z in predictions.groupby("model"):
        assert set(z.tree_id) == expected_ids
        y, score, pred = z.y_true.to_numpy(), z.score.to_numpy(), z.y_pred.to_numpy()
        recomputed = {
            "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score),
        }
        for key, value in recomputed.items():
            assert np.isclose(value, comparison.loc[model, key], atol=1e-12), (model, key, value, comparison.loc[model, key])
        checks.append({"model": model, **recomputed, "n_test": len(z)})

    required = [
        ROOT / "outputs" / "models" / "threshold.joblib", ROOT / "outputs" / "models" / "cusum.joblib",
        ROOT / "outputs" / "models" / "isolation_forest.joblib", ROOT / "outputs" / "models" / "siamese_cnn_best.pt",
        ROOT / "outputs" / "models" / "tcn_best.pt", ROOT / "outputs" / "figures" / "roc_curves_test.png",
        ROOT / "outputs" / "figures" / "precision_recall_curves_test.png", RESULTS / "FINAL_SCIENTIFIC_REPORT.md",
    ]
    assert all(p.exists() and p.stat().st_size > 0 for p in required)
    artifact_files = sorted([p for p in (ROOT / "outputs").rglob("*") if p.is_file()] + list((ROOT / "scripts").glob("*.py")) + [ROOT / "OLV6_Comparaison_Complete.ipynb"])
    manifest = pd.DataFrame([{"path": p.relative_to(ROOT).as_posix(), "size_bytes": p.stat().st_size, "sha256": digest(p)} for p in artifact_files])
    manifest.to_csv(RESULTS / "artifact_manifest.csv", index=False)
    report = {
        "status": "PASS", "notebook_code_cells": len(code_cells), "notebook_errors": 0,
        "spatial_leakage": False, "tree_leakage": False, "test_tree_count": len(expected_ids),
        "metrics_recomputed": checks, "required_artifacts": [p.relative_to(ROOT).as_posix() for p in required],
        "manifest_entries": len(manifest),
    }
    (RESULTS / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
