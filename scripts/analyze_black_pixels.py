"""Analyse post-hoc des pixels noirs du Siamese sur le Test spatial figé, sans réentraînement."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from train_evaluate import SiameseCNN


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"
MODELS = ROOT / "outputs" / "models"
PATCH = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_ndvi(year: int) -> Path:
    matches = sorted((ROOT / "data" / "raw" / "repository").rglob(f"{year}*NDVI.tif"))
    if len(matches) != 1:
        raise RuntimeError(f"Raster alpha {year} ambigu: {matches}")
    return matches[0]


def alpha_masks(path: Path, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Le canal alpha=0 est le seul masque d'invalidité explicite disponible."""
    masks = np.zeros((len(rows), PATCH, PATCH), dtype=bool)
    half = PATCH // 2
    with rasterio.open(path) as src:
        alpha = src.read(4)
        for i, (r, c) in enumerate(zip(rows, cols)):
            masks[i] = alpha[int(r)-half:int(r)+half, int(c)-half:int(c)+half] == 0
    return masks


def model_scores(model, a, b, mean, std, mask_a=None, mask_b=None, batch_size=512):
    scores = []
    neutral = mean[:, None, None].astype(np.float32)
    mean3, std3 = neutral, std[:, None, None].astype(np.float32)
    with torch.inference_mode():
        for lo in range(0, len(a), batch_size):
            hi = min(lo + batch_size, len(a))
            x1 = a[lo:hi].astype(np.float32) / 255.0
            x2 = b[lo:hi].astype(np.float32) / 255.0
            if mask_a is not None:
                x1 = np.where(mask_a[lo:hi, None], neutral[None], x1)
                x2 = np.where(mask_b[lo:hi, None], neutral[None], x2)
            x1 = (x1 - mean3[None]) / std3[None]
            x2 = (x2 - mean3[None]) / std3[None]
            logits = model(torch.from_numpy(x1).to(DEVICE), torch.from_numpy(x2).to(DEVICE))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def metric_row(y, score, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two_classes = np.unique(y).size == 2
    return {
        "n": len(y), "positives": int(np.sum(y)), "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score) if two_classes else np.nan,
        "pr_auc": average_precision_score(y, score) if two_classes else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> None:
    features = pd.read_csv(PROC / "oqds_features_splits.csv", dtype={"tree_id": str})
    test = features.loc[features.split == "test"].copy().reset_index(drop=True)
    saved = pd.read_csv(RESULTS / "test_predictions_siamese_cnn.csv", dtype={"tree_id": str})
    saved = saved.set_index("tree_id").loc[test.tree_id].reset_index()
    if not np.array_equal(saved.y_true.to_numpy(), test.target.to_numpy()):
        raise AssertionError("Labels Test non alignés")

    idx = test.patch_index.to_numpy(int)
    rgb19_all = np.load(PROC / "oqds_rgb_2019.npy", mmap_mode="r")
    rgb21_all = np.load(PROC / "oqds_rgb_2021.npy", mmap_mode="r")
    rgb19, rgb21 = np.asarray(rgb19_all[idx]), np.asarray(rgb21_all[idx])
    black19 = np.all(rgb19 == 0, axis=1); black21 = np.all(rgb21 == 0, axis=1)
    strict19, strict21 = black19.mean(axis=(1, 2)), black21.mean(axis=(1, 2))
    if not np.allclose(strict19, test.rgb_black_fraction_2019, atol=1e-8) or not np.allclose(strict21, test.rgb_black_fraction_2021, atol=1e-8):
        raise AssertionError("La définition du noir ne reproduit pas les fractions préparées")

    invalid19 = alpha_masks(find_ndvi(2019), test.row.to_numpy(int), test.col.to_numpy(int))
    invalid21 = alpha_masks(find_ndvi(2021), test.row.to_numpy(int), test.col.to_numpy(int))
    luminance19 = .2126*rgb19[:, 0] + .7152*rgb19[:, 1] + .0722*rgb19[:, 2]
    luminance21 = .2126*rgb21[:, 0] + .7152*rgb21[:, 1] + .0722*rgb21[:, 2]

    out = pd.DataFrame({
        "tree_id": test.tree_id, "y_true": saved.y_true, "score_original_saved": saved.score,
        "y_pred_original_saved": saved.y_pred, "black_fraction_2019": strict19,
        "black_fraction_2021": strict21, "black_fraction_delta_2021_minus_2019": strict21-strict19,
        "black_fraction_max": np.maximum(strict19, strict21),
        "invalid_alpha_fraction_2019": invalid19.mean(axis=(1, 2)),
        "invalid_alpha_fraction_2021": invalid21.mean(axis=(1, 2)),
        "black_and_invalid_fraction_2019": (black19 & invalid19).mean(axis=(1, 2)),
        "black_and_invalid_fraction_2021": (black21 & invalid21).mean(axis=(1, 2)),
        "black_but_alpha_valid_fraction_2019": (black19 & ~invalid19).mean(axis=(1, 2)),
        "black_but_alpha_valid_fraction_2021": (black21 & ~invalid21).mean(axis=(1, 2)),
        "dark_but_alpha_valid_fraction_2019": ((luminance19 <= 20) & ~invalid19).mean(axis=(1, 2)),
        "dark_but_alpha_valid_fraction_2021": ((luminance21 <= 20) & ~invalid21).mean(axis=(1, 2)),
    })

    # 80,7 % des patches Test n'ont aucun noir strict: terciles ordinaires seraient dégénérés.
    # Les groupes sont donc: aucun noir, moitié basse des non-zéro, moitié haute des non-zéro.
    positive_black = out.loc[out.black_fraction_max > 0, "black_fraction_max"]
    nonzero_median = float(positive_black.median())
    out["black_group"] = np.select(
        [out.black_fraction_max == 0, out.black_fraction_max <= nonzero_median],
        ["none_0", f"nonzero_low_le_{nonzero_median:.6f}"],
        default=f"nonzero_high_gt_{nonzero_median:.6f}",
    )
    out["error_type"] = np.select(
        [(out.y_true == 0) & (out.y_pred_original_saved == 1), (out.y_true == 1) & (out.y_pred_original_saved == 0)],
        ["false_positive", "false_negative"], default="correct",
    )

    checkpoint = torch.load(MODELS / "siamese_cnn_best.pt", map_location=DEVICE, weights_only=False)
    cfg = checkpoint["config"]; mean = np.asarray(checkpoint["rgb_mean"], dtype=np.float32); std = np.asarray(checkpoint["rgb_std"], dtype=np.float32)
    model = SiameseCNN(width=cfg["width"], dropout=cfg["dropout"]).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    recomputed = model_scores(model, rgb19, rgb21, mean, std)
    max_score_difference = float(np.max(np.abs(recomputed - out.score_original_saved.to_numpy())))
    if max_score_difference > 2e-6:
        raise AssertionError(f"Checkpoint non reproductible: écart score max {max_score_difference}")
    sensitivity = model_scores(model, rgb19, rgb21, mean, std, invalid19, invalid21)
    threshold = float(json.loads((RESULTS / "selected_configurations.json").read_text(encoding="utf-8"))["siamese_cnn"]["threshold"])
    out["score_alpha0_neutralized"] = sensitivity
    out["score_change_after_neutralization"] = sensitivity - recomputed
    out["y_pred_alpha0_neutralized"] = (sensitivity >= threshold).astype(np.uint8)
    out["prediction_changed_after_neutralization"] = out.y_pred_alpha0_neutralized != out.y_pred_original_saved
    out.to_csv(RESULTS / "black_pixels_analysis.csv", index=False)

    group_rows = []
    for group, z in out.groupby("black_group", sort=False):
        group_rows.append({"analysis": "original_by_black_group", "group": group,
                           "black_threshold_definition": f"0 puis médiane des valeurs non-nulles={nonzero_median:.6f}",
                           **metric_row(z.y_true.to_numpy(), z.score_original_saved.to_numpy(), z.y_pred_original_saved.to_numpy())})
    for label, score_col, pred_col in [
        ("original_all", "score_original_saved", "y_pred_original_saved"),
        ("alpha0_neutralized_all", "score_alpha0_neutralized", "y_pred_alpha0_neutralized")]:
        group_rows.append({"analysis": "sensitivity_all", "group": label,
                           "black_threshold_definition": "alpha=0 remplacé par moyenne RGB Train; aucun réentraînement",
                           **metric_row(out.y_true.to_numpy(), out[score_col].to_numpy(), out[pred_col].to_numpy())})
    group_metrics = pd.DataFrame(group_rows)
    group_metrics.to_csv(RESULTS / "black_pixels_group_metrics.csv", index=False)

    label_stats = out.groupby("y_true")[["black_fraction_2019", "black_fraction_2021", "black_fraction_delta_2021_minus_2019", "invalid_alpha_fraction_2019", "invalid_alpha_fraction_2021"]].agg(["mean", "median", "std"])
    error_stats = out.groupby("error_type")[["black_fraction_max", "invalid_alpha_fraction_2019", "invalid_alpha_fraction_2021"]].agg(["count", "mean", "median"])
    summary = {
        "n_test": len(out), "strict_black_definition": "R=G=B=0",
        "invalid_definition": "alpha=0 dans le raster RGBA *NDVI aligné; proxy de transparence/nodata, pas masque physique certifié",
        "group_rule": f"aucun noir; puis <= ou > médiane des proportions non-nulles ({nonzero_median:.6f})",
        "zero_black_share": float((out.black_fraction_max == 0).mean()),
        "max_abs_difference_recomputed_vs_saved": max_score_difference,
        "changed_predictions_after_neutralization": int(out.prediction_changed_after_neutralization.sum()),
        "mean_abs_score_change_after_neutralization": float(out.score_change_after_neutralization.abs().mean()),
        "label_distributions": json.loads(label_stats.to_json()),
        "error_distributions": json.loads(error_stats.to_json()),
    }
    (RESULTS / "black_pixels_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    bins = np.linspace(0, .80, 33)
    for label, color in [(0, "#557a46"), (1, "#b84a3a")]:
        axes[0, 0].hist(out.loc[out.y_true == label, "black_fraction_max"], bins=bins, density=True, alpha=.55, color=color, label=f"classe {label}")
    axes[0, 0].axvline(nonzero_median, color="black", ls="--", label="médiane non-zéro")
    axes[0, 0].set(title="Noir strict maximal par classe", xlabel="Proportion max(2019, 2021)", ylabel="Densité"); axes[0, 0].legend()
    data_box = [out.loc[out.y_true == v, "black_fraction_delta_2021_minus_2019"] for v in (0, 1)]
    axes[0, 1].boxplot(data_box, tick_labels=["négatifs", "positifs"], showfliers=False)
    axes[0, 1].axhline(0, color="black", lw=.8); axes[0, 1].set(title="Variation du noir 2021 − 2019", ylabel="Différence de proportion")
    order = [x for x in ("correct", "false_positive", "false_negative") if x in set(out.error_type)]
    axes[1, 0].boxplot([out.loc[out.error_type == x, "black_fraction_max"] for x in order], tick_labels=order, showfliers=False)
    axes[1, 0].set(title="Noir strict et erreurs", ylabel="Proportion maximale")
    axes[1, 1].scatter(recomputed, sensitivity, c=out.invalid_alpha_fraction_2019 + out.invalid_alpha_fraction_2021, s=8, alpha=.45, cmap="viridis")
    axes[1, 1].plot([0, 1], [0, 1], "k--", lw=.8); axes[1, 1].axvline(threshold, color="red", ls=":"); axes[1, 1].axhline(threshold, color="red", ls=":")
    axes[1, 1].set(title=f"Sensibilité alpha=0 ({int(out.prediction_changed_after_neutralization.sum())} décisions changées)", xlabel="Score original", ylabel="Score après neutralisation")
    fig.suptitle("OLV-6 — analyse post-hoc des pixels noirs du Siamese CNN", fontsize=14)
    fig.tight_layout(); fig.savefig(FIGURES / "black_pixels_analysis.png", dpi=180); plt.close(fig)
    print(group_metrics.to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("zero_black_share", "changed_predictions_after_neutralization", "mean_abs_score_change_after_neutralization")}, indent=2))


if __name__ == "__main__":
    main()
