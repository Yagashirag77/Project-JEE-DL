"""Optimisation Validation puis évaluation Test unique des cinq pipelines OLV-6."""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


SEED = 20261106
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "outputs" / "models"
FIGURES = ROOT / "outputs" / "figures"
RESULTS = ROOT / "outputs" / "results"
for p in (MODELS, FIGURES, RESULTS):
    p.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = (score >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "threshold": float(threshold),
    }


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))]) if len(thresholds) else 0.5


def baseline_score(frame: pd.DataFrame, mode: str) -> np.ndarray:
    a, b, c = (frame[f"intensity_mean_{y}"].to_numpy() for y in (2019, 2020, 2021))
    if mode == "decline_2019_2021":
        return a - c
    if mode == "max_annual_decline":
        return np.maximum(a - b, b - c)
    if mode == "relative_decline":
        return (a - c) / (np.abs(a) + 1.0)
    raise KeyError(mode)


def cusum_score(frame: pd.DataFrame, relative: bool, k: float) -> np.ndarray:
    x = frame[[f"intensity_mean_{y}" for y in (2019, 2020, 2021)]].to_numpy(float)
    diffs = x[:, :-1] - x[:, 1:]
    if relative:
        diffs = diffs / (np.abs(x[:, :-1]) + 1.0)
    state = np.zeros(len(frame))
    peak = np.zeros(len(frame))
    for t in range(diffs.shape[1]):
        state = np.maximum(0.0, state + diffs[:, t] - k)
        peak = np.maximum(peak, state)
    return peak


class PairDataset(Dataset):
    def __init__(self, indices, labels, a, b, mean, std):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.a, self.b = a, b
        self.mean = mean.astype(np.float32)[:, None, None]
        self.std = std.astype(np.float32)[:, None, None]

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        x1 = ((self.a[j].astype(np.float32) / 255.0 - self.mean) / self.std).astype(np.float32)
        x2 = ((self.b[j].astype(np.float32) / 255.0 - self.mean) / self.std).astype(np.float32)
        return torch.from_numpy(x1), torch.from_numpy(x2), torch.tensor(self.labels[i])


class SiameseCNN(nn.Module):
    def __init__(self, width=12, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Linear(width * 4, width * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width * 2, 1))

    def forward(self, a, b):
        za = self.encoder(a).flatten(1); zb = self.encoder(b).flatten(1)
        return self.head(torch.cat([torch.abs(za - zb), za * zb], dim=1)).squeeze(1)


class SequenceDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]


class Chomp1d(nn.Module):
    def __init__(self, n): super().__init__(); self.n = n
    def forward(self, x): return x[:, :, :-self.n] if self.n else x


class TemporalBlock(nn.Module):
    def __init__(self, inp, out, kernel, dilation, dropout):
        super().__init__(); pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(inp, out, kernel, padding=pad, dilation=dilation), Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out, out, kernel, padding=pad, dilation=dilation), Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()
    def forward(self, x): return torch.relu(self.net(x) + self.skip(x))


class TCN(nn.Module):
    def __init__(self, channels=16, kernel=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(TemporalBlock(4, channels, kernel, 1, dropout), TemporalBlock(channels, channels, kernel, 2, dropout))
        self.head = nn.Linear(channels, 1)
    def forward(self, x): return self.head(self.net(x)[:, :, -1]).squeeze(1)


def predict_torch(model, loader):
    model.eval(); scores = []
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 3:
                a, b, _ = batch; logits = model(a.to(DEVICE), b.to(DEVICE))
            else:
                x, _ = batch; logits = model(x.to(DEVICE))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores), time.perf_counter() - start


def train_neural(model, train_loader, val_loader, y_val, lr, epochs, checkpoint):
    model.to(DEVICE)
    y_train = np.concatenate([batch[-1].numpy() for batch in train_loader]) if False else None
    # pos_weight est transmis via l'attribut du loader pour rester calculé sur Train uniquement.
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(train_loader.pos_weight, device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    history, best_ap, total_train = [], -np.inf, 0.0
    for epoch in range(epochs):
        model.train(); losses = []; start = time.perf_counter()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if len(batch) == 3:
                a, b, yb = batch; logits = model(a.to(DEVICE), b.to(DEVICE))
            else:
                x, yb = batch; logits = model(x.to(DEVICE))
            loss = loss_fn(logits, yb.to(DEVICE)); loss.backward(); optimizer.step(); losses.append(loss.item())
        total_train += time.perf_counter() - start
        val_score, _ = predict_torch(model, val_loader)
        model.eval(); val_losses = []
        with torch.inference_mode():
            for batch in val_loader:
                if len(batch) == 3:
                    a, b, yb = batch; logits = model(a.to(DEVICE), b.to(DEVICE))
                else:
                    x, yb = batch; logits = model(x.to(DEVICE))
                val_losses.append(loss_fn(logits, yb.to(DEVICE)).item())
        ap, auc = average_precision_score(y_val, val_score), roc_auc_score(y_val, val_score)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": float(np.mean(val_losses)), "val_pr_auc": ap, "val_roc_auc": auc})
        if ap > best_ap:
            best_ap = ap
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    return model, pd.DataFrame(history), total_train


def main() -> None:
    frame = pd.read_csv(PROC / "oqds_features_splits.csv")
    parts = {s: frame[frame.split == s].copy().reset_index(drop=True) for s in ("train", "val", "test")}
    y = {s: parts[s].target.to_numpy(np.uint8) for s in parts}
    experiments, chosen, val_prediction_rows, histories = [], {}, [], []

    # 1. Seuil d'intensité
    start = time.perf_counter()
    for mode in ("decline_2019_2021", "max_annual_decline", "relative_decline"):
        score = baseline_score(parts["val"], mode); threshold = best_threshold(y["val"], score)
        m = metrics(y["val"], score, threshold)
        rec = {"model": "threshold", "config": mode, "params": {"mode": mode}, **m}
        experiments.append(rec)
        val_prediction_rows.append(pd.DataFrame({"tree_id": parts["val"].tree_id, "model": "threshold", "config": mode, "y_true": y["val"], "score": score, "y_pred": score >= threshold}))
    tuning_threshold = time.perf_counter() - start
    chosen["threshold"] = max((x for x in experiments if x["model"] == "threshold"), key=lambda z: z["pr_auc"])
    joblib.dump(chosen["threshold"], MODELS / "threshold.joblib")

    # 2. CUSUM
    start = time.perf_counter()
    for relative, ks in ((False, (0.0, 5.0, 10.0)), (True, (0.0, 0.03, 0.06))):
        for k in ks:
            score = cusum_score(parts["val"], relative, k); threshold = best_threshold(y["val"], score)
            m = metrics(y["val"], score, threshold); config = f"relative={relative}_k={k}"
            experiments.append({"model": "cusum", "config": config, "params": {"relative": relative, "k": k}, **m})
            val_prediction_rows.append(pd.DataFrame({"tree_id": parts["val"].tree_id, "model": "cusum", "config": config, "y_true": y["val"], "score": score, "y_pred": score >= threshold}))
    tuning_cusum = time.perf_counter() - start
    chosen["cusum"] = max((x for x in experiments if x["model"] == "cusum"), key=lambda z: z["pr_auc"])
    joblib.dump(chosen["cusum"], MODELS / "cusum.joblib")

    # 3. Isolation Forest, avec normalisation ajustée sur Train uniquement.
    feature_cols = [c for c in frame if c.startswith(("intensity_", "delta_mean_", "relative_delta_"))]
    scaler = StandardScaler().fit(parts["train"][feature_cols])
    joblib.dump({"scaler": scaler, "features": feature_cols}, MODELS / "numeric_scaler.joblib")
    xnum = {s: scaler.transform(parts[s][feature_cols]) for s in parts}
    iso_configs = [
        {"n_estimators": 150, "max_samples": 256, "max_features": 0.7},
        {"n_estimators": 250, "max_samples": 512, "max_features": 1.0},
        {"n_estimators": 350, "max_samples": 1024, "max_features": 0.8},
    ]
    iso_models, start = {}, time.perf_counter()
    normal_train = xnum["train"][y["train"] == 0]
    for i, cfg in enumerate(iso_configs):
        t0 = time.perf_counter()
        model = IsolationForest(**cfg, contamination="auto", random_state=SEED, n_jobs=-1).fit(normal_train)
        fit_time = time.perf_counter() - t0; score = -model.decision_function(xnum["val"]); threshold = best_threshold(y["val"], score)
        m = metrics(y["val"], score, threshold); config = f"iso_{i}"
        rec = {"model": "isolation_forest", "config": config, "params": cfg, "fit_time_s": fit_time, **m}; experiments.append(rec); iso_models[config] = model
        val_prediction_rows.append(pd.DataFrame({"tree_id": parts["val"].tree_id, "model": "isolation_forest", "config": config, "y_true": y["val"], "score": score, "y_pred": score >= threshold}))
    tuning_iso = time.perf_counter() - start
    chosen["isolation_forest"] = max((x for x in experiments if x["model"] == "isolation_forest"), key=lambda z: z["pr_auc"])
    iso_best = iso_models[chosen["isolation_forest"]["config"]]
    joblib.dump(iso_best, MODELS / "isolation_forest.joblib")

    # Préparation RGB Train-only.
    rgb19 = np.load(PROC / "oqds_rgb_2019.npy", mmap_mode="r"); rgb21 = np.load(PROC / "oqds_rgb_2021.npy", mmap_mode="r")
    train_patch_idx = parts["train"].patch_index.to_numpy(int)
    sums = np.zeros(3); sums2 = np.zeros(3); count = 0
    for arr in (rgb19, rgb21):
        for lo in range(0, len(train_patch_idx), 4096):
            z = arr[train_patch_idx[lo:lo+4096]].astype(np.float64) / 255.0
            sums += z.sum(axis=(0, 2, 3)); sums2 += (z * z).sum(axis=(0, 2, 3)); count += z.shape[0] * z.shape[2] * z.shape[3]
    rgb_mean = sums / count; rgb_std = np.sqrt(np.maximum(sums2 / count - rgb_mean ** 2, 1e-6))
    np.savez(MODELS / "rgb_normalization_train_only.npz", mean=rgb_mean, std=rgb_std)

    # 4. Siamese CNN
    pair_ds = {s: PairDataset(parts[s].patch_index, y[s], rgb19, rgb21, rgb_mean, rgb_std) for s in parts}
    siamese_configs = [
        {"lr": 1e-3, "width": 8, "dropout": 0.15},
        {"lr": 4e-4, "width": 12, "dropout": 0.25},
        {"lr": 7e-4, "width": 16, "dropout": 0.35},
    ]
    siamese_models, start = {}, time.perf_counter()
    for i, cfg in enumerate(siamese_configs):
        config = f"siamese_{i}"; torch.manual_seed(SEED + i)
        loaders = {s: DataLoader(pair_ds[s], batch_size=512, shuffle=(s == "train"), num_workers=0) for s in parts}
        loaders["train"].pos_weight = float((y["train"] == 0).sum() / (y["train"] == 1).sum())
        model = SiameseCNN(cfg["width"], cfg["dropout"])
        ckpt = MODELS / f"{config}.pt"; model, hist, fit_time = train_neural(model, loaders["train"], loaders["val"], y["val"], cfg["lr"], 6, ckpt)
        score, _ = predict_torch(model, loaders["val"]); threshold = best_threshold(y["val"], score); m = metrics(y["val"], score, threshold)
        experiments.append({"model": "siamese_cnn", "config": config, "params": cfg, "fit_time_s": fit_time, **m}); siamese_models[config] = (model, loaders)
        hist.assign(model="siamese_cnn", config=config).to_csv(RESULTS / f"history_{config}.csv", index=False); histories.append(hist.assign(model="siamese_cnn", config=config))
        val_prediction_rows.append(pd.DataFrame({"tree_id": parts["val"].tree_id, "model": "siamese_cnn", "config": config, "y_true": y["val"], "score": score, "y_pred": score >= threshold}))
    tuning_siamese = time.perf_counter() - start
    chosen["siamese_cnn"] = max((x for x in experiments if x["model"] == "siamese_cnn"), key=lambda z: z["pr_auc"])
    siamese_best, siamese_loaders = siamese_models[chosen["siamese_cnn"]["config"]]
    torch.save({"state_dict": siamese_best.state_dict(), "config": chosen["siamese_cnn"]["params"], "rgb_mean": rgb_mean, "rgb_std": rgb_std}, MODELS / "siamese_cnn_best.pt")

    # 5. TCN, séquences de quatre statistiques sur trois années.
    seq = np.stack([frame[[f"intensity_{stat}_{year}" for year in (2019, 2020, 2021)]].to_numpy() for stat in ("mean", "std", "p10", "p90")], axis=1)
    train_mask = frame.split.to_numpy() == "train"
    seq_scaler = StandardScaler().fit(seq[train_mask].transpose(0, 2, 1).reshape(-1, 4))
    seq_scaled = seq_scaler.transform(seq.transpose(0, 2, 1).reshape(-1, 4)).reshape(len(frame), 3, 4).transpose(0, 2, 1)
    joblib.dump(seq_scaler, MODELS / "tcn_scaler_train_only.joblib")
    seq_parts = {s: seq_scaled[frame.split.to_numpy() == s] for s in parts}
    seq_ds = {s: SequenceDataset(seq_parts[s], y[s]) for s in parts}
    tcn_configs = [
        {"lr": 1e-3, "channels": 12, "dropout": 0.10, "kernel": 2},
        {"lr": 4e-4, "channels": 20, "dropout": 0.25, "kernel": 2},
        {"lr": 7e-4, "channels": 16, "dropout": 0.35, "kernel": 3},
    ]
    tcn_models, start = {}, time.perf_counter()
    for i, cfg in enumerate(tcn_configs):
        config = f"tcn_{i}"; torch.manual_seed(SEED + 10 + i)
        loaders = {s: DataLoader(seq_ds[s], batch_size=512, shuffle=(s == "train"), num_workers=0) for s in parts}
        loaders["train"].pos_weight = float((y["train"] == 0).sum() / (y["train"] == 1).sum())
        model = TCN(cfg["channels"], cfg["kernel"], cfg["dropout"]); ckpt = MODELS / f"{config}.pt"
        model, hist, fit_time = train_neural(model, loaders["train"], loaders["val"], y["val"], cfg["lr"], 8, ckpt)
        score, _ = predict_torch(model, loaders["val"]); threshold = best_threshold(y["val"], score); m = metrics(y["val"], score, threshold)
        experiments.append({"model": "tcn", "config": config, "params": cfg, "fit_time_s": fit_time, **m}); tcn_models[config] = (model, loaders)
        hist.assign(model="tcn", config=config).to_csv(RESULTS / f"history_{config}.csv", index=False); histories.append(hist.assign(model="tcn", config=config))
        val_prediction_rows.append(pd.DataFrame({"tree_id": parts["val"].tree_id, "model": "tcn", "config": config, "y_true": y["val"], "score": score, "y_pred": score >= threshold}))
    tuning_tcn = time.perf_counter() - start
    chosen["tcn"] = max((x for x in experiments if x["model"] == "tcn"), key=lambda z: z["pr_auc"])
    tcn_best, tcn_loaders = tcn_models[chosen["tcn"]["config"]]
    torch.save({"state_dict": tcn_best.state_dict(), "config": chosen["tcn"]["params"]}, MODELS / "tcn_best.pt")

    pd.concat(val_prediction_rows, ignore_index=True).to_csv(RESULTS / "validation_predictions_all_configs.csv", index=False)
    pd.DataFrame(experiments).to_csv(RESULTS / "validation_experiments.csv", index=False)
    (RESULTS / "selected_configurations.json").write_text(json.dumps(chosen, indent=2, ensure_ascii=False), encoding="utf-8")

    # Test final: appelé une seule fois après gel des cinq choix Validation.
    score_test, infer_times = {}, {}
    t0 = time.perf_counter(); score_test["threshold"] = baseline_score(parts["test"], chosen["threshold"]["params"]["mode"]); infer_times["threshold"] = time.perf_counter() - t0
    t0 = time.perf_counter(); score_test["cusum"] = cusum_score(parts["test"], **chosen["cusum"]["params"]); infer_times["cusum"] = time.perf_counter() - t0
    t0 = time.perf_counter(); score_test["isolation_forest"] = -iso_best.decision_function(xnum["test"]); infer_times["isolation_forest"] = time.perf_counter() - t0
    score_test["siamese_cnn"], infer_times["siamese_cnn"] = predict_torch(siamese_best, siamese_loaders["test"])
    score_test["tcn"], infer_times["tcn"] = predict_torch(tcn_best, tcn_loaders["test"])
    tuning_times = {"threshold": tuning_threshold, "cusum": tuning_cusum, "isolation_forest": tuning_iso, "siamese_cnn": tuning_siamese, "tcn": tuning_tcn}
    result_rows, test_pred_rows = [], []
    for name, score in score_test.items():
        threshold = chosen[name]["threshold"]; m = metrics(y["test"], score, threshold)
        row = {"model": name, **m, "training_tuning_time_s": tuning_times[name], "inference_time_s": infer_times[name], "inference_ms_per_tree": infer_times[name] * 1000 / len(y["test"])}
        result_rows.append(row)
        pred = pd.DataFrame({"tree_id": parts["test"].tree_id, "model": name, "y_true": y["test"], "score": score, "y_pred": (score >= threshold).astype(np.uint8),
                             "x": parts["test"].x, "y": parts["test"].y, "spatial_block": parts["test"].spatial_block,
                             "rgb_black_fraction_2019": parts["test"].rgb_black_fraction_2019, "rgb_black_fraction_2021": parts["test"].rgb_black_fraction_2021})
        pred.to_csv(RESULTS / f"test_predictions_{name}.csv", index=False); test_pred_rows.append(pred)
    results = pd.DataFrame(result_rows).sort_values("pr_auc", ascending=False)
    results.to_csv(RESULTS / "model_comparison_test.csv", index=False)
    all_test = pd.concat(test_pred_rows, ignore_index=True); all_test.to_csv(RESULTS / "test_predictions_all_models.csv", index=False)

    # Figures finales.
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, score in score_test.items():
        fpr, tpr, _ = roc_curve(y["test"], score); ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y['test'], score):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=.5); ax.set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs", title="ROC — Test spatial figé"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES / "roc_curves_test.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    prevalence = y["test"].mean()
    for name, score in score_test.items():
        p, r, _ = precision_recall_curve(y["test"], score); ax.plot(r, p, label=f"{name} ({average_precision_score(y['test'], score):.3f})")
    ax.axhline(prevalence, color="k", ls="--", alpha=.5); ax.set(xlabel="Rappel", ylabel="Précision", title="Précision–rappel — Test spatial figé"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES / "precision_recall_curves_test.png", dpi=180); plt.close(fig)
    for name, score in score_test.items():
        cm = confusion_matrix(y["test"], score >= chosen[name]["threshold"], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4)); im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm): ax.text(j, i, str(v), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Prédit", ylabel="Réel", title=f"Matrice — {name}"); fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(FIGURES / f"confusion_matrix_{name}.png", dpi=180); plt.close(fig)
    selected_hist = pd.concat([h for h in histories if any(h.config.iloc[0] == chosen[m]["config"] for m in ("siamese_cnn", "tcn"))], ignore_index=True)
    selected_hist.to_csv(RESULTS / "learning_curves_selected.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (model, config), h in selected_hist.groupby(["model", "config"]):
        axes[0].plot(h.epoch, h.train_loss, marker="o", label=f"{model} train"); axes[0].plot(h.epoch, h.val_loss, marker="s", ls="--", label=f"{model} val")
        axes[1].plot(h.epoch, h.val_pr_auc, marker="o", label=model)
    axes[0].set(title="Pertes Train / Validation", xlabel="Époque"); axes[1].set(title="PR-AUC Validation", xlabel="Époque"); [a.legend(fontsize=8) for a in axes]; fig.tight_layout(); fig.savefig(FIGURES / "learning_curves_neural.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); results.sort_values("training_tuning_time_s").plot.barh(x="model", y="training_tuning_time_s", ax=ax, legend=False); ax.set(xlabel="Secondes", title="Temps total d'optimisation/entraînement"); fig.tight_layout(); fig.savefig(FIGURES / "training_times.png", dpi=180); plt.close(fig)

    # Erreurs observées, sans causalité inventée.
    errors = []
    for name, grp in all_test.groupby("model"):
        wrong = grp[grp.y_true != grp.y_pred]
        errors.append({"model": name, "errors": len(wrong), "false_positives": int(((grp.y_true == 0) & (grp.y_pred == 1)).sum()),
                       "false_negatives": int(((grp.y_true == 1) & (grp.y_pred == 0)).sum()),
                       "mean_black_2021_errors": float(wrong.rgb_black_fraction_2021.mean()), "mean_black_2021_correct": float(grp[grp.y_true == grp.y_pred].rgb_black_fraction_2021.mean())})
    pd.DataFrame(errors).to_csv(RESULTS / "error_analysis.csv", index=False)
    report = ["# Résultats scientifiques OLV-6", "", f"Exécution: appareil `{DEVICE}`, graine {SEED}. Sélection sur PR-AUC Validation; Test consulté après gel des configurations.", "",
              "## Tableau Test", "", "```text", results.round(4).to_string(index=False), "```", "", "## Limites", "",
              "- La cible OQDS correspond à une disparition/retrait observé, considérée comme infection probable par les auteurs; ce n'est pas un diagnostic PCR individuel.",
              "- Les fichiers `NDVI` locaux sont des rendus RGBA 8 bits. Les méthodes numériques utilisent donc une intensité, pas un NDVI physique.",
              "- Avec seulement trois dates, CUSUM et TCN décrivent une courte trajectoire; une datation fine de rupture n'est pas défendable.",
              "- Les partitions sont séparées par blocs de 250 m, mais des blocs voisins peuvent rester dans des partitions différentes.", "",
              "## Recommandation", "", f"Le meilleur pipeline selon la PR-AUC Test descriptive est **{results.iloc[0].model}**. Le choix de déploiement JEE doit aussi considérer le temps d'inférence et la disponibilité des images bi-temporelles; les scores Test ne doivent pas servir à un nouveau réglage."]
    (RESULTS / "SCIENTIFIC_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
