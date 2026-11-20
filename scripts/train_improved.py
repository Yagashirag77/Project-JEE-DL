"""OLV-6 — run amélioré du Siamese CNN, isolé dans outputs/runs/<run_id>/.

Corrige les faiblesses identifiées sur le run initial, sans rien écraser :
1. Sous-entraînement : jusqu'à 40 époques, early stopping sur PR-AUC Validation,
   ReduceLROnPlateau, BatchNorm, augmentation (rot90/flip identiques aux deux dates
   + jitter radiométrique indépendant par date).
2. Déséquilibre : comparaison BCE pondérée vs focal loss.
3. Contexte : patches 16×16 (identiques au run initial) vs 32×32 (16 m).
4. Variance : la config retenue sur Validation est ré-entraînée avec 5 graines ;
   moyenne ± écart-type, ensemble (moyenne des probabilités) et IC95 par bootstrap
   de blocs spatiaux sur Test.
5. Seuils : choisis sur Validation uniquement — F1 optimal et rappel cible 60 %.

Le partitionnement (oqds_features_splits.csv) est réutilisé tel quel.
Usage : python scripts/train_improved.py [run_id]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from torch import nn
import torch.nn.functional as F

SEED = 20261106
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "improved_v1"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "repository"
PROC = ROOT / "data" / "processed"
RUN_DATA = PROC / "runs" / RUN_ID
RUN = ROOT / "outputs" / "runs" / RUN_ID
MODELS, RESULTS, FIGURES = RUN / "models", RUN / "results", RUN / "figures"
ORIG_RESULTS = ROOT / "outputs" / "results"
for p in (RUN_DATA, MODELS, RESULTS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

BIG_PATCH = 32
MAX_EPOCHS = 40
PATIENCE = 7
BATCH = 256
N_SEEDS = 5
TARGET_RECALL = 0.60
N_BOOT = 1000
torch.set_num_threads(max(1, min(16, torch.get_num_threads())))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- métriques
def metrics(y, score, threshold):
    pred = (score >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "threshold": float(threshold),
    }


def threshold_f1(y, score):
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.nanargmax(f1))])


def threshold_recall(y, score, target):
    """Plus haut seuil dont le rappel Validation atteint la cible."""
    p, r, t = precision_recall_curve(y, score)
    ok = np.flatnonzero(r[:-1] >= target)
    return float(t[ok[-1]]) if len(ok) else float(t[0])


def block_bootstrap(y, scores: dict, blocks, n_boot=N_BOOT, seed=SEED):
    """IC95 de PR-AUC/ROC-AUC par ré-échantillonnage de blocs spatiaux (respecte l'autocorrélation)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    idx_by_block = {b: np.flatnonzero(blocks == b) for b in uniq}
    out = {k: {"pr_auc": [], "roc_auc": []} for k in scores}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_block[b] for b in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for k, s in scores.items():
            out[k]["pr_auc"].append(average_precision_score(yb, s[idx]))
            out[k]["roc_auc"].append(roc_auc_score(yb, s[idx]))
    return {k: {m: np.asarray(v) for m, v in d.items()} for k, d in out.items()}


# --------------------------------------------------------------------------- patches 32×32
def extract_big_patches(qc: pd.DataFrame) -> dict[int, Path]:
    paths = {}
    for year in (2019, 2021):
        out = RUN_DATA / f"oqds_rgb{BIG_PATCH}_{year}.npy"
        paths[year] = out
        if out.exists():
            continue
        print(f"Extraction patches {BIG_PATCH}×{BIG_PATCH} {year}…", flush=True)
        with rasterio.open(RAW / "raster_data" / f"{year}_WV2_RGB.tif") as src:
            rgb = src.read([1, 2, 3])
        h, w = rgb.shape[1:]
        half = BIG_PATCH // 2
        off = np.arange(-half, half)
        rows, cols = qc.row.to_numpy(int), qc["col"].to_numpy(int)
        arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(len(qc), 3, BIG_PATCH, BIG_PATCH))
        for lo in range(0, len(qc), 8192):
            r = np.clip(rows[lo:lo + 8192, None, None] + off[None, :, None], 0, h - 1)
            c = np.clip(cols[lo:lo + 8192, None, None] + off[None, None, :], 0, w - 1)
            arr[lo:lo + 8192] = rgb[:, r, c].transpose(1, 0, 2, 3)  # bords : réplication
        arr.flush()
        del rgb, arr
    return paths


# --------------------------------------------------------------------------- modèles
class SiameseOriginal(nn.Module):
    """Architecture du run initial (contrôle : effet des seules époques/early stopping)."""
    def __init__(self, width=16, dropout=0.35):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Linear(width * 4, width * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width * 2, 1))

    def forward(self, a, b):
        za, zb = self.encoder(a).flatten(1), self.encoder(b).flatten(1)
        return self.head(torch.cat([torch.abs(za - zb), za * zb], 1)).squeeze(1)


def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SiameseV2(nn.Module):
    def __init__(self, width=16, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(3, width), nn.MaxPool2d(2),
            conv_block(width, width * 2), nn.MaxPool2d(2),
            conv_block(width * 2, width * 4), nn.AdaptiveAvgPool2d(1))
        d = width * 4
        self.head = nn.Sequential(nn.Linear(d * 3, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, a, b):
        za, zb = self.encoder(a).flatten(1), self.encoder(b).flatten(1)
        return self.head(torch.cat([torch.abs(za - zb), za * zb, zb - za], 1)).squeeze(1)


def focal_loss(logits, y, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * y + (1 - p) * (1 - y)
    a = alpha * y + (1 - alpha) * (1 - y)
    return (a * (1 - pt) ** gamma * bce).mean()


# --------------------------------------------------------------------------- données en mémoire
class Pairs:
    def __init__(self, a19, a21, idx, y, mean, std, patch):
        crop = slice((BIG_PATCH - patch) // 2, (BIG_PATCH + patch) // 2)
        self.a = torch.from_numpy(np.ascontiguousarray(a19[idx][:, :, crop, crop]))
        self.b = torch.from_numpy(np.ascontiguousarray(a21[idx][:, :, crop, crop]))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def norm(self, x):
        return (x.float() / 255.0 - self.mean) / self.std

    def batches(self, shuffle=False, augment=False, gen=None):
        n = len(self.y)
        order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
        for lo in range(0, n, BATCH):
            j = order[lo:lo + BATCH]
            a, b = self.norm(self.a[j]), self.norm(self.b[j])
            if augment:
                a, b = augment_pair(a, b, gen)
            yield a, b, self.y[j]


def augment_pair(a, b, gen):
    """Même rotation/flip géométrique aux deux dates ; jitter radiométrique indépendant par date."""
    n = a.shape[0]
    k = torch.randint(0, 4, (n,), generator=gen)
    flip = torch.rand(n, generator=gen) < 0.5
    out_a, out_b = a.clone(), b.clone()
    for kk in range(4):
        for ff in (False, True):
            m = (k == kk) & (flip == ff)
            if not m.any():
                continue
            xa, xb = a[m], b[m]
            if ff:
                xa, xb = xa.flip(-1), xb.flip(-1)
            out_a[m], out_b[m] = torch.rot90(xa, kk, (-2, -1)), torch.rot90(xb, kk, (-2, -1))
    for x in (out_a, out_b):
        gain = 1 + 0.1 * (torch.rand(n, 1, 1, 1, generator=gen) - 0.5) * 2
        bias = 0.1 * (torch.rand(n, 1, 1, 1, generator=gen) - 0.5) * 2
        x.mul_(gain).add_(bias)
    return out_a, out_b


def predict(model, data: Pairs):
    model.eval(); out = []
    with torch.inference_mode():
        for a, b, _ in data.batches():
            out.append(torch.sigmoid(model(a.to(DEVICE), b.to(DEVICE))).cpu().numpy())
    return np.concatenate(out)


def train(cfg, data, y_val, seed, ckpt):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = (SiameseOriginal(16, 0.35) if cfg["arch"] == "original" else SiameseV2(cfg["width"], cfg["dropout"])).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    ytr = data["train"].y.numpy()
    if cfg["loss"] == "focal":
        loss_fn = focal_loss
    else:
        pw = (ytr == 0).sum() / (ytr == 1).sum()
        pw = np.sqrt(pw) if cfg["loss"] == "bce_sqrtpw" else pw
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pw), device=DEVICE))
        loss_fn = bce
    hist, best, best_ep, t0 = [], -1.0, 0, time.perf_counter()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); losses = []
        for a, b, yb in data["train"].batches(shuffle=True, augment=cfg["augment"], gen=gen):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(a.to(DEVICE), b.to(DEVICE)), yb.to(DEVICE))
            loss.backward(); opt.step(); losses.append(loss.item())
        sv = predict(model, data["val"])
        ap, auc = average_precision_score(y_val, sv), roc_auc_score(y_val, sv)
        sched.step(ap)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_pr_auc": ap, "val_roc_auc": auc,
                     "lr": opt.param_groups[0]["lr"], "elapsed_s": time.perf_counter() - t0})
        print(f"  [{cfg['name']} s{seed}] ep{ep:02d} loss={np.mean(losses):.4f} valAP={ap:.4f} valAUC={auc:.4f}", flush=True)
        if ap > best:
            best, best_ep = ap, ep
            torch.save(model.state_dict(), ckpt)
        elif ep - best_ep >= PATIENCE:
            break
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    return model, pd.DataFrame(hist), best, best_ep, time.perf_counter() - t0


# --------------------------------------------------------------------------- main
def main():
    t_start = time.perf_counter()
    qc = pd.read_csv(PROC / "oqds_qc_all.csv", usecols=["row", "col"])
    frame = pd.read_csv(PROC / "oqds_features_splits.csv")
    parts = {s: frame[frame.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    y = {s: parts[s].target.to_numpy(np.uint8) for s in parts}

    paths = extract_big_patches(qc)
    a19, a21 = np.load(paths[2019], mmap_mode="r"), np.load(paths[2021], mmap_mode="r")
    # contrôle : le recadrage central 16×16 doit être identique aux patches du run initial
    orig19 = np.load(PROC / "oqds_rgb_2019.npy", mmap_mode="r")
    chk = parts["train"].patch_index.to_numpy(int)[:2000]
    assert np.array_equal(a19[chk][:, :, 8:24, 8:24], orig19[chk]), "Recadrage 16×16 ≠ patches initiaux"

    # normalisation Train-only (sur la fenêtre 32×32)
    tr = parts["train"].patch_index.to_numpy(int)
    s1 = np.zeros(3); s2 = np.zeros(3); cnt = 0
    for arr in (a19, a21):
        for lo in range(0, len(tr), 4096):
            z = arr[tr[lo:lo + 4096]].astype(np.float64) / 255.0
            s1 += z.sum((0, 2, 3)); s2 += (z * z).sum((0, 2, 3)); cnt += z.shape[0] * z.shape[2] * z.shape[3]
    mean = s1 / cnt; std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 1e-6))
    np.savez(MODELS / "rgb_normalization_train_only.npz", mean=mean, std=std)

    datasets = {}
    def get_data(patch):
        if patch not in datasets:
            datasets[patch] = {s: Pairs(a19, a21, parts[s].patch_index.to_numpy(int), y[s], mean, std, patch) for s in parts}
        return datasets[patch]

    configs = [
        {"name": "orig_longer",   "arch": "original", "patch": 16, "loss": "bce_pw",     "augment": False, "lr": 7e-4},
        {"name": "v2_p16_bce",    "arch": "v2", "patch": 16, "loss": "bce_sqrtpw", "augment": True, "lr": 1e-3, "width": 16, "dropout": 0.3},
        {"name": "v2_p32_bce",    "arch": "v2", "patch": 32, "loss": "bce_sqrtpw", "augment": True, "lr": 1e-3, "width": 16, "dropout": 0.3},
        {"name": "v2_p32_focal",  "arch": "v2", "patch": 32, "loss": "focal",      "augment": True, "lr": 1e-3, "width": 16, "dropout": 0.3},
    ]

    # Étape 1 — sélection sur Validation (graine de base)
    rows, histories = [], []
    for cfg in configs:
        print(f"== Config {cfg['name']}", flush=True)
        data = get_data(cfg["patch"])
        _, hist, best, best_ep, dur = train(cfg, data, y["val"], SEED, MODELS / f"{cfg['name']}_seed0.pt")
        histories.append(hist.assign(config=cfg["name"], seed=0))
        rows.append({**cfg, "val_pr_auc": best, "best_epoch": best_ep, "epochs_run": len(hist), "train_time_s": dur})
    val_df = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
    val_df.to_csv(RESULTS / "validation_selection.csv", index=False)
    chosen = next(c for c in configs if c["name"] == val_df.iloc[0]["name"])
    print("Config retenue sur Validation :", chosen["name"], flush=True)

    # Étape 2 — graines supplémentaires pour la config retenue
    data = get_data(chosen["patch"])
    seed_scores = {"val": [], "test": []}
    models = []
    for k in range(N_SEEDS):
        ckpt = MODELS / f"{chosen['name']}_seed{k}.pt"
        if k == 0:
            model = (SiameseOriginal(16, 0.35) if chosen["arch"] == "original" else SiameseV2(chosen["width"], chosen["dropout"])).to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        else:
            model, hist, *_ = train(chosen, data, y["val"], SEED + 1000 * k, ckpt)
            histories.append(hist.assign(config=chosen["name"], seed=k))
        models.append(model)
        seed_scores["val"].append(predict(model, data["val"]))
    pd.concat(histories, ignore_index=True).to_csv(RESULTS / "training_histories.csv", index=False)

    # Seuils figés sur Validation AVANT tout calcul Test
    ens_val = np.mean(seed_scores["val"], axis=0)
    thr = {"f1_opt": threshold_f1(y["val"], ens_val), f"recall_{int(TARGET_RECALL*100)}": threshold_recall(y["val"], ens_val, TARGET_RECALL)}
    seed_thr = [threshold_f1(y["val"], s) for s in seed_scores["val"]]
    frozen = {"run_id": RUN_ID, "chosen_config": chosen, "n_seeds": N_SEEDS, "ensemble_thresholds": thr,
              "per_seed_f1_thresholds": seed_thr, "val_ensemble": metrics(y["val"], ens_val, thr["f1_opt"])}
    (RESULTS / "frozen_choices.json").write_text(json.dumps(frozen, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    torch.save({"state_dicts": [m.state_dict() for m in models], "config": chosen, "rgb_mean": mean, "rgb_std": std,
                "thresholds": thr}, MODELS / "siamese_ensemble_best.pt")

    # Étape 3 — Test (une seule fois)
    for m in models:
        seed_scores["test"].append(predict(m, data["test"]))
    ens_test = np.mean(seed_scores["test"], axis=0)
    per_seed = pd.DataFrame([{"seed": k, **metrics(y["test"], s, seed_thr[k])} for k, s in enumerate(seed_scores["test"])])
    per_seed.to_csv(RESULTS / "test_per_seed.csv", index=False)

    orig = pd.read_csv(ORIG_RESULTS / "test_predictions_all_models.csv")
    orig_scores = {m: g.set_index("tree_id").loc[parts["test"].tree_id, "score"].to_numpy() for m, g in orig.groupby("model")}
    orig_thr = {m: float(json.loads((ORIG_RESULTS / "selected_configurations.json").read_text(encoding="utf-8"))[m]["threshold"]) for m in orig_scores}

    table = []
    for name, thr_val in thr.items():
        table.append({"model": f"siamese_v2_ensemble@{name}", **metrics(y["test"], ens_test, thr_val)})
    for m, s in orig_scores.items():
        table.append({"model": f"original_{m}", **metrics(y["test"], s, orig_thr[m])})
    table = pd.DataFrame(table).sort_values("pr_auc", ascending=False)
    table.to_csv(RESULTS / "model_comparison_test.csv", index=False)

    blocks = parts["test"].spatial_block.to_numpy()
    boot = block_bootstrap(y["test"], {"ensemble": ens_test, **{f"original_{m}": s for m, s in orig_scores.items()}}, blocks)
    ci = []
    for k, d in boot.items():
        ci.append({"model": k, "pr_auc_lo": np.percentile(d["pr_auc"], 2.5), "pr_auc_hi": np.percentile(d["pr_auc"], 97.5),
                   "roc_auc_lo": np.percentile(d["roc_auc"], 2.5), "roc_auc_hi": np.percentile(d["roc_auc"], 97.5)})
    diff = boot["ensemble"]["pr_auc"] - boot["original_siamese_cnn"]["pr_auc"]
    ci_df = pd.DataFrame(ci); ci_df.to_csv(RESULTS / "bootstrap_ci_test.csv", index=False)
    diff_summary = {"delta_pr_auc_mean": float(diff.mean()), "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5)),
                    "p_improvement": float((diff > 0).mean()), "n_boot_valid": int(len(diff))}

    pd.DataFrame({"tree_id": parts["test"].tree_id, "y_true": y["test"], "score_ensemble": ens_test,
                  **{f"score_seed{k}": s for k, s in enumerate(seed_scores["test"])},
                  "spatial_block": blocks}).to_csv(RESULTS / "test_predictions.csv", index=False)

    # Figures
    hist_all = pd.concat(histories, ignore_index=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for (c, s), h in hist_all.groupby(["config", "seed"]):
        ax[0].plot(h.epoch, h.val_pr_auc, label=f"{c} s{s}", alpha=.8)
        ax[1].plot(h.epoch, h.train_loss, label=f"{c} s{s}", alpha=.8)
    ax[0].set(title="PR-AUC Validation", xlabel="Époque"); ax[1].set(title="Perte Train", xlabel="Époque")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGURES / "learning_curves.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, s in [("Siamese v2 ensemble", ens_test), ("Siamese original", orig_scores["siamese_cnn"]), ("TCN original", orig_scores["tcn"])]:
        p, r, _ = precision_recall_curve(y["test"], s); ax.plot(r, p, label=f"{name} (AP={average_precision_score(y['test'], s):.3f})")
    ax.axhline(y["test"].mean(), color="k", ls="--", alpha=.5, label="hasard")
    ax.set(xlabel="Rappel", ylabel="Précision", title="Précision–rappel — Test spatial"); ax.legend(); fig.tight_layout()
    fig.savefig(FIGURES / "pr_curves_original_vs_improved.png", dpi=160); plt.close(fig)

    # Rapport
    def fmt(d):
        d = d.round(4).astype(str)
        return "\n".join(["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)] +
                         ["| " + " | ".join(r) + " |" for r in d.to_numpy()])
    ps = per_seed[["pr_auc", "roc_auc", "f1"]].agg(["mean", "std"]).round(4)
    report = [
        f"# OLV-6 — run amélioré `{RUN_ID}`", "",
        f"Appareil `{DEVICE}`, graine de base {SEED}, durée totale {(time.perf_counter()-t_start)/60:.1f} min. Partition spatiale identique au run initial. "
        "Sélection et seuils fixés sur Validation, Test calculé une fois après gel (`frozen_choices.json`).", "",
        "## 1. Sélection sur Validation", "", fmt(val_df[["name", "patch", "loss", "augment", "val_pr_auc", "best_epoch", "epochs_run", "train_time_s"]]), "",
        f"Retenue : **{chosen['name']}** (rappel : Siamese initial, PR-AUC Validation 0.1665 après 6 époques).", "",
        f"## 2. Stabilité sur {N_SEEDS} graines (Test, seuil F1 propre à chaque graine)", "", fmt(per_seed[["seed", "pr_auc", "roc_auc", "f1", "precision", "recall", "tp", "fp"]]), "",
        f"Moyenne ± écart-type : PR-AUC {ps.loc['mean','pr_auc']} ± {ps.loc['std','pr_auc']}, ROC-AUC {ps.loc['mean','roc_auc']} ± {ps.loc['std','roc_auc']}, F1 {ps.loc['mean','f1']} ± {ps.loc['std','f1']}.", "",
        "## 3. Comparaison Test (ensemble des graines vs run initial)", "",
        fmt(table[["model", "pr_auc", "roc_auc", "f1", "precision", "recall", "tp", "fp", "fn", "threshold"]]), "",
        f"`@recall_60` : seuil choisi sur Validation pour atteindre ≥ {TARGET_RECALL:.0%} de rappel (mode surveillance). `@f1_opt` : seuil F1 optimal Validation.", "",
        "## 4. Intervalles de confiance (bootstrap de blocs spatiaux, 1000 tirages)", "", fmt(ci_df), "",
        f"Gain PR-AUC ensemble − Siamese initial : {diff_summary['delta_pr_auc_mean']:+.4f} [IC95 {diff_summary['lo']:+.4f} ; {diff_summary['hi']:+.4f}], "
        f"part des tirages en faveur de l'ensemble : {diff_summary['p_improvement']:.1%}.", "",
        "## Limites", "",
        "- Le Test avait déjà été consulté pour le run initial ; ce run n'utilise toutefois que la Validation pour ses choix. Une confirmation sur une zone externe reste nécessaire.",
        "- TCN non modifié : avec 3 dates seulement, la séquence est trop courte pour qu'un modèle temporel apporte davantage.",
        "- La cible reste indirecte (disparition d'arbre) et la radiométrie non calibrée.",
    ]
    (RESULTS / "IMPROVED_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (RESULTS / "summary.json").write_text(json.dumps({"delta_vs_original": diff_summary, "chosen": chosen["name"],
                                                       "total_minutes": (time.perf_counter() - t_start) / 60}, indent=2), encoding="utf-8")
    sys.stdout.buffer.write(("\n".join(report) + "\n").encode("utf-8", "replace"))


if __name__ == "__main__":
    main()
