"""OLV-6 — expériences complémentaires (remarques jury 6, 7, 9, 15, 18, 29).

A. Validation croisée spatiale du Siamese v2 (config retenue v2_p32_bce) pour trois tailles de blocs
   (250 m, 500 m, 1 km) : StratifiedGroupKFold(5) sur les blocs ; pour chaque fold k : test = k,
   validation = (k+1) mod 5, entraînement = le reste. Normalisation recalculée sur l'entraînement de
   chaque partition. Une graine par partition.
B. Variantes sur la partition d'origine (3 graines, mêmes graines que l'ablation) :
   - attention : Siamese v2 + attention canal (SE) et pooling par attention spatiale ;
   - oversample : rééchantillonnage équilibré des classes (BCE sans poids) ;
   - oversample_spatial : rééchantillonnage des positifs pondéré par 1/(nb positifs du bloc),
     pour ne pas sur-représenter les blocs denses.
Historiques complets : perte train/val, PR-AUC train (sous-échantillon)/val.
Reprise automatique : les runs déjà présents dans les CSV sont sautés.
Sorties : outputs/runs/improved_v1/experiments/spatial/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
import torch.nn.functional as F

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_improved as ti  # noqa: E402

OUT = ti.RUN / "experiments" / "spatial"
CKPT = OUT / "models"
CKPT.mkdir(parents=True, exist_ok=True)
CFG = {"name": "v2_p32_bce", "lr": 1e-3, "width": 16, "dropout": 0.3}
SEEDS = [ti.SEED + 1000 * k for k in range(3)]
N_FOLDS = 5


# ----------------------------------------------------------------------------- modèle avec attention
class SE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(4, c // 4), 1), nn.ReLU(inplace=True),
                               nn.Conv2d(max(4, c // 4), c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.f(x)


class SiameseAttn(nn.Module):
    def __init__(self, width=16, dropout=0.3):
        super().__init__()
        w = width
        self.body = nn.Sequential(ti.conv_block(3, w), SE(w), nn.MaxPool2d(2),
                                  ti.conv_block(w, 2 * w), SE(2 * w), nn.MaxPool2d(2),
                                  ti.conv_block(2 * w, 4 * w), SE(4 * w))
        self.att = nn.Conv2d(4 * w, 1, 1)
        d = 4 * w
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def encode(self, x):
        f = self.body(x)
        a = torch.softmax(self.att(f).flatten(2), -1)          # attention spatiale (B,1,HW)
        return (f.flatten(2) * a).sum(-1)

    def forward(self, a, b):
        za, zb = self.encode(a), self.encode(b)
        return self.head(torch.cat([torch.abs(za - zb), za * zb, zb - za], 1)).squeeze(1)


# ----------------------------------------------------------------------------- données
class Data:
    def __init__(self, A19, A21, idx, y, mean, std):
        self.a, self.b = A19[idx], A21[idx]
        self.y = torch.from_numpy(y.astype(np.float32))
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def norm(self, x):
        return (x.float() / 255.0 - self.mean) / self.std

    def batches(self, order):
        for lo in range(0, len(order), ti.BATCH):
            j = order[lo:lo + ti.BATCH]
            yield self.norm(self.a[j]), self.norm(self.b[j]), self.y[j]


def norm_stats(A19, A21, idx):
    s1 = np.zeros(3); s2 = np.zeros(3); cnt = 0
    for arr in (A19, A21):
        for lo in range(0, len(idx), 8192):
            z = arr[idx[lo:lo + 8192]].double() / 255.0
            s1 += z.sum((0, 2, 3)).numpy(); s2 += (z * z).sum((0, 2, 3)).numpy(); cnt += z.shape[0] * z.shape[2] * z.shape[3]
    mean = s1 / cnt
    return mean, np.sqrt(np.maximum(s2 / cnt - mean ** 2, 1e-6))


def predict(model, d: Data, idx=None):
    model.eval(); out = []
    order = torch.arange(len(d.y)) if idx is None else torch.as_tensor(idx)
    with torch.inference_mode():
        for a, b, _ in d.batches(order):
            out.append(model(a, b).numpy())
    return np.concatenate(out)


def train(model, dtr: Data, dva: Data, seed, ckpt, sampling="none", blocks_tr=None, tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    ytr = dtr.y.numpy(); n = len(ytr)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    if sampling == "none":
        pw = float(np.sqrt((ytr == 0).sum() / (ytr == 1).sum()))
        weights = None
    else:
        pw = 1.0
        w = np.where(ytr == 1, (ytr == 0).sum() / (ytr == 1).sum(), 1.0)
        if sampling == "oversample_spatial":
            pos = pd.Series(ytr).groupby(blocks_tr).transform("sum").to_numpy()
            wp = 1.0 / np.maximum(pos, 1)
            wp = wp / wp[ytr == 1].mean()
            w = np.where(ytr == 1, w * wp, w)
        weights = torch.as_tensor(w, dtype=torch.double)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
    rng = np.random.default_rng(seed)
    sub = rng.choice(n, size=min(10000, n), replace=False)      # sous-échantillon train pour PR-AUC train
    yva = dva.y.numpy()
    hist, best, best_ep, t0 = [], -1.0, 0, time.perf_counter()
    for ep in range(1, ti.MAX_EPOCHS + 1):
        model.train(); losses = []
        order = (torch.randperm(n, generator=gen) if weights is None
                 else torch.multinomial(weights, n, replacement=True, generator=gen))
        for a, b, yb in dtr.batches(order):
            a, b = ti.augment_pair(a, b, gen)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(a, b), yb)
            loss.backward(); opt.step(); losses.append(loss.item())
        lv = predict(model, dva); sv = 1 / (1 + np.exp(-lv))
        ap, auc = average_precision_score(yva, sv), roc_auc_score(yva, sv)
        vloss = float(F.binary_cross_entropy_with_logits(torch.from_numpy(lv), torch.from_numpy(yva),
                                                         pos_weight=torch.tensor(pw)).item())
        lt = predict(model, dtr, sub)
        tap = average_precision_score(ytr[sub], lt)
        sched.step(ap)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": vloss, "train_pr_auc": tap,
                     "val_pr_auc": ap, "val_roc_auc": auc, "elapsed_s": time.perf_counter() - t0})
        print(f"  [{tag}] ep{ep:02d} loss={np.mean(losses):.4f} vloss={vloss:.4f} trAP={tap:.3f} valAP={ap:.4f}", flush=True)
        if ap > best:
            best, best_ep = ap, ep
            torch.save(model.state_dict(), ckpt)
        elif ep - best_ep >= ti.PATIENCE:
            break
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    return model, hist, best, best_ep, time.perf_counter() - t0


def evaluate(model, dva, dte, blocks_te):
    sv = 1 / (1 + np.exp(-predict(model, dva))); st = 1 / (1 + np.exp(-predict(model, dte)))
    yv, yt = dva.y.numpy().astype(int), dte.y.numpy().astype(int)
    thr = ti.threshold_f1(yv, sv)
    m = ti.metrics(yt, st, thr)
    return {"val_pr_auc": average_precision_score(yv, sv), "test_pr_auc": m["pr_auc"], "test_roc_auc": m["roc_auc"],
            "test_f1": m["f1"], "test_recall": m["recall"], "test_precision": m["precision"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "threshold": thr}, st


def load_done(path, keys):
    if path.exists():
        df = pd.read_csv(path)
        return df, {tuple(r[k] for k in keys) for _, r in df.iterrows()}
    return pd.DataFrame(), set()


def append(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    t0 = time.perf_counter()
    torch.set_num_threads(max(1, min(16, torch.get_num_threads())))
    frame = pd.read_csv(ti.PROC / "oqds_features_splits.csv")
    pidx = frame.patch_index.to_numpy(int)
    A19 = torch.from_numpy(np.load(ti.RUN_DATA / "oqds_rgb32_2019.npy")[pidx])
    A21 = torch.from_numpy(np.load(ti.RUN_DATA / "oqds_rgb32_2021.npy")[pidx])
    y = frame.target.to_numpy(np.uint8)
    print(f"Données chargées : {len(y)} arbres", flush=True)

    # ------------------------------------------------------------------ A. validation croisée spatiale
    cv_path, hist_path = OUT / "spatial_cv.csv", OUT / "histories.csv"
    _, done = load_done(cv_path, ["block_m", "fold"])
    folds_info = []
    for B in (250, 500, 1000):
        blocks = (np.floor(frame.x / B).astype(int).astype(str) + "_" + np.floor(frame.y / B).astype(int).astype(str)).to_numpy()
        fold = np.full(len(y), -1)
        for k, (_, idx) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=ti.SEED).split(frame, y, blocks)):
            fold[idx] = k
        for k in range(N_FOLDS):
            te, va = np.flatnonzero(fold == k), np.flatnonzero(fold == (k + 1) % N_FOLDS)
            tr = np.flatnonzero((fold != k) & (fold != (k + 1) % N_FOLDS))
            info = {"block_m": B, "fold": k, "n_blocks": len(np.unique(blocks)), "n_test": len(te),
                    "pos_test": int(y[te].sum()), "prev_test_%": 100 * y[te].mean(),
                    "blocks_test": len(np.unique(blocks[te])), "blocks_test_pos": len(np.unique(blocks[te][y[te] == 1]))}
            folds_info.append(info)
            if (B, k) in done:
                print(f"[CV {B} m fold {k}] déjà fait", flush=True); continue
            mean, std = norm_stats(A19, A21, tr)
            dtr, dva, dte = (Data(A19, A21, i, y[i], mean, std) for i in (tr, va, te))
            seed = ti.SEED + 17 * k
            torch.manual_seed(seed)
            model = ti.SiameseV2(CFG["width"], CFG["dropout"])
            tag = f"CV {B}m f{k}"
            print(f"[{tag}] train={len(tr)} val={len(va)} test={len(te)} pos_test={info['pos_test']}", flush=True)
            model, hist, best, be, dur = train(model, dtr, dva, seed, CKPT / f"cv_{B}m_f{k}.pt", tag=tag)
            res, _ = evaluate(model, dva, dte, blocks[te])
            append(cv_path, {**info, **res, "best_epoch": be, "train_time_s": dur})
            for h in hist:
                append(hist_path, {"run": f"cv_{B}m_f{k}", **h})
            print(f"[{tag}] val={res['val_pr_auc']:.4f} test={res['test_pr_auc']:.4f} roc={res['test_roc_auc']:.4f}", flush=True)
    pd.DataFrame(folds_info).drop_duplicates(["block_m", "fold"]).to_csv(OUT / "cv_folds_info.csv", index=False)

    # ------------------------------------------------------------------ B. variantes (partition d'origine)
    var_path = OUT / "variants.csv"
    _, vdone = load_done(var_path, ["variant", "seed"])
    split = frame.split.to_numpy()
    blocks0 = frame.spatial_block.to_numpy()
    tr, va, te = (np.flatnonzero(split == s) for s in ("train", "val", "test"))
    mean, std = norm_stats(A19, A21, tr)
    dtr, dva, dte = (Data(A19, A21, i, y[i], mean, std) for i in (tr, va, te))
    for variant in ("attention", "oversample", "oversample_spatial"):
        for k, seed in enumerate(SEEDS):
            if (variant, k) in vdone:
                print(f"[{variant} s{k}] déjà fait", flush=True); continue
            torch.manual_seed(seed)
            model = SiameseAttn(CFG["width"], CFG["dropout"]) if variant == "attention" else ti.SiameseV2(CFG["width"], CFG["dropout"])
            sampling = "none" if variant == "attention" else variant
            tag = f"{variant} s{k}"
            model, hist, best, be, dur = train(model, dtr, dva, seed, CKPT / f"{variant}_s{k}.pt", sampling, blocks0[tr], tag)
            res, st = evaluate(model, dva, dte, blocks0[te])
            n_params = sum(p.numel() for p in model.parameters())
            append(var_path, {"variant": variant, "seed": k, "n_params": n_params, **res, "best_epoch": be, "train_time_s": dur})
            for h in hist:
                append(hist_path, {"run": f"{variant}_s{k}", **h})
            print(f"[{tag}] val={res['val_pr_auc']:.4f} test={res['test_pr_auc']:.4f}", flush=True)

    # ------------------------------------------------------------------ résumés
    cv = pd.read_csv(cv_path)
    summ = cv.groupby("block_m").agg(val_mean=("val_pr_auc", "mean"), val_std=("val_pr_auc", "std"),
                                     test_mean=("test_pr_auc", "mean"), test_std=("test_pr_auc", "std"),
                                     roc_mean=("test_roc_auc", "mean"), roc_std=("test_roc_auc", "std"),
                                     f1_mean=("test_f1", "mean"), recall_mean=("test_recall", "mean"),
                                     prev_mean=("prev_test_%", "mean"), folds=("fold", "count")).reset_index()
    summ.to_csv(OUT / "spatial_cv_summary.csv", index=False)
    var = pd.read_csv(var_path)
    var.groupby("variant").agg(val_mean=("val_pr_auc", "mean"), val_std=("val_pr_auc", "std"),
                               test_mean=("test_pr_auc", "mean"), test_std=("test_pr_auc", "std"),
                               n_params=("n_params", "first")).reset_index().to_csv(OUT / "variants_summary.csv", index=False)
    print(summ.to_string(), flush=True)
    print(f"Terminé en {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
