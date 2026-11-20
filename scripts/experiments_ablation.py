"""OLV-6 — ablation du Siamese v2 et comparaison 2 dates / 3 dates (remarques 13, 27, et test de raccourci pour 5).

Variantes (patch 32×32, mêmes hyperparamètres que v2_p32_bce, mêmes 3 graines que les seeds 0–2 de référence) :
  ref        : 2019+2021, encodeur partagé, RGB            (checkpoints existants, non ré-entraînés)
  three_dates: 2019+2020+2021, encodeur partagé            (remarque 13)
  no_share   : 2019+2021, deux encodeurs distincts          (partage des poids)
  gray       : 2019+2021, niveaux de gris                   (apport de la couleur RGB)
  only_2021  : image 2021 seule                             (apport de l'information de changement)
  only_2019  : image 2019 seule (avant retrait)             (test de raccourci contextuel / fuite spatiale)

Sorties : outputs/runs/improved_v1/experiments/ablation/. Rien n'est écrasé.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

sys.argv = sys.argv[:1]  # train_improved lit sys.argv[1] comme run_id
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_improved as ti  # noqa: E402

OUT = ti.RUN / "experiments" / "ablation"
CKPT = OUT / "models"
CKPT.mkdir(parents=True, exist_ok=True)
SEEDS = [ti.SEED + 1000 * k for k in range(3)]
DATES = {2019: 0, 2020: 1, 2021: 2}


def extract_2020(qc):
    out = ti.RUN_DATA / f"oqds_rgb{ti.BIG_PATCH}_2020.npy"
    if out.exists():
        return out
    print("Extraction patches 32×32 2020…", flush=True)
    with rasterio.open(ti.RAW / "raster_data" / "2020_WV2_RGB.tif") as src:
        rgb = src.read([1, 2, 3])
    h, w = rgb.shape[1:]
    off = np.arange(-ti.BIG_PATCH // 2, ti.BIG_PATCH // 2)
    rows, cols = qc.row.to_numpy(int), qc["col"].to_numpy(int)
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(len(qc), 3, ti.BIG_PATCH, ti.BIG_PATCH))
    for lo in range(0, len(qc), 8192):
        r = np.clip(rows[lo:lo + 8192, None, None] + off[None, :, None], 0, h - 1)
        c = np.clip(cols[lo:lo + 8192, None, None] + off[None, None, :], 0, w - 1)
        arr[lo:lo + 8192] = rgb[:, r, c].transpose(1, 0, 2, 3)
    arr.flush()
    return out


class Multi:
    """Patches de plusieurs dates en mémoire, normalisés à la volée (stats Train 2019/2021 du run de référence)."""
    def __init__(self, arrays, idx, y, mean, std, gray=False):
        self.x = [torch.from_numpy(np.ascontiguousarray(a[idx])) for a in arrays]
        self.y = torch.from_numpy(y.astype(np.float32))
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.gray = gray

    def batches(self, shuffle=False, augment=False, gen=None):
        n = len(self.y)
        order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
        for lo in range(0, n, ti.BATCH):
            j = order[lo:lo + ti.BATCH]
            xs = [(x[j].float() / 255.0 - self.mean) / self.std for x in self.x]
            if augment:
                xs = augment_multi(xs, gen)
            if self.gray:
                xs = [x.mean(1, keepdim=True) for x in xs]
            yield xs, self.y[j]


def augment_multi(xs, gen):
    n = xs[0].shape[0]
    k = torch.randint(0, 4, (n,), generator=gen)
    flip = torch.rand(n, generator=gen) < 0.5
    outs = [x.clone() for x in xs]
    for kk in range(4):
        for ff in (False, True):
            m = (k == kk) & (flip == ff)
            if not m.any():
                continue
            for x, o in zip(xs, outs):
                z = x[m].flip(-1) if ff else x[m]
                o[m] = torch.rot90(z, kk, (-2, -1))
    for o in outs:
        gain = 1 + 0.2 * (torch.rand(n, 1, 1, 1, generator=gen) - 0.5)
        bias = 0.2 * (torch.rand(n, 1, 1, 1, generator=gen) - 0.5)
        o.mul_(gain).add_(bias)
    return outs


def encoder(in_ch, width):
    return nn.Sequential(ti.conv_block(in_ch, width), nn.MaxPool2d(2), ti.conv_block(width, width * 2), nn.MaxPool2d(2),
                         ti.conv_block(width * 2, width * 4), nn.AdaptiveAvgPool2d(1))


class Variant(nn.Module):
    def __init__(self, n_dates, in_ch=3, shared=True, width=16, dropout=0.3):
        super().__init__()
        self.n, self.shared = n_dates, shared
        self.enc = nn.ModuleList([encoder(in_ch, width)] if shared else [encoder(in_ch, width) for _ in range(n_dates)])
        d = width * 4
        fused = {1: d, 2: 3 * d, 3: 4 * d}[n_dates]
        self.head = nn.Sequential(nn.Linear(fused, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, xs):
        z = [(self.enc[0] if self.shared else self.enc[i])(x).flatten(1) for i, x in enumerate(xs)]
        if self.n == 1:
            f = z[0]
        elif self.n == 2:
            f = torch.cat([torch.abs(z[0] - z[1]), z[0] * z[1], z[1] - z[0]], 1)
        else:
            f = torch.cat([torch.abs(z[1] - z[0]), torch.abs(z[2] - z[1]), torch.abs(z[2] - z[0]), z[0] * z[2]], 1)
        return self.head(f).squeeze(1)


def predict(model, data):
    model.eval(); out = []
    with torch.inference_mode():
        for xs, _ in data.batches():
            out.append(torch.sigmoid(model(xs)).numpy())
    return np.concatenate(out)


def train(name, model, data, y_val, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    ytr = data["train"].y.numpy()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(np.sqrt((ytr == 0).sum() / (ytr == 1).sum()))))
    ckpt = CKPT / f"{name}_seed{SEEDS.index(seed)}.pt"
    best, best_ep, hist, t0 = -1, 0, [], time.perf_counter()
    for ep in range(1, ti.MAX_EPOCHS + 1):
        model.train(); losses = []
        for xs, yb in data["train"].batches(shuffle=True, augment=True, gen=gen):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xs), yb); loss.backward(); opt.step(); losses.append(loss.item())
        sv = predict(model, data["val"]); ap = average_precision_score(y_val, sv)
        sched.step(ap)
        hist.append({"variant": name, "seed": SEEDS.index(seed), "epoch": ep, "train_loss": float(np.mean(losses)), "val_pr_auc": ap,
                     "val_roc_auc": roc_auc_score(y_val, sv), "elapsed_s": time.perf_counter() - t0})
        print(f"  [{name} s{SEEDS.index(seed)}] ep{ep:02d} loss={np.mean(losses):.4f} valAP={ap:.4f}", flush=True)
        if ap > best:
            best, best_ep = ap, ep; torch.save(model.state_dict(), ckpt)
        elif ep - best_ep >= ti.PATIENCE:
            break
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    return model, hist, best_ep, time.perf_counter() - t0


def main():
    qc = pd.read_csv(ti.PROC / "oqds_qc_all.csv", usecols=["row", "col"])
    frame = pd.read_csv(ti.PROC / "oqds_features_splits.csv")
    parts = {s: frame[frame.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    y = {s: parts[s].target.to_numpy(np.uint8) for s in parts}
    p2020 = extract_2020(qc)
    arr = {2019: np.load(ti.RUN_DATA / "oqds_rgb32_2019.npy", mmap_mode="r"), 2020: np.load(p2020, mmap_mode="r"),
           2021: np.load(ti.RUN_DATA / "oqds_rgb32_2021.npy", mmap_mode="r")}
    norm = np.load(ti.MODELS / "rgb_normalization_train_only.npz")
    mean, std = norm["mean"], norm["std"]

    variants = {
        "ref":         {"dates": [2019, 2021], "shared": True,  "gray": False},
        "only_2019":   {"dates": [2019],       "shared": True,  "gray": False},
        "only_2021":   {"dates": [2021],       "shared": True,  "gray": False},
        "gray":        {"dates": [2019, 2021], "shared": True,  "gray": True},
        "no_share":    {"dates": [2019, 2021], "shared": False, "gray": False},
        "three_dates": {"dates": [2019, 2020, 2021], "shared": True, "gray": False},
    }
    rows, hists, scores = [], [], {}
    # Reprise : une ligne dans ablation_per_seed.csv = graine terminée (checkpoint final sur disque).
    done = {}
    if (OUT / "ablation_per_seed.csv").exists():
        prev = pd.read_csv(OUT / "ablation_per_seed.csv")
        done = {(r.variant, int(r.seed)): r.to_dict() for _, r in prev.iterrows()}
    if (OUT / "ablation_histories.csv").exists():
        hp = pd.read_csv(OUT / "ablation_histories.csv")
        hists = [r for r in hp.to_dict("records") if (r["variant"], int(r["seed"])) in done]
    for name, v in variants.items():
        data = {s: Multi([arr[d] for d in v["dates"]], parts[s].patch_index.to_numpy(int), y[s], mean, std, v["gray"]) for s in parts}
        scores[name] = {"val": [], "test": []}
        for k, seed in enumerate(SEEDS):
            model = Variant(len(v["dates"]), 1 if v["gray"] else 3, v["shared"])
            if name != "ref" and (name, k) in done:
                print(f"  [{name} s{k}] déjà terminé — rechargement du checkpoint", flush=True)
                model.load_state_dict(torch.load(CKPT / f"{name}_seed{k}.pt", weights_only=True))
                best_ep, dur = done[(name, k)]["best_epoch"], done[(name, k)]["train_time_s"]
            elif name == "ref":
                ref = ti.SiameseV2(16, 0.3)
                ref.load_state_dict(torch.load(ti.MODELS / f"v2_p32_bce_seed{k}.pt", weights_only=True))
                model.enc[0].load_state_dict(ref.encoder.state_dict()); model.head.load_state_dict(ref.head.state_dict())
                best_ep, dur = None, None
            else:
                model, h, best_ep, dur = train(name, model, data, y["val"], seed); hists += h
            sv, st = predict(model, data["val"]), predict(model, data["test"])
            scores[name]["val"].append(sv); scores[name]["test"].append(st)
            rows.append({"variant": name, "seed": k, "n_params": sum(p.numel() for p in model.parameters()),
                         "best_epoch": best_ep, "train_time_s": dur,
                         "val_pr_auc": average_precision_score(y["val"], sv), "val_roc_auc": roc_auc_score(y["val"], sv),
                         "test_pr_auc": average_precision_score(y["test"], st), "test_roc_auc": roc_auc_score(y["test"], st)})
            pd.DataFrame(rows).to_csv(OUT / "ablation_per_seed.csv", index=False)
            if hists:
                pd.DataFrame(hists).to_csv(OUT / "ablation_histories.csv", index=False)
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("variant", sort=False)[["val_pr_auc", "val_roc_auc", "test_pr_auc", "test_roc_auc"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    for name in variants:
        summary.loc[name, "n_params"] = per_seed[per_seed.variant == name].n_params.iloc[0]
        summary.loc[name, "test_pr_auc_ens3"] = average_precision_score(y["test"], np.mean(scores[name]["test"], 0))
        summary.loc[name, "test_roc_auc_ens3"] = roc_auc_score(y["test"], np.mean(scores[name]["test"], 0))
    summary.reset_index().to_csv(OUT / "ablation_summary.csv", index=False)
    np.savez_compressed(OUT / "ablation_scores.npz", **{f"{n}_{s}": np.array(scores[n][s]) for n in scores for s in ("val", "test")})
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
