"""OLV-6 — analyses complémentaires sans ré-entraînement (remarques jury 11, 12, 24, 26).

1. Radiométrie (11) : luminosité, couleur, part de pixels sombres (proxy d'ombre) par année ;
   variation de luminosité 2019→2021 par bloc spatial (différences d'acquisition).
2. Fausses alertes du TCN (24) : taux de FP selon la variation globale d'intensité du bloc,
   comparé au Siamese v2.
3. Robustesse (26) : perturbations synthétiques des patches test (éclairage, gamma, contraste,
   ombre portée, zone masquée, bord noir) appliquées au Siamese v2 ensemble.
4. Architectures (12) : couches, dimensions de sortie, paramètres.
Sorties : outputs/runs/improved_v1/experiments/complementaires/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_improved as ti  # noqa: E402
import train_evaluate as te  # noqa: E402

OUT = ti.RUN / "experiments" / "complementaires"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)


def lum(x):  # x : (N,3,H,W) uint8 -> luminance (N,H,W)
    x = x.astype(np.float32)
    return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]


# ----------------------------------------------------------------------------- 1. radiométrie
def radiometry(frame, arrs):
    rows = []
    neg = frame.target.to_numpy() == 0
    pidx = frame.patch_index.to_numpy(int)
    rng = np.random.default_rng(ti.SEED)
    samp = rng.choice(np.flatnonzero(neg), 20000, replace=False)
    lums = {}
    for yr, A in arrs.items():
        x = A[pidx[samp]]
        L = lum(x); valid = x.max(1) > 0           # exclut les pixels strictement noirs (hors raster)
        Lv = L[valid]
        lums[yr] = L.reshape(len(samp), -1).mean(1)
        rows.append({"annee": yr, "luminance_moy": float(Lv.mean()), "luminance_std": float(Lv.std()),
                     "R": float(x[:, 0][valid].mean()), "G": float(x[:, 1][valid].mean()), "B": float(x[:, 2][valid].mean()),
                     "pixels_sombres_%": float(100 * (Lv < 60).mean()), "pixels_noirs_%": float(100 * (~valid).mean())})
    df = pd.DataFrame(rows); df.to_csv(OUT / "radiometrie_par_annee.csv", index=False)
    # variation de luminosité par bloc (arbres négatifs uniquement : aucun changement réel attendu)
    d = pd.DataFrame({"block": frame.spatial_block.to_numpy()[samp], "d1920": lums[2020] - lums[2019],
                      "d2021": lums[2021] - lums[2020], "d1921": lums[2021] - lums[2019]})
    blk = d.groupby("block").agg(n=("d1921", "size"), d1920=("d1920", "median"), d2021=("d2021", "median"),
                                 d1921=("d1921", "median"))
    blk = blk[blk.n >= 20]
    blk.to_csv(OUT / "radiometrie_par_bloc.csv")
    summ = {f"delta_bloc_{c}": {"mediane": float(blk[c].median()), "p10": float(blk[c].quantile(.1)),
                                "p90": float(blk[c].quantile(.9))} for c in ("d1920", "d2021", "d1921")}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    for yr, c in zip(arrs, ("#4C78A8", "#F58518", "#54A24B")):
        ax[0].hist(lums[yr], bins=80, histtype="step", lw=1.4, color=c, label=str(yr), density=True)
    ax[0].set(xlabel="Luminance moyenne du patch (0–255)", ylabel="Densité", title="Arbres négatifs : luminance par année")
    ax[0].legend()
    ax[1].hist(blk.d1921, bins=50, color="#8FB59A")
    ax[1].axvline(0, color="k", lw=.8)
    ax[1].set(xlabel="Variation médiane de luminance 2019→2021 par bloc", ylabel="Blocs",
              title="Différences d'acquisition entre blocs")
    fig.tight_layout(); fig.savefig(FIG / "radiometrie.png", dpi=170); plt.close(fig)
    return df, summ, blk


# ----------------------------------------------------------------------------- 2. FP du TCN
def tcn_false_alarms(frame):
    tcn = pd.read_csv(ti.ORIG_RESULTS / "test_predictions_tcn.csv")
    v2 = pd.read_csv(ti.RESULTS / "test_predictions.csv")
    thr_v2 = json.loads((ti.RESULTS / "frozen_choices.json").read_text())["ensemble_thresholds"]["f1_opt"]
    te_ = frame[frame.split == "test"].reset_index(drop=True)
    assert np.allclose(te_.tree_id.to_numpy(), tcn.tree_id.to_numpy()) and np.allclose(te_.tree_id, v2.tree_id)
    te_["tcn_pred"] = tcn.y_pred.to_numpy(); te_["v2_pred"] = (v2.score_ensemble.to_numpy() >= thr_v2).astype(int)
    # variation globale d'intensité du bloc, mesurée sur les négatifs (indépendante de l'étiquette de l'arbre)
    negb = te_[te_.target == 0].groupby("spatial_block").delta_mean_2019_2021.median().rename("block_delta")
    te_ = te_.join(negb, on="spatial_block")
    neg = te_[te_.target == 0].copy()
    neg["quartile"] = pd.qcut(neg.block_delta, 4, labels=["Q1 (forte baisse)", "Q2", "Q3", "Q4 (hausse)"])
    tab = neg.groupby("quartile", observed=True).agg(negatifs=("tcn_pred", "size"), delta_median=("block_delta", "median"),
                                                     fp_tcn_pct=("tcn_pred", lambda s: 100 * s.mean()),
                                                     fp_v2_pct=("v2_pred", lambda s: 100 * s.mean())).reset_index()
    tab.to_csv(OUT / "fp_selon_variation_bloc.csv", index=False)
    corr = float(np.corrcoef(neg.delta_mean_2019_2021, tcn[tcn.y_true == 0].score)[0, 1])
    share = float(((neg.tcn_pred == 1) & (neg.delta_mean_2019_2021 < 0)).sum() / max(1, (neg.tcn_pred == 1).sum()))
    return tab, {"corr_score_tcn_delta_intensite_negatifs": corr, "part_fp_tcn_avec_baisse_intensite": share}


# ----------------------------------------------------------------------------- 3. robustesse
def robustness(frame):
    ck = torch.load(ti.MODELS / "siamese_ensemble_best.pt", weights_only=False)
    models = []
    for sd in ck["state_dicts"]:
        m = ti.SiameseV2(ck["config"]["width"], ck["config"]["dropout"]); m.load_state_dict(sd); m.eval(); models.append(m)
    mean = torch.tensor(ck["rgb_mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(ck["rgb_std"], dtype=torch.float32).view(1, 3, 1, 1)
    thr = ck["thresholds"]["f1_opt"]
    te_ = frame[frame.split == "test"]
    idx = te_.patch_index.to_numpy(int); y = te_.target.to_numpy()
    a19 = torch.from_numpy(np.load(ti.RUN_DATA / "oqds_rgb32_2019.npy")[idx]).float() / 255
    a21 = torch.from_numpy(np.load(ti.RUN_DATA / "oqds_rgb32_2021.npy")[idx]).float() / 255
    g = torch.Generator().manual_seed(ti.SEED)
    n = len(y)

    def shadow(x):  # demi-plan assombri (×0,5), orientation aléatoire
        x = x.clone(); H = x.shape[-1]
        k = torch.randint(0, 4, (n,), generator=g)
        mask = torch.zeros(n, 1, H, H); mask[:, :, :, : H // 2] = 1
        for kk in range(4):
            mask[k == kk] = torch.rot90(mask[k == kk], kk, (-2, -1))
        return x * (1 - 0.5 * mask)

    def occlude(x):  # carré 10×10 noir à position aléatoire
        x = x.clone(); H = x.shape[-1]
        r = torch.randint(0, H - 10, (n,), generator=g); c = torch.randint(0, H - 10, (n,), generator=g)
        for i in range(n):
            x[i, :, r[i]:r[i] + 10, c[i]:c[i] + 10] = 0
        return x

    def border(x):  # bord noir de 6 pixels (simule un bord de raster / transparence)
        x = x.clone(); x[..., :6, :] = 0; x[..., :, :6] = 0
        return x

    conds = {
        "référence": (lambda a: a, lambda b: b),
        "2021 plus sombre (×0,8)": (lambda a: a, lambda b: (b * 0.8).clamp(0, 1)),
        "2021 plus claire (×1,2)": (lambda a: a, lambda b: (b * 1.2).clamp(0, 1)),
        "2019 et 2021 ×0,8 (contrôle)": (lambda a: (a * 0.8).clamp(0, 1), lambda b: (b * 0.8).clamp(0, 1)),
        "gamma 2021 = 1,25": (lambda a: a, lambda b: b ** 1.25),
        "contraste 2021 ×0,7": (lambda a: a, lambda b: ((b - b.mean((1, 2, 3), keepdim=True)) * 0.7 + b.mean((1, 2, 3), keepdim=True)).clamp(0, 1)),
        "ombre portée sur 2021": (lambda a: a, shadow),
        "zone masquée 10×10 sur 2021": (lambda a: a, occlude),
        "bord noir 6 px sur 2021": (lambda a: a, border),
        "ombre portée sur 2019": (shadow, lambda b: b),
    }
    rows = []
    for name, (fa, fb) in conds.items():
        xa, xb = (fa(a19) - mean) / std, (fb(a21) - mean) / std
        with torch.inference_mode():
            s = np.mean([np.concatenate([torch.sigmoid(m(xa[i:i + 1024], xb[i:i + 1024])).numpy() for i in range(0, n, 1024)])
                         for m in models], 0)
        pred = s >= thr
        rows.append({"condition": name, "pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
                     "alertes": int(pred.sum()), "vp": int((pred & (y == 1)).sum()), "fp": int((pred & (y == 0)).sum())})
        print(f"  {name}: AP={rows[-1]['pr_auc']:.4f} alertes={rows[-1]['alertes']}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "robustesse_perturbations.csv", index=False)
    return df


# ----------------------------------------------------------------------------- 4. architectures
def summarize(model, inputs, name):
    rows = []
    hooks = []
    def hook(mod, inp, out):
        if len(list(mod.children())) == 0:
            shp = tuple(out.shape[1:]) if torch.is_tensor(out) else ""
            rows.append({"modele": name, "couche": mod.__class__.__name__, "sortie": str(shp),
                         "parametres": sum(p.numel() for p in mod.parameters(recurse=False))})
    for m in model.modules():
        hooks.append(m.register_forward_hook(hook))
    model.eval()
    with torch.inference_mode():
        model(*inputs)
    for h in hooks:
        h.remove()
    df = pd.DataFrame(rows)
    # un seul passage par encodeur (le siamois l'appelle deux fois)
    if "Siamese" in name:
        df = df.drop_duplicates(subset=None).iloc[: len(df)]
    return df, sum(p.numel() for p in model.parameters())


def architectures():
    specs = {
        "Siamese initial (16×16)": (te.SiameseCNN(16, 0.35), (torch.zeros(1, 3, 16, 16), torch.zeros(1, 3, 16, 16))),
        "Siamese v2 (32×32)": (ti.SiameseV2(16, 0.3), (torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 32, 32))),
        "TCN": (te.TCN(12, 2, 0.10), (torch.zeros(1, 4, 3),)),
    }
    out, tot = [], {}
    for name, (m, x) in specs.items():
        df, n = summarize(m, x, name); out.append(df); tot[name] = n
    pd.concat(out).to_csv(OUT / "architectures_couches.csv", index=False)
    return tot


def main():
    frame = pd.read_csv(ti.PROC / "oqds_features_splits.csv")
    arrs = {yr: np.load(ti.RUN_DATA / f"oqds_rgb32_{yr}.npy", mmap_mode="r") for yr in (2019, 2020, 2021)}
    res = {}
    print("1. Radiométrie", flush=True)
    rad, res["radiometrie"], _ = radiometry(frame, arrs)
    print(rad.to_string(), flush=True)
    print("2. Fausses alertes TCN", flush=True)
    tab, res["tcn_fp"] = tcn_false_alarms(frame)
    print(tab.to_string(), flush=True)
    print("3. Robustesse", flush=True)
    robustness(frame)
    print("4. Architectures", flush=True)
    res["parametres"] = architectures()
    (OUT / "resume.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
