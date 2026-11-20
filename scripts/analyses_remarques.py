"""OLV-6 — analyses demandées par les remarques du jury (3, 5, 8, 16, 19, 20, 21, 22, 23, 25).

Aucun résultat existant n'est modifié ; sorties dans outputs/runs/improved_v1/experiments/analyses/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_improved as ti  # noqa: E402
import train_evaluate as te  # noqa: E402

ROOT = ti.ROOT
OUT = ti.RUN / "experiments" / "analyses"
FIG = OUT / "figures"
CK = OUT / "models"
for p in (OUT, FIG, CK):
    p.mkdir(parents=True, exist_ok=True)
SEED = ti.SEED
N_SEEDS = 5


def thr_f1(y, s): return ti.threshold_f1(y, s)
def thr_recall(y, s, r): return ti.threshold_recall(y, s, r)
def thr_budget(s, frac): return float(np.quantile(s, 1 - frac))


# ------------------------------------------------------------------ 3. exclusions
def exclusions():
    qc = pd.read_csv(ti.PROC / "oqds_qc_all.csv", usecols=["target", "valid", "invalid_reason", "removal_year"])
    qc["invalid_reason"] = qc.invalid_reason.fillna("").replace("", "conservé")
    tab = qc.groupby(["invalid_reason", "target"]).size().unstack(fill_value=0).rename(columns={0: "negatifs", 1: "positifs"})
    tab["total"] = tab.sum(1); tab["prevalence_%"] = 100 * tab.positifs / tab.total
    chi2, p, *_ = chi2_contingency(pd.crosstab(qc.valid, qc.target))
    res = {"before_prevalence_%": 100 * qc.target.mean(), "after_prevalence_%": 100 * qc[qc.valid].target.mean(),
           "excluded_prevalence_%": 100 * qc[~qc.valid].target.mean(), "excluded_positives": int(qc[~qc.valid].target.sum()),
           "total_positives": int(qc.target.sum()), "chi2": chi2, "p_value": p}
    tab.reset_index().to_csv(OUT / "exclusions_by_class.csv", index=False)
    return tab.reset_index(), res


# ------------------------------------------------------------------ 8. déséquilibre
def imbalance(frame):
    rows = []
    for s, g in frame.groupby("split"):
        b = g.groupby("spatial_block").target.agg(["sum", "size"])
        pos_sorted = np.sort(b["sum"].to_numpy())[::-1]
        top10 = pos_sorted[:max(1, int(np.ceil(0.1 * len(b))))].sum() / max(1, pos_sorted.sum())
        rows.append({"split": s, "arbres": len(g), "positifs": int(g.target.sum()), "prevalence_%": 100 * g.target.mean(),
                     "ratio_neg_pos": (len(g) - g.target.sum()) / g.target.sum(), "blocs": len(b),
                     "blocs_avec_positif": int((b["sum"] > 0).sum()), "part_positifs_top10%_blocs": 100 * top10,
                     "prevalence_max_bloc_%": 100 * (b["sum"] / b["size"]).max()})
    df = pd.DataFrame(rows)
    yr = frame[frame.target == 1].groupby(["split", "removal_year"]).size().unstack(fill_value=0)
    df.to_csv(OUT / "imbalance_by_split.csv", index=False); yr.to_csv(OUT / "positives_by_removal_year.csv")
    # courbe de concentration des positifs par bloc (tous splits)
    b = frame.groupby("spatial_block").target.sum().sort_values(ascending=False).to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(np.arange(1, len(b) + 1) / len(b) * 100, np.cumsum(b) / b.sum() * 100, color="#2F6B3A")
    ax[0].plot([0, 100], [0, 100], "k--", lw=.8)
    ax[0].set(xlabel="% des blocs (triés par nb de positifs)", ylabel="% cumulé des positifs", title="Concentration spatiale des positifs")
    pr = frame.groupby("spatial_block").target.mean() * 100
    ax[1].hist(pr, bins=40, color="#8FB59A"); ax[1].set(xlabel="Prévalence du bloc (%)", ylabel="Nombre de blocs", title="Prévalence par bloc de 250 m")
    fig.tight_layout(); fig.savefig(FIG / "imbalance_blocks.png", dpi=170); plt.close(fig)
    return df, yr


# ------------------------------------------------------------------ 5. fuite spatiale
def leakage(frame, ens_test, thr):
    tr = frame[frame.split == "train"]; te_ = frame[frame.split == "test"].reset_index(drop=True); va = frame[frame.split == "val"]
    tree = cKDTree(tr[["x", "y"]].to_numpy())
    d_test, _ = tree.query(te_[["x", "y"]].to_numpy()); d_val, _ = tree.query(va[["x", "y"]].to_numpy())
    pos_tree = cKDTree(tr[tr.target == 1][["x", "y"]].to_numpy())
    d_pos, _ = pos_tree.query(te_[te_.target == 1][["x", "y"]].to_numpy())
    y = te_.target.to_numpy()
    dist = {"test_min_m": float(d_test.min()), "test_median_m": float(np.median(d_test)), "val_median_m": float(np.median(d_val)),
            **{f"test_within_{r}m_%": float(100 * (d_test < r).mean()) for r in (16, 32, 50, 100)},
            "test_pos_median_dist_to_train_pos_m": float(np.median(d_pos))}
    rows = []
    for buf in (0, 16, 32, 50, 100, 150):
        m = d_test >= buf
        if y[m].sum() < 5:
            continue
        s = ens_test[m]
        rows.append({"buffer_m": buf, "arbres": int(m.sum()), "positifs": int(y[m].sum()), "pr_auc": average_precision_score(y[m], s),
                     "roc_auc": roc_auc_score(y[m], s), "f1": f1_score(y[m], s >= thr), "prevalence_%": 100 * y[m].mean()})
    df = pd.DataFrame(rows); df.to_csv(OUT / "leakage_buffer.csv", index=False)
    # autocorrélation : un positif Test a-t-il un autre positif proche ?
    tp = cKDTree(te_[te_.target == 1][["x", "y"]].to_numpy())
    dd, _ = tp.query(te_[te_.target == 1][["x", "y"]].to_numpy(), k=2)
    tn = cKDTree(te_[["x", "y"]].to_numpy()); dn, _ = tn.query(te_[te_.target == 1][["x", "y"]].to_numpy(), k=2)
    dist["test_pos_nn_pos_median_m"] = float(np.median(dd[:, 1])); dist["test_pos_nn_any_median_m"] = float(np.median(dn[:, 1]))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(np.clip(d_test, 0, 600), bins=60, color="#8FB59A")
    for r in (16, 50):
        ax.axvline(r, color="k", ls="--", lw=.8); ax.text(r + 4, ax.get_ylim()[1] * .9, f"{r} m", fontsize=7)
    ax.set(xlabel="Distance au plus proche arbre Train (m, tronquée à 600)", ylabel="Arbres Test", title="Proximité Test–Train")
    fig.tight_layout(); fig.savefig(FIG / "leakage_distance.png", dpi=170); plt.close(fig)
    return df, dist


# ------------------------------------------------------------------ 16 & 19. graines multiples
class MLP(nn.Module):
    def __init__(self, hidden=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(12, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(1)


def multiseed(frame, parts, y):
    """Ré-entraîne les configurations retenues du run initial avec 5 graines (même protocole, mêmes époques)."""
    seq = np.stack([frame[[f"intensity_{st}_{yr}" for yr in (2019, 2020, 2021)]].to_numpy() for st in ("mean", "std", "p10", "p90")], 1)
    trm = frame.split.to_numpy() == "train"
    sc = StandardScaler().fit(seq[trm].transpose(0, 2, 1).reshape(-1, 4))
    seq = sc.transform(seq.transpose(0, 2, 1).reshape(-1, 4)).reshape(len(frame), 3, 4).transpose(0, 2, 1)
    sp = {s: seq[frame.split.to_numpy() == s] for s in parts}
    rgb19 = np.load(ti.PROC / "oqds_rgb_2019.npy", mmap_mode="r"); rgb21 = np.load(ti.PROC / "oqds_rgb_2021.npy", mmap_mode="r")
    nrm = np.load(ROOT / "outputs" / "models" / "rgb_normalization_train_only.npz")
    pair = {s: te.PairDataset(parts[s].patch_index, y[s], rgb19, rgb21, nrm["mean"], nrm["std"]) for s in parts}
    feat = [c for c in frame if c.startswith(("intensity_", "delta_mean_", "relative_delta_"))]
    fsc = StandardScaler().fit(parts["train"][feat]); xnum = {s: fsc.transform(parts[s][feat]) for s in parts}
    pw = float((y["train"] == 0).sum() / (y["train"] == 1).sum())
    rows, scores = [], {}
    specs = {
        "siamese_cnn_initial": lambda: (te.SiameseCNN(16, 0.35), pair, 7e-4, 6),
        "tcn": lambda: (te.TCN(12, 2, 0.10), {s: te.SequenceDataset(sp[s], y[s]) for s in parts}, 1e-3, 8),
        "mlp_temporel": lambda: (MLP(32, 0.1), {s: te.SequenceDataset(sp[s], y[s]) for s in parts}, 1e-3, 8),
    }
    for name, spec in specs.items():
        scores[name] = {"val": [], "test": []}
        for k in range(N_SEEDS):
            torch.manual_seed(SEED + 500 + k); np.random.seed(SEED + 500 + k)
            model, ds, lr, ep = spec()
            loaders = {s: DataLoader(ds[s], batch_size=512, shuffle=(s == "train"), num_workers=0) for s in parts}
            loaders["train"].pos_weight = pw
            model, _, fit = te.train_neural(model, loaders["train"], loaders["val"], y["val"], lr, ep, CK / f"{name}_s{k}.pt")
            sv, _ = te.predict_torch(model, loaders["val"]); st, _ = te.predict_torch(model, loaders["test"])
            scores[name]["val"].append(sv); scores[name]["test"].append(st)
            th = thr_f1(y["val"], sv)
            rows.append({"model": name, "seed": k, "n_params": sum(p.numel() for p in model.parameters()), "fit_s": fit,
                         **te.metrics(y["test"], st, th)})
            print(f"  {name} s{k}: testAP={rows[-1]['pr_auc']:.4f}", flush=True)
    scores["isolation_forest"] = {"val": [], "test": []}
    normal = xnum["train"][y["train"] == 0]
    for k in range(N_SEEDS):
        m = IsolationForest(n_estimators=350, max_samples=1024, max_features=0.8, random_state=SEED + 500 + k, n_jobs=-1).fit(normal)
        sv, st = -m.decision_function(xnum["val"]), -m.decision_function(xnum["test"])
        scores["isolation_forest"]["val"].append(sv); scores["isolation_forest"]["test"].append(st)
        rows.append({"model": "isolation_forest", "seed": k, **te.metrics(y["test"], st, thr_f1(y["val"], sv))})
    df = pd.DataFrame(rows); df.to_csv(OUT / "multiseed_per_seed.csv", index=False)
    return df, scores


# ------------------------------------------------------------------ 20. IC95 + comparaisons appariées
def bootstrap(y, blocks, models: dict, thresholds: dict, pairs, n=1000):
    rng = np.random.default_rng(SEED)
    ub = np.unique(blocks); ib = {b: np.flatnonzero(blocks == b) for b in ub}
    stats = {m: {"pr_auc": [], "roc_auc": [], "f1": []} for m in models}
    for _ in range(n):
        idx = np.concatenate([ib[b] for b in rng.choice(ub, len(ub), replace=True)])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for m, s in models.items():
            stats[m]["pr_auc"].append(average_precision_score(yb, s[idx])); stats[m]["roc_auc"].append(roc_auc_score(yb, s[idx]))
            stats[m]["f1"].append(f1_score(yb, s[idx] >= thresholds[m]))
    ci = []
    for m, d in stats.items():
        row = {"model": m}
        for k, v in d.items():
            v = np.asarray(v); row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile(v, [2.5, 97.5])
        ci.append(row)
    paired = []
    for a, b in pairs:
        for k in ("pr_auc", "roc_auc", "f1"):
            dlt = np.asarray(stats[a][k]) - np.asarray(stats[b][k])
            paired.append({"A": a, "B": b, "metric": k, "delta_mean": dlt.mean(), "lo": np.percentile(dlt, 2.5), "hi": np.percentile(dlt, 97.5),
                           "p_A_not_better": float((dlt <= 0).mean())})
    ci, paired = pd.DataFrame(ci), pd.DataFrame(paired)
    ci.to_csv(OUT / "bootstrap_ci_all.csv", index=False); paired.to_csv(OUT / "paired_comparisons.csv", index=False)
    return ci, paired


# ------------------------------------------------------------------ 21–23. points de fonctionnement
def operating_points(y_val, y_test, val_scores: dict, test_scores: dict):
    rows = []
    for m in val_scores:
        sv, st = val_scores[m], test_scores[m]
        strategies = {"F1 optimal": thr_f1(y_val, sv), "Rappel ≥ 60 %": thr_recall(y_val, sv, .6), "Rappel ≥ 80 %": thr_recall(y_val, sv, .8),
                      "Budget 2 % inspectés": thr_budget(sv, .02), "Budget 5 % inspectés": thr_budget(sv, .05)}
        for strat, t in strategies.items():
            pred = st >= t; tp = int((pred & (y_test == 1)).sum()); fp = int((pred & (y_test == 0)).sum())
            rows.append({"model": m, "strategie": strat, "seuil_val": t, "inspectes": int(pred.sum()), "inspectes_%": 100 * pred.mean(),
                         "tp": tp, "fp": fp, "fn": int(y_test.sum()) - tp, "rappel": tp / y_test.sum(),
                         "precision": tp / max(1, pred.sum()), "inspections_par_vrai_positif": pred.sum() / max(1, tp)})
    df = pd.DataFrame(rows); df.to_csv(OUT / "operating_points.csv", index=False)
    return df


# ------------------------------------------------------------------ 25. exemples visuels
def visual_examples(parts, qc_rc, ens_test, thr):
    te_ = parts["test"].assign(score=ens_test)
    yv = te_.target.to_numpy(); pred = ens_test >= thr
    rng = np.random.default_rng(SEED)
    groups = {
        "Vrai positif": te_[(yv == 1) & pred].sort_values("score", ascending=False).index[[0, 5, 20]],
        "Faux positif": te_[(yv == 0) & pred].sort_values("score", ascending=False).index[[0, 5, 20]],
        "Faux négatif": te_[(yv == 1) & ~pred].sort_values("score").index[[0, 5, 20]],
        "Vrai négatif": rng.choice(te_[(yv == 0) & ~pred].index, 3, replace=False),
    }
    rasters = {}
    for yr in (2019, 2020, 2021):
        with rasterio.open(ti.RAW / "raster_data" / f"{yr}_WV2_RGB.tif") as src:
            rasters[yr] = src.read([1, 2, 3])
    H = 32  # demi-fenêtre affichée : 64×64 px = 32 m
    fig, axes = plt.subplots(12, 3, figsize=(6.3, 25))
    r_i = 0
    for g, idxs in groups.items():
        for i in idxs:
            row = te_.loc[i]; r, c = qc_rc.loc[int(row.patch_index), ["row", "col"]].astype(int)
            for j, yr in enumerate((2019, 2020, 2021)):
                ras = rasters[yr]; rr = np.clip(np.arange(r - H, r + H), 0, ras.shape[1] - 1); cc = np.clip(np.arange(c - H, c + H), 0, ras.shape[2] - 1)
                ax = axes[r_i, j]; ax.imshow(ras[:, rr][:, :, cc].transpose(1, 2, 0)); ax.set_xticks([]); ax.set_yticks([])
                ax.add_patch(mpatches.Rectangle((H - 16, H - 16), 32, 32, fill=False, ec="yellow", lw=1))
                if r_i == 0:
                    ax.set_title(str(yr), fontsize=10)
                if j == 0:
                    ax.set_ylabel(f"{g}\nscore {row.score:.2f}", fontsize=7.5)
            r_i += 1
    fig.tight_layout(h_pad=0.3, w_pad=0.2); fig.savefig(FIG / "examples_confusion.png", dpi=130); plt.close(fig)
    # version compacte pour le rapport : 1 exemple par groupe
    fig, axes = plt.subplots(4, 3, figsize=(5.6, 7.6))
    for gi, (g, idxs) in enumerate(groups.items()):
        row = te_.loc[idxs[1]]; r, c = qc_rc.loc[int(row.patch_index), ["row", "col"]].astype(int)
        for j, yr in enumerate((2019, 2020, 2021)):
            ras = rasters[yr]; rr = np.clip(np.arange(r - H, r + H), 0, ras.shape[1] - 1); cc = np.clip(np.arange(c - H, c + H), 0, ras.shape[2] - 1)
            ax = axes[gi, j]; ax.imshow(ras[:, rr][:, :, cc].transpose(1, 2, 0)); ax.set_xticks([]); ax.set_yticks([])
            ax.add_patch(mpatches.Rectangle((H - 16, H - 16), 32, 32, fill=False, ec="yellow", lw=1))
            if gi == 0:
                ax.set_title(str(yr), fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{g}\nscore {row.score:.2f}", fontsize=8)
    fig.tight_layout(h_pad=0.3, w_pad=0.2); fig.savefig(FIG / "examples_confusion_compact.png", dpi=170); plt.close(fig)
    return {g: [int(te_.loc[i].tree_id) for i in idxs] for g, idxs in groups.items()}


# ------------------------------------------------------------------ main
def ensemble_scores(parts, y):
    norm = np.load(ti.MODELS / "rgb_normalization_train_only.npz")
    a19 = np.load(ti.RUN_DATA / "oqds_rgb32_2019.npy", mmap_mode="r"); a21 = np.load(ti.RUN_DATA / "oqds_rgb32_2021.npy", mmap_mode="r")
    out = {}
    for s in ("val", "test"):
        data = ti.Pairs(a19, a21, parts[s].patch_index.to_numpy(int), y[s], norm["mean"], norm["std"], 32)
        ss = []
        for k in range(5):
            m = ti.SiameseV2(16, 0.3); m.load_state_dict(torch.load(ti.MODELS / f"v2_p32_bce_seed{k}.pt", weights_only=True))
            ss.append(ti.predict(m, data))
        out[s] = np.mean(ss, 0)
    return out


def main():
    frame = pd.read_csv(ti.PROC / "oqds_features_splits.csv")
    parts = {s: frame[frame.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    y = {s: parts[s].target.to_numpy(np.uint8) for s in parts}
    qc_rc = pd.read_csv(ti.PROC / "oqds_qc_all.csv", usecols=["row", "col"])
    frozen = json.loads((ti.RESULTS / "frozen_choices.json").read_text(encoding="utf-8"))
    thr_ens = frozen["ensemble_thresholds"]["f1_opt"]
    res = {}

    print("== Exclusions / déséquilibre", flush=True)
    excl_tab, res["exclusions"] = exclusions()
    imb, _ = imbalance(frame)

    print("== Scores ensemble v2", flush=True)
    ens = ensemble_scores(parts, y)
    saved = pd.read_csv(ti.RESULTS / "test_predictions.csv")
    res["ensemble_reproduction_max_abs_diff"] = float(np.abs(saved.score_ensemble.to_numpy() - ens["test"]).max())

    print("== Fuite spatiale", flush=True)
    _, res["leakage_distances"] = leakage(frame, ens["test"], thr_ens)

    print("== Graines multiples", flush=True)
    ms, ms_scores = multiseed(frame, parts, y)
    agg = ms.groupby("model")[["pr_auc", "roc_auc", "f1", "recall", "precision"]].agg(["mean", "std"])
    v2 = pd.read_csv(ti.RESULTS / "test_per_seed.csv").assign(model="siamese_v2")
    agg_v2 = v2.groupby("model")[["pr_auc", "roc_auc", "f1", "recall", "precision"]].agg(["mean", "std"])
    agg = pd.concat([agg_v2, agg]); agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.reset_index().to_csv(OUT / "multiseed_summary.csv", index=False)

    print("== Bootstrap / points de fonctionnement", flush=True)
    orig_val = pd.read_csv(ROOT / "outputs" / "results" / "validation_predictions_all_configs.csv")
    chosen = json.loads((ROOT / "outputs" / "results" / "selected_configurations.json").read_text(encoding="utf-8"))
    orig_test = pd.read_csv(ROOT / "outputs" / "results" / "test_predictions_all_models.csv")
    val_s, test_s, thr = {"siamese_v2": ens["val"]}, {"siamese_v2": ens["test"]}, {"siamese_v2": thr_ens}
    for m in ("siamese_cnn", "tcn", "cusum", "threshold", "isolation_forest"):
        gv = orig_val[(orig_val.model == m) & (orig_val.config == chosen[m]["config"])].set_index("tree_id")
        val_s[m] = gv.loc[parts["val"].tree_id, "score"].to_numpy()
        test_s[m] = orig_test[orig_test.model == m].set_index("tree_id").loc[parts["test"].tree_id, "score"].to_numpy()
        thr[m] = chosen[m]["threshold"]
    blocks = parts["test"].spatial_block.to_numpy()
    pairs = [("siamese_v2", m) for m in ("siamese_cnn", "tcn", "cusum", "threshold", "isolation_forest")] + \
            [("siamese_cnn", m) for m in ("tcn", "cusum", "threshold", "isolation_forest")] + [("tcn", "cusum"), ("cusum", "threshold")]
    bootstrap(y["test"], blocks, test_s, thr, pairs)
    operating_points(y["val"], y["test"], val_s, test_s)

    print("== Exemples visuels", flush=True)
    res["visual_examples_tree_ids"] = visual_examples(parts, qc_rc, ens["test"], thr_ens)
    (OUT / "analyses_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
