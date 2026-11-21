"""OLV-6 — coût de calcul comparé dans des conditions identiques (remarque jury 31).

Même machine, CPU seul, aucun autre calcul en cours. Pour chaque méthode :
paramètres, taille du fichier, latence d'une requête unitaire (1 arbre, 1 thread, médiane de 300 appels),
débit par lot (10 205 arbres test, 8 threads) et mémoire résidente supplémentaire pendant l'inférence.
Les patches sont supposés déjà extraits (le découpage raster n'est pas inclus).
Sortie : outputs/runs/improved_v1/experiments/complementaires/benchmark_compute.csv
"""
from __future__ import annotations

import gc
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import torch

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_improved as ti  # noqa: E402
import train_evaluate as te  # noqa: E402
from experiments_spatial import SiameseAttn  # noqa: E402

OUT = ti.RUN / "experiments" / "complementaires"
PROC = psutil.Process()


def rss():
    gc.collect(); return PROC.memory_info().rss / 2 ** 20


def bench(name, fn_one, fn_all, n_params, file_mb):
    torch.set_num_threads(1)
    for _ in range(20):
        fn_one()
    t = []
    for _ in range(300):
        t0 = time.perf_counter(); fn_one(); t.append(time.perf_counter() - t0)
    torch.set_num_threads(8)
    fn_all()
    m0 = rss(); peak = m0
    t0 = time.perf_counter()
    fn_all()
    dur = time.perf_counter() - t0
    peak = max(peak, rss())
    return {"methode": name, "parametres": n_params, "fichier_Mo": file_mb, "latence_1_arbre_ms": 1000 * float(np.median(t)),
            "lot_10205_s": dur, "debit_arbres_par_s": 10205 / dur, "memoire_sup_Mo": peak - m0}


def main():
    frame = pd.read_csv(ti.PROC / "oqds_features_splits.csv")
    tst = frame[frame.split == "test"].reset_index(drop=True)
    idx = tst.patch_index.to_numpy(int)
    M = ti.ROOT / "outputs" / "models"
    rows = []
    mb = lambda p: Path(p).stat().st_size / 2 ** 20

    # Siamese initial (16×16)
    ck = torch.load(M / "siamese_cnn_best.pt", weights_only=False)
    s1 = te.SiameseCNN(ck["config"]["width"], ck["config"]["dropout"]); s1.load_state_dict(ck["state_dict"]); s1.eval()
    a16 = torch.from_numpy(np.load(ti.PROC / "oqds_rgb_2019.npy")[idx]).float() / 255
    b16 = torch.from_numpy(np.load(ti.PROC / "oqds_rgb_2021.npy")[idx]).float() / 255
    def run(m, a, b, bs=1024):
        with torch.inference_mode():
            return [m(a[i:i + bs], b[i:i + bs]) for i in range(0, len(a), bs)]
    rows.append(bench("Siamese initial (16×16)", lambda: run(s1, a16[:1], b16[:1]), lambda: run(s1, a16, b16),
                      sum(p.numel() for p in s1.parameters()), mb(M / "siamese_cnn_best.pt")))

    # Siamese v2 (32×32) : un modèle, puis l'ensemble de 5
    ck2 = torch.load(ti.MODELS / "siamese_ensemble_best.pt", weights_only=False)
    v2s = []
    for sd in ck2["state_dicts"]:
        m = ti.SiameseV2(16, 0.3); m.load_state_dict(sd); m.eval(); v2s.append(m)
    a32 = torch.from_numpy(np.load(ti.RUN_DATA / "oqds_rgb32_2019.npy")[idx]).float() / 255
    b32 = torch.from_numpy(np.load(ti.RUN_DATA / "oqds_rgb32_2021.npy")[idx]).float() / 255
    npar = sum(p.numel() for p in v2s[0].parameters())
    rows.append(bench("Siamese v2 (1 modèle)", lambda: run(v2s[0], a32[:1], b32[:1]), lambda: run(v2s[0], a32, b32),
                      npar, mb(ti.MODELS / "siamese_ensemble_best.pt") / 5))
    rows.append(bench("Siamese v2 (ensemble 5)", lambda: [run(m, a32[:1], b32[:1]) for m in v2s],
                      lambda: [run(m, a32, b32) for m in v2s], 5 * npar, mb(ti.MODELS / "siamese_ensemble_best.pt")))
    att = SiameseAttn(16, 0.3).eval()
    rows.append(bench("Siamese v2 + attention", lambda: run(att, a32[:1], b32[:1]), lambda: run(att, a32, b32),
                      sum(p.numel() for p in att.parameters()), float("nan")))

    # TCN
    ckt = torch.load(M / "tcn_best.pt", weights_only=False)
    tcn = te.TCN(ckt["config"]["channels"], ckt["config"]["kernel"], ckt["config"]["dropout"]); tcn.load_state_dict(ckt["state_dict"]); tcn.eval()
    scaler = joblib.load(M / "tcn_scaler_train_only.joblib")
    seq = np.stack([tst[[f"intensity_{st}_{yr}" for yr in (2019, 2020, 2021)]].to_numpy() for st in ("mean", "std", "p10", "p90")], 1)
    seq = torch.from_numpy(scaler.transform(seq.transpose(0, 2, 1).reshape(-1, 4)).reshape(len(tst), 3, 4).transpose(0, 2, 1).astype(np.float32))
    def run_t(x, bs=4096):
        with torch.inference_mode():
            return [tcn(x[i:i + bs]) for i in range(0, len(x), bs)]
    rows.append(bench("TCN", lambda: run_t(seq[:1]), lambda: run_t(seq), sum(p.numel() for p in tcn.parameters()), mb(M / "tcn_best.pt")))

    # Isolation Forest
    iso = joblib.load(M / "isolation_forest.joblib"); sc = joblib.load(M / "numeric_scaler.joblib")
    X = sc["scaler"].transform(tst[sc["features"]])
    n_nodes = sum(e.tree_.node_count for e in iso.estimators_)
    rows.append(bench("Isolation Forest", lambda: iso.decision_function(X[:1]), lambda: iso.decision_function(X),
                      n_nodes, mb(M / "isolation_forest.joblib")))

    # CUSUM et seuil (calcul vectoriel)
    cfg = joblib.load(M / "cusum.joblib")
    one = tst.iloc[:1]
    rows.append(bench("CUSUM", lambda: te.cusum_score(one, False, 10.0), lambda: te.cusum_score(tst, False, 10.0), 2, mb(M / "cusum.joblib")))
    rows.append(bench("Seuil", lambda: te.baseline_score(one, "max_annual_decline"), lambda: te.baseline_score(tst, "max_annual_decline"),
                      1, mb(M / "threshold.joblib")))

    df = pd.DataFrame(rows)
    df["machine"] = f"{platform.processor()} | {psutil.cpu_count(logical=True)} threads logiques | torch {torch.__version__}"
    df.to_csv(OUT / "benchmark_compute.csv", index=False)
    print(df.drop(columns="machine").to_string())
    print(df.machine.iloc[0])


if __name__ == "__main__":
    main()
