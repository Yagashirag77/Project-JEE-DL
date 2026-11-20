"""Préparation OQDS : QC, caractéristiques temporelles, patches RGB et split spatial unique."""
from __future__ import annotations

import json
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from sklearn.model_selection import StratifiedGroupKFold


SEED = 20261106
PATCH = 16
BLOCK_M = 250
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "repository"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "outputs" / "results"
PROCESSED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def find_raster(year: int, token: str) -> Path:
    matches = sorted(p for p in RAW.rglob("*.tif") if str(year) in p.name and token.lower() in p.name.lower())
    if len(matches) != 1:
        raise RuntimeError(f"Attendu exactement un raster {year}/{token}, trouvé: {matches}")
    return matches[0]


def padded_indices(rows: np.ndarray, cols: np.ndarray, h: int, w: int, radius: int = 2):
    offsets = np.arange(-radius, radius + 1)
    rr = np.clip(rows[:, None, None] + offsets[None, :, None], 0, h - 1)
    cc = np.clip(cols[:, None, None] + offsets[None, None, :], 0, w - 1)
    return rr, cc


def extract_patch_array(path: Path, rows: np.ndarray, cols: np.ndarray, valid: np.ndarray, out_path: Path) -> dict:
    """Charge un seul grand raster à la fois et écrit les petits patches dans un memmap."""
    n = len(rows)
    patches = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(n, 3, PATCH, PATCH))
    patches[:] = 0
    with rasterio.open(path) as src:
        rgb = src.read([1, 2, 3])
        half = PATCH // 2
        black_ratios = np.ones(n, dtype=np.float32)
        for i in np.flatnonzero(valid):
            r, c = int(rows[i]), int(cols[i])
            r0, r1, c0, c1 = r - half, r + half, c - half, c + half
            if r0 < 0 or c0 < 0 or r1 > src.height or c1 > src.width:
                valid[i] = False
                continue
            patch = rgb[:, r0:r1, c0:c1]
            patches[i] = patch
            black_ratios[i] = float(np.mean(np.all(patch == 0, axis=0)))
        del rgb
    patches.flush()
    return {"black_ratio": black_ratios}


def main() -> None:
    shp = sorted(RAW.rglob("*.shp"))
    if len(shp) != 1:
        raise RuntimeError(f"Un shapefile OQDS attendu, trouvé: {shp}")
    gdf = gpd.read_file(shp[0])
    original_count = len(gdf)
    if str(gdf.crs) != "EPSG:32633":
        raise RuntimeError(f"CRS inattendu: {gdf.crs}")
    if "is.infect" not in gdf or "fid" not in gdf:
        raise RuntimeError(f"Colonnes attendues absentes: {list(gdf.columns)}")
    gdf = gdf.assign(_wkb=gdf.geometry.to_wkb())
    conflicting = gdf.groupby("_wkb")["is.infect"].nunique().gt(1).sum()
    if conflicting:
        raise RuntimeError(f"{conflicting} coordonnées dupliquées avec labels contradictoires")
    duplicate_rows = int(gdf.duplicated("_wkb", keep="first").sum())
    gdf = gdf.drop_duplicates("_wkb", keep="first").reset_index(drop=True)
    gdf["tree_id"] = gdf["fid"].astype(str)
    gdf["target"] = gdf["is.infect"].astype(bool).astype(np.uint8)
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y

    paths = {(year, kind): find_raster(year, kind) for year in (2019, 2020, 2021) for kind in ("NDVI", "RGB")}
    reference = rasterio.open(paths[(2019, "RGB")])
    ref_meta = (reference.crs, reference.transform, reference.width, reference.height, reference.bounds)
    reference.close()
    alignment = {}
    for key, path in paths.items():
        with rasterio.open(path) as src:
            ok = src.crs == ref_meta[0] and src.transform == ref_meta[1] and src.width == ref_meta[2] and src.height == ref_meta[3]
            alignment[f"{key[0]}_{key[1]}"] = bool(ok)
    if not all(alignment.values()):
        raise RuntimeError(f"Rasters non alignés: {alignment}")

    with rasterio.open(paths[(2019, "RGB")]) as src:
        rows, cols = rasterio.transform.rowcol(src.transform, gdf.x.to_numpy(), gdf.y.to_numpy())
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = (rows >= PATCH // 2) & (rows < src.height - PATCH // 2) & (cols >= PATCH // 2) & (cols < src.width - PATCH // 2)
    frame = pd.DataFrame({
        "tree_id": gdf.tree_id, "x": gdf.x, "y": gdf.y, "target": gdf.target,
        "removal_year": gdf["year"], "row": rows, "col": cols, "inside_all_rasters": inside,
    })

    # Les fichiers *NDVI sont des RGBA uint8 dont RGB est identique : intensité rendue, pas NDVI physique.
    for year in (2019, 2020, 2021):
        with rasterio.open(paths[(year, "NDVI")]) as src:
            gray = src.read(1)
            alpha = src.read(4)
            rr, cc = padded_indices(rows, cols, src.height, src.width, radius=2)
            values = gray[rr, cc].astype(np.float32)
            avals = alpha[rr, cc]
            frame[f"intensity_mean_{year}"] = values.mean(axis=(1, 2))
            frame[f"intensity_std_{year}"] = values.std(axis=(1, 2))
            frame[f"intensity_p10_{year}"] = np.percentile(values, 10, axis=(1, 2))
            frame[f"intensity_p90_{year}"] = np.percentile(values, 90, axis=(1, 2))
            frame[f"alpha_valid_fraction_{year}"] = (avals > 0).mean(axis=(1, 2))
            del gray, alpha, values, avals

    valid = inside.copy()
    rgb_qc = {}
    for year in (2019, 2021):
        info = extract_patch_array(paths[(year, "RGB")], rows, cols, valid, PROCESSED / f"oqds_rgb_{year}.npy")
        frame[f"rgb_black_fraction_{year}"] = info["black_ratio"]
        rgb_qc[str(year)] = {"mean_black_fraction": float(np.mean(info["black_ratio"][valid]))}

    frame["valid"] = valid
    frame["invalid_reason"] = ""
    frame.loc[~inside, "invalid_reason"] = "outside_or_patch_border"
    alpha_cols = [f"alpha_valid_fraction_{y}" for y in (2019, 2020, 2021)]
    alpha_bad = frame[alpha_cols].min(axis=1) < 0.80
    black_bad = frame[["rgb_black_fraction_2019", "rgb_black_fraction_2021"]].max(axis=1) > 0.80
    frame.loc[frame.valid & alpha_bad, "invalid_reason"] = "low_alpha_coverage"
    frame.loc[frame.valid & ~alpha_bad & black_bad, "invalid_reason"] = "mostly_black_rgb"
    frame.loc[alpha_bad | black_bad, "valid"] = False

    for a, b in ((2019, 2020), (2020, 2021), (2019, 2021)):
        frame[f"delta_mean_{a}_{b}"] = frame[f"intensity_mean_{b}"] - frame[f"intensity_mean_{a}"]
        frame[f"relative_delta_{a}_{b}"] = frame[f"delta_mean_{a}_{b}"] / (frame[f"intensity_mean_{a}"].abs() + 1.0)

    usable = frame[frame.valid].copy().reset_index(drop=True)
    usable["spatial_block"] = (np.floor(usable.x / BLOCK_M).astype(int).astype(str) + "_" + np.floor(usable.y / BLOCK_M).astype(int).astype(str))
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    usable["fold"] = -1
    for fold, (_, idx) in enumerate(splitter.split(usable, usable.target, usable.spatial_block)):
        usable.loc[idx, "fold"] = fold
    # Fold 0 est figé comme Test; parmi les six autres, le fold 5 donne une
    # prévalence Validation proche de Train/Test malgré l'agrégation spatiale des cas.
    usable["split"] = np.where(usable.fold == 0, "test", np.where(usable.fold == 5, "val", "train"))
    if usable.groupby("spatial_block").split.nunique().max() != 1:
        raise AssertionError("Fuite de bloc spatial")
    if usable.groupby("tree_id").split.nunique().max() != 1:
        raise AssertionError("Fuite d'identifiant d'arbre")
    if not all(usable.groupby("split").target.nunique() == 2):
        raise AssertionError("Une partition ne contient pas les deux classes")

    # Les patches ont été écrits avant filtrage ; cet index est la jointure immuable vers les .npy.
    usable["patch_index"] = frame.index[frame.valid].to_numpy()
    frame.to_csv(PROCESSED / "oqds_qc_all.csv", index=False)
    usable.to_csv(PROCESSED / "oqds_features_splits.csv", index=False)
    usable[["tree_id", "spatial_block", "fold", "split", "target", "patch_index"]].to_csv(PROCESSED / "oqds_splits.csv", index=False)

    summary = {
        "seed": SEED, "patch_size": PATCH, "spatial_block_m": BLOCK_M,
        "original_points": original_count, "duplicate_coordinate_rows_removed": duplicate_rows,
        "after_deduplication": len(gdf), "usable_points": len(usable),
        "invalid_counts": frame.loc[~frame.valid, "invalid_reason"].value_counts().to_dict(),
        "alignment": alignment, "rgb_qc": rgb_qc,
        "split_counts": usable.groupby(["split", "target"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "split_blocks": usable.groupby("split").spatial_block.nunique().to_dict(),
        "positive_definition": "is.infect=True: arbre 2019 absent en 2020/2021, interprété comme retiré après infection probable OQDS",
        "radiometry_warning": "Les fichiers nommés NDVI sont des rendus RGBA uint8 (bandes RGB identiques), sans métadonnée de calibration; les valeurs sont traitées comme intensité, pas comme NDVI physique.",
    }
    (RESULTS / "preparation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
