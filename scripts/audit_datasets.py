"""Audit reproductible des quatre sources OLV-6 (lecture seule des données brutes)."""
from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "outputs" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def add_file(rows: list[dict], dataset: str, path: Path, role: str, **meta) -> None:
    row = {
        "dataset": dataset,
        "path": rel(path),
        "role": role,
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
    }
    row.update(meta)
    rows.append(row)


def safe_unique(series: pd.Series, limit: int = 30) -> list:
    return series.dropna().astype(str).unique().tolist()[:limit]


def main() -> None:
    rows: list[dict] = []
    report: list[str] = [
        "# Audit scientifique des sources OLV-6", "",
        "Audit généré depuis les fichiers présents localement, sans modification des données brutes.", "",
    ]

    oqds_root = DATA / "raw"
    oqds_tifs = sorted([*oqds_root.rglob("*.tif"), *oqds_root.rglob("*.tiff")])
    oqds_shps = sorted(oqds_root.rglob("*.shp"))
    oqds_meta = []
    for path in oqds_tifs:
        with rasterio.open(path) as src:
            item = {
                "name": path.name, "width": src.width, "height": src.height, "bands": src.count,
                "dtype": ",".join(src.dtypes), "crs": str(src.crs), "nodata": src.nodata,
                "resolution": [float(abs(src.transform.a)), float(abs(src.transform.e))],
                "bounds": [float(x) for x in src.bounds],
            }
            oqds_meta.append(item)
            add_file(rows, "OQDS", path, "raster", **item)
    oqds_vectors = []
    for path in oqds_shps:
        gdf = gpd.read_file(path)
        vector = {
            "path": rel(path), "rows": int(len(gdf)), "crs": str(gdf.crs),
            "geometry_types": gdf.geometry.geom_type.value_counts(dropna=False).to_dict(),
            "columns": list(gdf.columns), "null_counts": {c: int(gdf[c].isna().sum()) for c in gdf.columns},
            "duplicates_full": int(gdf.drop(columns="geometry").duplicated().sum()),
            "duplicate_geometries": int(gdf.geometry.to_wkb().duplicated().sum()),
            "bounds": [float(x) for x in gdf.total_bounds],
        }
        categorical = {}
        for c in gdf.columns:
            if c != "geometry" and (gdf[c].dtype == "object" or gdf[c].nunique(dropna=True) <= 20):
                categorical[c] = gdf[c].value_counts(dropna=False).head(30).to_dict()
        vector["categorical_distributions"] = categorical
        oqds_vectors.append(vector)
        add_file(rows, "OQDS", path, "vector", row_count=len(gdf), crs=str(gdf.crs))
    (RESULTS / "oqds_metadata.json").write_text(
        json.dumps({"rasters": oqds_meta, "vectors": oqds_vectors}, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    report += ["## OQDS Insight", "", f"- {len(oqds_tifs)} GeoTIFF et {len(oqds_shps)} shapefile détectés.",
               f"- Années déduites des noms : {sorted({y for p in oqds_tifs for y in ('2019','2020','2021') if y in p.name})}.",
               "- Métadonnées détaillées : `outputs/results/oqds_metadata.json`.", ""]

    neon_paths = sorted((DATA / "neon").rglob("*.gpkg"))
    neon_summaries = []
    for path in neon_paths:
        neon = gpd.read_file(path)
        date_candidates = [c for c in neon.columns if any(k in c.lower() for k in ("date", "year", "time"))]
        id_candidates = [c for c in neon.columns if any(k in c.lower() for k in ("id", "tag", "individual"))]
        summary = {
            "path": rel(path), "rows": len(neon), "columns": list(neon.columns),
            "dtypes": {c: str(t) for c, t in neon.dtypes.items()}, "crs": str(neon.crs),
            "date_candidates": date_candidates, "id_candidates": id_candidates,
            "date_values": {c: safe_unique(neon[c]) for c in date_candidates},
            "id_unique": {c: int(neon[c].nunique(dropna=True)) for c in id_candidates},
            "null_counts": {c: int(neon[c].isna().sum()) for c in neon.columns},
        }
        neon_summaries.append(summary)
        add_file(rows, "NEON", path, "geopackage", row_count=len(neon), crs=str(neon.crs))
    (RESULTS / "neon_metadata.json").write_text(json.dumps(neon_summaries, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    report += ["## NEON Tree Crowns", "", f"- {len(neon_paths)} GeoPackage, {sum(x['rows'] for x in neon_summaries):,} entités.",
               "- Schéma, identifiants et dates candidates : `outputs/results/neon_metadata.json`.", ""]

    hs_root = DATA / "hyperspectral"
    hs_xlsx = sorted(hs_root.rglob("*.xlsx"))
    hs_tiffs = sorted([*hs_root.rglob("*.tif"), *hs_root.rglob("*.tiff")])
    hs_excel = []
    for path in hs_xlsx:
        book = pd.ExcelFile(path)
        for sheet in book.sheet_names:
            frame = pd.read_excel(path, sheet_name=sheet)
            item = {
                "path": rel(path), "sheet": sheet, "rows": len(frame), "columns": list(frame.columns),
                "dtypes": {str(c): str(t) for c, t in frame.dtypes.items()},
                "null_counts": {str(c): int(frame[c].isna().sum()) for c in frame.columns},
            }
            cat = {}
            for c in frame.columns:
                if frame[c].dtype == "object" or frame[c].nunique(dropna=True) <= 20:
                    cat[str(c)] = frame[c].value_counts(dropna=False).head(30).to_dict()
            item["categorical_distributions"] = cat
            hs_excel.append(item)
        add_file(rows, "Hyperspectral", path, "spreadsheet", sheets=len(book.sheet_names))
    hs_cubes = []
    for path in hs_tiffs:
        with rasterio.open(path) as src:
            item = {"path": rel(path), "width": src.width, "height": src.height, "bands": src.count,
                    "dtypes": list(src.dtypes), "crs": str(src.crs), "nodata": src.nodata,
                    "descriptions": list(src.descriptions), "tags": src.tags()}
            hs_cubes.append(item)
            add_file(rows, "Hyperspectral", path, "spectral_cube", width=src.width, height=src.height, bands=src.count, dtype=src.dtypes[0])
    (RESULTS / "hyperspectral_metadata.json").write_text(
        json.dumps({"excel": hs_excel, "cubes": hs_cubes}, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    report += ["## Olive Hyperspectral Stress", "", f"- {len(hs_xlsx)} classeur et {len(hs_tiffs)} cubes TIFF.",
               f"- Dimensions des cubes : {[(x['height'], x['width'], x['bands']) for x in hs_cubes]}.",
               "- Classes, bandes, valeurs manquantes et tags : `outputs/results/hyperspectral_metadata.json`.", ""]

    crowns_root = DATA / "crownsdb"
    crowns_files = [p for p in crowns_root.rglob("*") if p.is_file()]
    ext_counts = Counter(p.suffix.lower() for p in crowns_files)
    images = sorted([p for p in crowns_files if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    xmls = sorted(p for p in crowns_files if p.suffix.lower() == ".xml")
    txts = sorted(p for p in crowns_files if p.suffix.lower() == ".txt")
    image_shapes, corrupt_images = Counter(), []
    for path in images:
        try:
            with Image.open(path) as im:
                image_shapes[(im.width, im.height, im.mode)] += 1
        except Exception as exc:
            corrupt_images.append({"path": rel(path), "error": repr(exc)})
    voc_labels, voc_boxes, xml_errors = Counter(), 0, []
    for path in xmls:
        try:
            root = ET.parse(path).getroot()
            objects = root.findall("object")
            voc_boxes += len(objects)
            voc_labels.update((obj.findtext("name") or "<missing>").strip() for obj in objects)
        except Exception as exc:
            xml_errors.append({"path": rel(path), "error": repr(exc)})
    image_set, xml_set, txt_set = set(images), set(xmls), set(txts)
    for path in crowns_files:
        role = "image" if path in image_set else "voc_annotation" if path in xml_set else "yolo_annotation" if path in txt_set else "auxiliary"
        add_file(rows, "CrownsDb", path, role)
    crowns_summary = {
        "files": len(crowns_files), "extension_counts": ext_counts,
        "image_shapes": {str(k): v for k, v in image_shapes.items()}, "corrupt_images": corrupt_images,
        "voc_files": len(xmls), "voc_boxes": voc_boxes, "voc_labels": voc_labels,
        "txt_annotation_files": len(txts), "xml_errors": xml_errors,
        "temporal_tokens_in_names": Counter(y for p in crowns_files for y in range(2000, 2030) if str(y) in p.name),
    }
    (RESULTS / "crownsdb_metadata.json").write_text(json.dumps(crowns_summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    report += ["## OliveTreeCrownsDb", "", f"- {len(images):,} images, {len(xmls):,} XML Pascal VOC, {len(txts):,} fichiers texte.",
               f"- {voc_boxes:,} boîtes VOC ; classes : {dict(voc_labels)}.", f"- Résolutions/modes : {dict(image_shapes)}.",
               f"- Indices temporels explicites dans les noms : {dict(crowns_summary['temporal_tokens_in_names'])}.",
               "- Audit détaillé : `outputs/results/crownsdb_metadata.json`.", ""]

    inventory = pd.DataFrame(rows).sort_values(["dataset", "path"])
    inventory.to_csv(RESULTS / "dataset_inventory.csv", index=False, encoding="utf-8")
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "cuda_available": torch.cuda.is_available(), "numpy": np.__version__, "pandas": pd.__version__,
           "geopandas": gpd.__version__, "rasterio": rasterio.__version__}
    (RESULTS / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    report += ["## Conclusion d'aptitude initiale", "",
               "- **OQDS** est la seule source candidate à un benchmark temporel commun si les mêmes points sont couverts en 2019–2021 et si l'étiquette vectorielle représente bien l'état cible.",
               "- **NEON** peut documenter les observations répétées et la transférabilité structurelle, mais ne doit pas être fusionné avec les classes d'oliviers sans cible compatible.",
               "- **Hyperspectral** permet une expérience complémentaire de classification de stress spectral, distincte du benchmark temporel OQDS.",
               "- **CrownsDb** permet une expérience de contrôle qualité/détection de couronnes, mais l'absence de suivi temporel explicite interdit de le présenter comme preuve de détection du dépérissement.", "",
               f"Inventaire exhaustif : {len(inventory):,} fichiers dans `outputs/results/dataset_inventory.csv`."]
    (RESULTS / "PHASE_A_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Audit terminé:", len(inventory), "fichiers inventoriés")
    print("Rapport:", RESULTS / "PHASE_A_AUDIT.md")


if __name__ == "__main__":
    main()
