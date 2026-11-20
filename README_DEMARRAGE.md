# OLV-6 — Démarrage depuis zéro (Jupyter)

## Sources EXACTEMENT issues du sujet OLV-6 (PDF fourni)

1. **OQDS Insight** — https://doi.org/10.6084/m9.figshare.28191245.v4 — Cliquer Download / Download all, puis décompresser dans `data/raw/` en conservant les sous-dossiers GeoTIFF + shapefile (avec SHX/DBF/PRJ). Publication : https://doi.org/10.1016/j.dib.2025.111615.
2. **NEON Tree Crowns DTA** — https://huggingface.co/datasets/weecology/neon-tree-crowns-dta/tree/main — Onglet Files, télécharger `neon_crowns_dta.gpkg`, placer dans `data/neon/`.
3. **Olive hyperspectral stress** — https://zenodo.org/records/10336545 — Section Files, récupérer `merged_hs_olive_gr_tr.xlsx` et les fichiers `session_*_cube.tiff`, placer dans `data/hyperspectral/`.
4. **OliveTreeCrownsDb** — https://data.mendeley.com/datasets/xym8rd2srf/2 — bouton **Download All**, décompresser dans `data/crownsdb/`.

## Installation

1. Dézipper ce projet, ouvrir **Jupyter Notebook ou JupyterLab** au niveau du dossier contenant `OLV6_Comparaison_Complete.ipynb`.
2. Installer les dépendances via la première cellule `%pip install ...` ou `pip install -r requirements.txt` dans le terminal **du même environnement Python**.
3. Télécharger les 4 sources avant l'audit. **Les datasets ne sont pas inclus dans ce ZIP** (licences/tailles).
4. Exécuter les cellules **dans l'ordre**. L'extraction OQDS est lourde (images satellites volumineuses). Prévoir de l'espace disque et un GPU si disponible pour les réseaux. Les lignes `assert` sont des contrôles utiles : ne pas les contourner.
5. Les données préparées, modèles, scores Test par arbre, graphiques et historique seront enregistrés dans `outputs/`. Sauvegarder ce dossier avant la remise.

## Protocole scientifique

- Explorer les 4 sources et documenter la compatibilité avec la tâche visée.
- Benchmark principal : **5 approches sur OQDS**, mêmes arbres et mêmes partitions géographiques, même cible, réglages et choix sur validation, test final unique. Les méthodes n'utilisent pas des représentations d'images identiques ; comparaison de **pipelines**.
- NEON : autres essences, ne constitue pas des étiquettes de maladie d'oliviers.
- Hyperspectral : annotations de stress spectral, pas automatiquement comparables aux étiquettes OQDS.
- CrownsDb : couronnes annotées, pas automatiquement de suivi temporel.
- Les datasets complémentaires sont explorés et peuvent servir à des expériences spécifiques ; ce fichier **ne réalise pas 20 entraînements**. Si une consigne orale impose effectivement un benchmark des cinq architectures sur les quatre sources, il faut obtenir une cible commune et documenter les combinaisons impossibles.
- Les valeurs RGB/gris ne sont **pas du NDVI physique** sans preuve de calibration.
- Le code doit être exécuté sur les jeux réels : les performances finales ne sont pas connues à l'avance et une adaptation aux schémas téléchargés peut être nécessaire.

## Livrables visés

Notebook `.ipynb`, `outputs/tables`, `outputs/figures`, `outputs/scores`, `outputs/models`, historique d'entraînement. Rapport court de 3–4 pages après mesures.
