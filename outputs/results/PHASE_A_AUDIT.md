# Audit scientifique des sources OLV-6

Audit généré depuis les fichiers présents localement, sans modification des données brutes.

## OQDS Insight

- 6 GeoTIFF et 1 shapefile détectés.
- Années déduites des noms : ['2019', '2020', '2021'].
- Métadonnées détaillées : `outputs/results/oqds_metadata.json`.

## NEON Tree Crowns

- 1 GeoPackage, 41,738 entités.
- Schéma, identifiants et dates candidates : `outputs/results/neon_metadata.json`.

## Olive Hyperspectral Stress

- 1 classeur et 8 cubes TIFF.
- Dimensions des cubes : [(50, 50, 138), (50, 50, 138), (50, 50, 138), (50, 50, 138), (50, 50, 138), (50, 50, 138), (50, 50, 138), (50, 50, 138)].
- Classes, bandes, valeurs manquantes et tags : `outputs/results/hyperspectral_metadata.json`.

## OliveTreeCrownsDb

- 5,844 images, 5,084 XML Pascal VOC, 5,084 fichiers texte.
- 33,479 boîtes VOC ; classes : {'0': 33479}.
- Résolutions/modes : {(5472, 3648, 'RGB'): 46, (1824, 1216, 'RGB'): 414, (912, 608, 'RGB'): 1656, (608, 405, 'RGB'): 3726, (829, 701, 'RGBA'): 1, (855, 701, 'RGBA'): 1}.
- Indices temporels explicites dans les noms : {}.
- Audit détaillé : `outputs/results/crownsdb_metadata.json`.

## Conclusion d'aptitude initiale

- **OQDS** est la seule source candidate à un benchmark temporel commun si les mêmes points sont couverts en 2019–2021 et si l'étiquette vectorielle représente bien l'état cible.
- **NEON** peut documenter les observations répétées et la transférabilité structurelle, mais ne doit pas être fusionné avec les classes d'oliviers sans cible compatible.
- **Hyperspectral** permet une expérience complémentaire de classification de stress spectral, distincte du benchmark temporel OQDS.
- **CrownsDb** permet une expérience de contrôle qualité/détection de couronnes, mais l'absence de suivi temporel explicite interdit de le présenter comme preuve de détection du dépérissement.

Inventaire exhaustif : 16,033 fichiers dans `outputs/results/dataset_inventory.csv`.
