# Phases B–C — protocole et préparation

- Benchmark commun: OQDS seulement, cible `is.infect` (retrait/disparition probable après OQDS).
- Représentations: intensités temporelles 2019–2021 pour seuil/CUSUM/Isolation Forest/TCN; paires RGB 2019–2021 pour Siamese.
- Les fichiers nommés NDVI sont des RGBA 8 bits sans calibration: ils sont traités comme intensité, jamais comme NDVI physique.
- Nettoyage: 104 doublons géométriques retirés, 5026 observations invalides exclues, 71956 arbres retenus.
- Split unique par blocs de 250 m: {'test': {'0': 9968, '1': 237}, 'train': {'0': 49884, '1': 1541}, 'val': {'0': 10035, '1': 291}}. Aucun arbre ni bloc partagé.
- Les scalers RGB, numériques et TCN sont ajustés exclusivement sur Train.
