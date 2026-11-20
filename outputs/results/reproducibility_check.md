# Vérification non destructive de reproductibilité OLV-6

**État : CONDITIONNELLEMENT REPRODUCTIBLE.**

Cette vérification n'a exécuté aucun `main()` scientifique et n'a entraîné aucun modèle. Elle a analysé la syntaxe, importé les modules sans lancer leurs traitements, contrôlé les dépendances, chemins, graines, partitions, configurations et empreintes des artefacts initiaux.

## Contrôles

| Contrôle | Statut | Détail |
|---|---|---|
| racine autorisée | PASS | C:\Users\Microsoft\Desktop\Project-JEE_DL |
| syntaxe audit_datasets.py | PASS | AST Python valide |
| syntaxe prepare_oqds.py | PASS | AST Python valide |
| syntaxe train_evaluate.py | PASS | AST Python valide |
| syntaxe complementary_report.py | PASS | AST Python valide |
| dépendances Python | PASS | toutes présentes |
| OQDS shapefile | PASS | chemin attendu |
| OQDS six rasters | PASS | chemin attendu |
| NEON GeoPackage | PASS | chemin attendu |
| Hyperspectral Excel | PASS | chemin attendu |
| Hyperspectral 8 cubes | PASS | chemin attendu |
| CrownsDb présent | PASS | chemin attendu |
| imports/démarrage non destructif | PASS | IMPORT_OK |
| graine préparation | PASS | 20261106 |
| graine entraînement | PASS | 20261106 |
| graine complémentaires | PASS | 20261106 |
| sélection Validation | PASS | PR-AUC et seuil sur Validation |
| Test après gel | PASS | bloc Test situé après sauvegarde des choix |
| partition arbre | PASS | {'test': 10205, 'train': 51425, 'val': 10326} |
| partition bloc spatial | PASS | {'test': 52, 'train': 258, 'val': 53} |
| folds fixés | PASS | [0, 1, 2, 3, 4, 5, 6] |
| cinq configurations sauvegardées | PASS | {'threshold': 'max_annual_decline', 'cusum': 'relative=False_k=10.0', 'isolation_forest': 'iso_2', 'siamese_cnn': 'siamese_2', 'tcn': 'tcn_0'} |
| cinq modèles/checkpoints | PASS | ['threshold.joblib', 'cusum.joblib', 'isolation_forest.joblib', 'siamese_cnn_best.pt', 'tcn_best.pt'] |
| ordre reproduction complète | PASS | audit → préparation → entraînement/évaluation → complémentaires/rapport |
| garde-fou entraînement | PASS | confirmation explicite requise |
| intégrité artefacts initiaux | PASS | {'outputs/models/threshold.joblib': True, 'outputs/models/cusum.joblib': True, 'outputs/models/isolation_forest.joblib': True, 'outputs/models/siamese_cnn_best.pt': True, 'outputs/models/tcn_best.pt': True, 'outputs/results/model_comparison_test.csv': True, 'outputs/results/test_predictions_all_models.csv': True, 'outputs/results/validation_experiments.csv': True, 'outputs/results/selected_configurations.json': True} |

## Ordre du mode complet

1. audit des quatre sources ;
2. préparation OQDS et partition spatiale ;
3. optimisation Validation, checkpoints et évaluation Test des cinq modèles ;
4. expériences complémentaires et rapport.

## Ce qui n'a pas été testé

- La reproductibilité bit-à-bit d'un nouvel entraînement PyTorch CPU/GPU.
- Le temps et la mémoire d'une nouvelle exécution intégrale.
- La stabilité des scores sur une seconde graine ou un autre matériel.
- La réussite de chaque écriture au terme d'un run complet interrompu/repris.

## Point de sécurité important

Les scripts actuels utilisent des chemins de sortie fixes. Une véritable `reproduction_complete` dans ce même dossier remplacerait notamment :
- `data/processed/oqds_features_splits.csv`
- `outputs/results/model_comparison_test.csv`
- `outputs/results/test_predictions_all_models.csv`
- `outputs/models/siamese_cnn_best.pt`
- `outputs/models/tcn_best.pt`

Elle ne doit donc pas être lancée tant que les résultats initiaux doivent rester intacts. Pour une preuve complète, il faut exécuter dans une copie isolée du projet ou ajouter un identifiant de run (`outputs/runs/<run_id>/` et `data/processed/runs/<run_id>/`), puis comparer les métriques et empreintes.

## Conclusion

Les contrôles structurels passent et le pipeline est cohérent avec le protocole annoncé. La reproductibilité scientifique complète reste à démontrer par une réexécution isolée autorisée; elle n'est pas équivalente au présent contrôle non destructif.
