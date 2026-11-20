# OLV-6 — run amélioré `improved_v1`

Appareil `cpu`, graine de base 20261106, durée totale 66.0 min. Partition spatiale identique au run initial. Sélection et seuils fixés sur Validation, Test calculé une fois après gel (`frozen_choices.json`).

## 1. Sélection sur Validation

| name | patch | loss | augment | val_pr_auc | best_epoch | epochs_run | train_time_s |
|---|---|---|---|---|---|---|---|
| v2_p32_bce | 32 | bce_sqrtpw | True | 0.293 | 7 | 14 | 514.6953 |
| v2_p32_focal | 32 | focal | True | 0.2916 | 7 | 14 | 446.6653 |
| v2_p16_bce | 16 | bce_sqrtpw | True | 0.2649 | 5 | 12 | 141.5192 |
| orig_longer | 16 | bce_pw | False | 0.2279 | 9 | 16 | 45.1513 |

Retenue : **v2_p32_bce** (rappel : Siamese initial, PR-AUC Validation 0.1665 après 6 époques).

## 2. Stabilité sur 5 graines (Test, seuil F1 propre à chaque graine)

| seed | pr_auc | roc_auc | f1 | precision | recall | tp | fp |
|---|---|---|---|---|---|---|---|
| 0 | 0.4253 | 0.9631 | 0.4577 | 0.3497 | 0.6624 | 157 | 292 |
| 1 | 0.4224 | 0.9406 | 0.4926 | 0.4389 | 0.5612 | 133 | 170 |
| 2 | 0.4532 | 0.9561 | 0.4339 | 0.3118 | 0.7131 | 169 | 373 |
| 3 | 0.5075 | 0.9555 | 0.4344 | 0.3124 | 0.7131 | 169 | 372 |
| 4 | 0.5341 | 0.9627 | 0.4768 | 0.3521 | 0.7384 | 175 | 322 |

Moyenne ± écart-type : PR-AUC 0.4685 ± 0.0501, ROC-AUC 0.9556 ± 0.0091, F1 0.4591 ± 0.0259.

## 3. Comparaison Test (ensemble des graines vs run initial)

| model | pr_auc | roc_auc | f1 | precision | recall | tp | fp | fn | threshold |
|---|---|---|---|---|---|---|---|---|---|
| siamese_v2_ensemble@f1_opt | 0.4856 | 0.9606 | 0.4817 | 0.3615 | 0.7215 | 171 | 302 | 66 | 0.7185 |
| siamese_v2_ensemble@recall_60 | 0.4856 | 0.9606 | 0.3695 | 0.2379 | 0.827 | 196 | 628 | 41 | 0.5666 |
| original_siamese_cnn | 0.2205 | 0.8739 | 0.3222 | 0.3015 | 0.346 | 82 | 190 | 155 | 0.8815 |
| original_tcn | 0.0617 | 0.7635 | 0.1026 | 0.0554 | 0.692 | 164 | 2795 | 73 | 0.6234 |
| original_cusum | 0.039 | 0.6522 | 0.0712 | 0.0474 | 0.1435 | 34 | 684 | 203 | 104.12 |
| original_threshold | 0.0334 | 0.6254 | 0.0438 | 0.0324 | 0.0675 | 16 | 478 | 221 | 114.12 |
| original_isolation_forest | 0.0259 | 0.5558 | 0.0514 | 0.0267 | 0.692 | 164 | 5975 | 73 | -0.0951 |

`@recall_60` : seuil choisi sur Validation pour atteindre ≥ 60% de rappel (mode surveillance). `@f1_opt` : seuil F1 optimal Validation.

## 4. Intervalles de confiance (bootstrap de blocs spatiaux, 1000 tirages)

| model | pr_auc_lo | pr_auc_hi | roc_auc_lo | roc_auc_hi |
|---|---|---|---|---|
| ensemble | 0.1105 | 0.7293 | 0.9289 | 0.9757 |
| original_cusum | 0.0062 | 0.0896 | 0.5742 | 0.6888 |
| original_isolation_forest | 0.0044 | 0.0567 | 0.4623 | 0.6011 |
| original_siamese_cnn | 0.0325 | 0.4416 | 0.767 | 0.9103 |
| original_tcn | 0.0072 | 0.1468 | 0.6397 | 0.8148 |
| original_threshold | 0.0051 | 0.0782 | 0.5443 | 0.6704 |

Gain PR-AUC ensemble − Siamese initial : +0.2414 [IC95 +0.0754 ; +0.3510], part des tirages en faveur de l'ensemble : 100.0%.

## Limites

- Le Test avait déjà été consulté pour le run initial ; ce run n'utilise toutefois que la Validation pour ses choix. Une confirmation sur une zone externe reste nécessaire.
- TCN non modifié : avec 3 dates seulement, la séquence est trop courte pour qu'un modèle temporel apporte davantage.
- La cible reste indirecte (disparition d'arbre) et la radiométrie non calibrée.
