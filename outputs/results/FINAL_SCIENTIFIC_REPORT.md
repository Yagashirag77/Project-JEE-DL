# OliveSmart OLV-6 — comparaison scientifique de cinq algorithmes

## Objectif et données

L'expérience compare cinq pipelines sur les mêmes arbres OQDS et une séparation spatiale figée. OQDS contient trois images WorldView-2 co-enregistrées (2019–2021) et des arbres considérés infectés lorsqu'un objet présent en 2019 est absent ensuite. Cette définition est cohérente avec l'article source, mais reste une infection probable fondée sur le retrait et non un diagnostic PCR individuel ([publication OQDS](https://pmc.ncbi.nlm.nih.gov/articles/PMC12142346/)).

L'audit réel a couvert quatre sources. OQDS est seul compatible avec le benchmark temporel commun. NEON contient d'autres taxons et pas de cible OQDS; le classeur hyperspectral décrit un stress spectral très déséquilibré; CrownsDb fournit des boîtes de couronnes sans suivi sanitaire temporel.

## Préparation et protocole anti-fuite

Après retrait de 104 coordonnées dupliquées et 5026 observations invalides, 71956 arbres restent. Les six rasters OQDS partagent EPSG:32633, une résolution de 0,5 m, la même transformation et la même emprise. Les fichiers appelés NDVI sont en réalité des rendus RGBA `uint8` sans calibration; les variables correspondantes sont donc nommées intensités.

Une partition par blocs de 250 m donne Train/Validation/Test sans arbre ni bloc partagé. Toutes les transformations sont ajustées sur Train. Trois configurations sont comparées pour le seuil, Isolation Forest, Siamese et TCN; six pour CUSUM. PR-AUC Validation est le critère fixé à l'avance, adapté aux 2–3 % de positifs. Les seuils de décision sont aussi choisis sur Validation. Test n'est calculé qu'après gel des choix.

## Résultats Test

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc | tn | fp | fn | tp | threshold | training_tuning_time_s | inference_time_s | inference_ms_per_tree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| siamese_cnn | 0.9662 | 0.3015 | 0.346 | 0.3222 | 0.8739 | 0.2205 | 9778 | 190 | 155 | 82 | 0.8815 | 70.7332 | 0.3963 | 0.0388 |
| tcn | 0.719 | 0.0554 | 0.692 | 0.1026 | 0.7635 | 0.0617 | 7173 | 2795 | 73 | 164 | 0.6234 | 21.3124 | 0.0562 | 0.0055 |
| cusum | 0.9131 | 0.0474 | 0.1435 | 0.0712 | 0.6522 | 0.039 | 9284 | 684 | 203 | 34 | 104.12 | 0.0564 | 0.0003 | 0.0 |
| threshold | 0.9315 | 0.0324 | 0.0675 | 0.0438 | 0.6254 | 0.0334 | 9490 | 478 | 221 | 16 | 114.12 | 0.0375 | 0.0001 | 0.0 |
| isolation_forest | 0.4073 | 0.0267 | 0.692 | 0.0514 | 0.5558 | 0.0259 | 3993 | 5975 | 73 | 164 | -0.0951 | 1.6827 | 0.1777 | 0.0174 |

Le Siamese CNN est nettement premier: ROC-AUC 0.874, PR-AUC 0.220, F1 0.322. Il détecte 82 des 237 positifs, avec 190 faux positifs et 155 faux négatifs au seuil fixé sur Validation. Le TCN classe mieux que les méthodes simples mais son seuil Validation produit 2795 faux positifs. L'Isolation Forest est mal adaptée à cette représentation: une anomalie radiométrique n'est pas équivalente à un retrait OQDS.

Les écarts Validation/Test et les performances modestes des baselines s'expliquent notamment par la rareté et l'agrégation spatiale des cas, la radiométrie 8 bits, trois dates seulement, et le fait que le changement d'apparence n'est pas spécifique à la maladie. Les erreurs ne prouvent pas une cause unique; `error_analysis.csv` compare seulement leur proportion de pixels noirs.

## Expériences complémentaires réelles

- **NEON**: 0 individus disposent d'au moins deux années mesurables. Aucun individu n'a deux années de détection distinctes exploitables; aucun changement longitudinal n'est estimé. Les répétitions présentes ne justifient donc ni modèle temporel ni fusion avec OQDS.
- **Hyperspectral**: régression logistique après standardisation/PCA dans une CV stratifiée à trois plis, 76 observations et seulement 6 positives. PR-AUC hors-fold = 0.262; résultat exploratoire, non comparable à OQDS. Les huit cubes n'ont pas de clé permettant une correspondance fiable avec les 76 lignes.
- **CrownsDb**: 33479 boîtes vérifiées; aire relative médiane 0.0154. Usage justifié: détection/pré-entraînement de couronnes, pas classification temporelle.

## Analyse post-hoc des pixels noirs du Siamese CNN

### Protocole expérimental

L'analyse utilise exclusivement les 10 205 arbres du Test spatial figé, les patches RGB 16×16 existants, le checkpoint `siamese_cnn_best.pt` et le seuil déjà choisi sur Validation. Aucun modèle n'est réentraîné et aucun seuil n'est optimisé sur les performances Test.

Dans `prepare_oqds.py`, un pixel noir est défini strictement par `R=G=B=0`. Cette règle ne sépare pas à elle seule une zone invalide, une ombre profonde ou une valeur noire valide. Le canal `alpha=0` des rasters RGBA alignés est utilisé comme proxy plus spécifique de transparence/nodata. Il reste un masque fourni par le rendu, pas une vérité physique certifiée.

Comme 80,67 % des arbres n'ont aucun pixel noir strict dans les deux dates, les terciles ordinaires seraient dégénérés. Les groupes sont donc définis à partir de la distribution seule : (1) aucun noir ; (2) proportion non nulle inférieure ou égale à la médiane des valeurs non nulles, 0,242188 ; (3) proportion supérieure à cette médiane.

| Groupe | n | Positifs | Accuracy | Précision | Rappel | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aucun noir | 8 232 | 213 | 0,961 | 0,300 | 0,376 | 0,333 | 0,895 | 0,236 | 7 832 | 187 | 133 | 80 |
| Noir non nul ≤ 0,242188 | 987 | 11 | 0,988 | 0,400 | 0,182 | 0,250 | 0,896 | 0,150 | 973 | 3 | 9 | 2 |
| Noir non nul > 0,242188 | 986 | 13 | 0,987 | 0,000 | 0,000 | 0,000 | 0,811 | 0,059 | 973 | 0 | 13 | 0 |

Les négatifs ont en moyenne 5,76 % de noir en 2019 et 5,77 % en 2021, contre 2,70 % aux deux dates pour les positifs. La variation moyenne 2021−2019 est seulement +0,018 point de pourcentage chez les négatifs et nulle chez les positifs. La médiane est zéro dans les deux classes.

Les prédictions correctes présentent en moyenne 5,84 % de noir maximal, contre 4,12 % pour les faux négatifs et 0,012 % pour les faux positifs. Les erreurs ne présentent donc pas davantage de noir que les bonnes décisions. En particulier, les faux positifs ne sont pas associés à des patches fortement noirs.

### Test de sensibilité sans réentraînement

Seulement 34 arbres Test (0,33 %) contiennent au moins un pixel `alpha=0`. Ces pixels sont remplacés par la moyenne RGB calculée sur Train, qui correspond à une valeur normalisée neutre. Les scores originaux recalculés depuis le checkpoint reproduisent les scores sauvegardés à moins de 2×10⁻⁶.

La neutralisation ne modifie aucune des 10 205 classes prédites. La variation absolue moyenne du score est 0,000318 sur tout le Test et 0,0955 parmi les 34 arbres concernés ; la variation maximale atteint 0,402, mais reste sans franchissement du seuil de décision. ROC-AUC passe de 0,873907 à 0,873756 et PR-AUC de 0,220459 à 0,220329. Les TP, TN, FP et FN restent respectivement 82, 9 778, 190 et 155.

### Interprétation et limites

Ces résultats ne montrent pas d'indice d'un raccourci simple fondé sur les pixels noirs invalides : les performances sont maximales sans noir, les faux positifs n'en contiennent presque pas et la neutralisation `alpha=0` ne change aucune décision. En revanche, le groupe très noir contient seulement 13 positifs et aucun vrai positif, ce qui signale une faiblesse de robustesse dans ce sous-groupe.

Cette analyse n'est pas une preuve causale. La majorité des pixels noirs stricts ont `alpha>0` et peuvent correspondre à des ombres ou à des valeurs valides impossibles à distinguer avec les métadonnées disponibles. Le test ne contrôle pas les textures du sol, les variations d'illumination, les défauts de co-enregistrement ou d'autres artefacts contextuels. Une étude causale plus forte nécessiterait des masques de couronne fiables, des occlusions contrôlées et éventuellement un nouvel entraînement avec augmentation/masking explicites.

## Recommandation pour l'application JEE

Retenir le **Siamese CNN** comme moteur principal, avec un service Python d'inférence séparé exposé à JEE par REST. Le service doit recevoir deux patches RGB géoréférencés cohérents, appliquer la normalisation sauvegardée, renvoyer score, classe et version du modèle. Conserver le seuil d'intensité comme contrôle explicable/fallback, jamais comme équivalent de précision. Avant production, valider sur une nouvelle exploitation avec un test externe et, idéalement, de vrais rasters NDVI/NIR calibrés et une vérité terrain indépendante.

L'analyse des pixels noirs ne justifie pas d'abandonner le checkpoint actuel pour le prototype : aucun raccourci simple lié à `alpha=0` n'est observé. Le prétraitement doit néanmoins être renforcé avant production avec un masque explicite de validité/couronne et des contrôles d'illumination. Un prochain entraînement, autorisé séparément, devrait intégrer augmentation d'ombres/occlusions, canal de masque et validation externe afin de tester la robustesse du sous-groupe très noir.

Le TCN peut être conservé comme modèle secondaire très rapide, mais sa précision actuelle est insuffisante pour déclencher seul une alerte. Toute nouvelle région doit provoquer une validation de dérive, pas un ajustement sur le Test actuel.

## Reproductibilité et limites

Graine 20261106; Python local `.venv`; PyTorch CPU; prédictions individuelles et configurations complètes enregistrées. Les modèles neuronaux ont leurs pertes Train/Validation, ROC-AUC et PR-AUC par époque. Les limites majeures sont la cible indirecte, l'absence de NDVI physique, seulement trois dates, la dépendance spatiale résiduelle entre blocs voisins et l'extrême déséquilibre.
