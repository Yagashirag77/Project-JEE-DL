# Project-JEE-DL — OliveSmart OLV-6

Détection temporelle des oliviers disparus (proxy de dépérissement) à partir d'images satellitaires
WorldView-2 2019–2021 (jeu OQDS), avec comparaison de cinq méthodes : seuil, CUSUM, Isolation Forest,
Siamese CNN et TCN.

## Contenu
- `OLV6_Comparaison_Complete.ipynb` : notebook principal (audit, préparation, entraînement, évaluation).
- `scripts/` : préparation des données, entraînement amélioré (`train_improved.py`), analyses
  complémentaires (`analyses_remarques.py`), ablation (`experiments_ablation.py`), génération des rapports PDF.
- `outputs/` : résultats, figures, modèles entraînés et rapports.
  - `runs/improved_v1/` : Siamese v2 (ensemble de 5 graines), analyses et ablation.
  - `corrections_rapport/` : passages LaTeX corrigés suite aux remarques du jury.

## Résultats principaux (test spatial, 10 205 arbres)
| Modèle | ROC-AUC | PR-AUC | F1 | Rappel |
|---|---|---|---|---|
| Siamese v2 (ensemble 5 graines) | 0,961 | 0,486 | 0,482 | 72,2 % |
| Siamese CNN initial | 0,874 | 0,220 | 0,322 | 34,6 % |
| TCN | 0,763 | 0,062 | 0,103 | 69,2 % |

## Installation
Voir `README_DEMARRAGE.md`. Les données (`data/`) ne sont pas versionnées.

## Auteurs
NAZIH Salma, GARTITE Yassine, TECHA Abderrahmane — 5IIR, 2026–2027.
