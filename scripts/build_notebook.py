"""Construit le notebook pédagogique OLV-6 à partir du code scientifique exécuté."""
from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def extract_definitions(filename: str, names: list[str]) -> str:
    """Copie exactement des fonctions/classes du script dans une cellule notebook."""
    path = ROOT / "scripts" / filename
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = []
    available = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            available.add(node.name)
            if node.name in names:
                found.append(ast.get_source_segment(source, node))
    missing = set(names) - available
    if missing:
        raise RuntimeError(f"Définitions absentes de {filename}: {sorted(missing)}")
    return "\n\n\n".join(found)


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


metric_code = extract_definitions("train_evaluate.py", ["metrics", "best_threshold"])
change_code = extract_definitions("train_evaluate.py", ["baseline_score", "cusum_score"])
siamese_code = extract_definitions("train_evaluate.py", ["PairDataset", "SiameseCNN"])
tcn_code = extract_definitions("train_evaluate.py", ["SequenceDataset", "Chomp1d", "TemporalBlock", "TCN"])
training_code = extract_definitions("train_evaluate.py", ["predict_torch", "train_neural"])
prep_code = extract_definitions("prepare_oqds.py", ["padded_indices", "extract_patch_array"])
hyperspectral_code = extract_definitions("complementary_report.py", ["hyperspectral_experiment"])

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3 (OLV6 local)", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.12"}

nb.cells = [
    md("""
# OliveSmart OLV-6 — comparaison scientifique de cinq algorithmes

## Introduction et objectifs

Ce notebook documente l'Étape 1 du projet **OliveSmart OLV-6 : détection temporelle des oliviers dépérissants ou morts**. Il compare sur les mêmes arbres OQDS :

1. un seuil de variation d'intensité ;
2. un CUSUM unilatéral ;
3. une Isolation Forest ;
4. un Siamese CNN sur deux observations RGB ;
5. un Temporal Convolutional Network (TCN) sur trois observations annuelles.

La cible `is.infect=True` désigne un arbre présent en 2019 puis absent en 2020/2021, interprété par les auteurs comme un retrait après infection probable. Elle ne constitue pas un diagnostic PCR individuel. Les rasters locaux nommés « NDVI » sont des rendus RGBA 8 bits sans calibration : les variables dérivées sont donc appelées **intensités**, jamais NDVI physique.

Les traitements volumineux restent organisés dans quatre scripts auditables : [audit](scripts/audit_datasets.py), [préparation](scripts/prepare_oqds.py), [entraînement/évaluation](scripts/train_evaluate.py) et [expériences complémentaires/rapport](scripts/complementary_report.py). Les principaux blocs scientifiques sont néanmoins reproduits directement ci-dessous.
"""),
    md("""
## 0 — Deux modes d'exécution

- `consultation` — mode par défaut : recharge les artefacts déjà calculés, vérifie leur cohérence et recrée seulement les tableaux/visualisations légères du notebook. **Aucun modèle n'est réentraîné.**
- `reproduction_complete` — repart des données brutes et exécute successivement audit, préparation, optimisation des cinq modèles, Test final et expériences complémentaires. Il faut également positionner `CONFIRMER_REENTRAINEMENT_COUTEUX=True` afin d'éviter un lancement accidentel.

La reproduction complète remplace les artefacts calculés portant les mêmes noms, mais ne modifie jamais les datasets bruts.
"""),
    code("""
from pathlib import Path
import json, os, random, subprocess, sys, time
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
import geopandas as gpd, rasterio, sklearn, torch
from IPython.display import display, Markdown, Image
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path.cwd().resolve()
assert ROOT.name == 'Project-JEE_DL', f'Ouvrir depuis la racine du projet : {ROOT}'
DATA, PROC = ROOT/'data', ROOT/'data/processed'
RESULTS, FIGURES, MODELS = ROOT/'outputs/results', ROOT/'outputs/figures', ROOT/'outputs/models'
SEED = 20261106
PATCH = 16
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXECUTION_MODE = 'consultation'  # ou 'reproduction_complete'
CONFIRMER_REENTRAINEMENT_COUTEUX = False

print({'mode': EXECUTION_MODE, 'python': sys.version.split()[0], 'torch': torch.__version__,
       'device': str(DEVICE), 'sklearn': sklearn.__version__, 'rasterio': rasterio.__version__})
"""),
    code("""
def executer_script(nom):
    print(f'Exécution de scripts/{nom} ...')
    subprocess.run([sys.executable, str(ROOT/'scripts'/nom)], cwd=ROOT, check=True)

if EXECUTION_MODE not in {'consultation', 'reproduction_complete'}:
    raise ValueError('Mode inconnu')
if EXECUTION_MODE == 'reproduction_complete':
    if not CONFIRMER_REENTRAINEMENT_COUTEUX:
        raise RuntimeError('Positionner CONFIRMER_REENTRAINEMENT_COUTEUX=True pour autoriser le recalcul complet.')
    for script in ['audit_datasets.py', 'prepare_oqds.py', 'train_evaluate.py', 'complementary_report.py']:
        executer_script(script)
else:
    print('Consultation : les résultats sauvegardés sont chargés sans réentraînement.')
"""),
    md("""
## 1 — Exploration des quatre datasets

L'audit parcourt réellement les fichiers, lit les métadonnées géospatiales, les schémas tabulaires, les dimensions des cubes/images et les annotations. Le CSV d'inventaire est exhaustif ; les JSON détaillent chaque source.
"""),
    code("""
audit_path = ROOT/'outputs/results/PHASE_A_AUDIT.md'
inventory_path = ROOT/'outputs/results/dataset_inventory.csv'
assert audit_path.exists() and inventory_path.exists(), 'Exécuter audit_datasets.py'
display(Markdown(audit_path.read_text(encoding='utf-8')))
inventory = pd.read_csv(inventory_path)
inventory_summary = (inventory.groupby(['dataset','role'], dropna=False)
                     .agg(fichiers=('path','size'), taille_Mio=('size_bytes', lambda x: x.sum()/2**20))
                     .reset_index())
display(inventory_summary)
"""),
    code("""
oqds_meta = json.loads((ROOT/'outputs/results/oqds_metadata.json').read_text(encoding='utf-8'))
neon_meta = json.loads((ROOT/'outputs/results/neon_metadata.json').read_text(encoding='utf-8'))
hs_meta = json.loads((ROOT/'outputs/results/hyperspectral_metadata.json').read_text(encoding='utf-8'))
crowns_meta = json.loads((ROOT/'outputs/results/crownsdb_metadata.json').read_text(encoding='utf-8'))

display(pd.DataFrame(oqds_meta['rasters'])[['name','width','height','bands','dtype','crs','resolution']])
print('OQDS vectoriel :', oqds_meta['vectors'][0]['rows'], 'arbres ; colonnes', oqds_meta['vectors'][0]['columns'])
print('NEON :', neon_meta[0]['rows'], 'couronnes ; années candidates', neon_meta[0]['date_values'])
print('Hyperspectral :', [(x['sheet'], x['rows']) for x in hs_meta['excel']], '; cubes =', len(hs_meta['cubes']))
print('CrownsDb :', crowns_meta['files'], 'fichiers ;', crowns_meta['voc_boxes'], 'boîtes VOC')
"""),
    md("""
### Interprétation de l'audit

OQDS est la seule source réunissant trois dates co-enregistrées et une cible compatible avec le benchmark commun. NEON concerne d'autres taxons ; le fichier présent ne permet finalement pas un suivi multi-annuel par individu. Le classeur hyperspectral porte une cible de stress distincte. CrownsDb documente les couronnes mais ne possède ni date sanitaire ni classe de dépérissement. Ces sources ne sont donc pas fusionnées artificiellement.
"""),
    md("## 2 — Nettoyage, contrôle qualité et visualisation des données"),
    code("""
qc = pd.read_csv(ROOT/'data/processed/oqds_qc_all.csv')
features = pd.read_csv(ROOT/'data/processed/oqds_features_splits.csv', dtype={'tree_id': str})
prep = json.loads((ROOT/'outputs/results/preparation_summary.json').read_text(encoding='utf-8'))
display(pd.Series(prep['invalid_counts'], name='observations_exclues').to_frame())
display(qc.groupby(['valid','target']).size().rename('effectif').reset_index())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
qc.target.value_counts().sort_index().plot.bar(ax=axes[0], color=['#6b8e23','#b84a3a'])
axes[0].set(title='Déséquilibre de la cible avant filtrage', xlabel='Classe', ylabel='Arbres')
sample = features.sample(min(12000, len(features)), random_state=SEED)
for label, color in [(0, '#6b8e23'), (1, '#b84a3a')]:
    z = sample[sample.target == label]
    axes[1].scatter(z.x, z.y, s=3 if label == 0 else 12, alpha=.45, color=color, label=f'classe {label}')
axes[1].set(title='Répartition spatiale (échantillon)', xlabel='Easting', ylabel='Northing'); axes[1].legend()
plt.tight_layout(); plt.show()
"""),
    md("""
Règles appliquées avant tout split : suppression des coordonnées dupliquées sans contradiction de label, exclusion des centres hors emprise/bordure, couverture alpha minimale de 80 %, et exclusion des patches RGB noirs à plus de 80 %. Les données brutes ne sont jamais modifiées.
"""),
    md("## 3 — Prétraitement et extraction des caractéristiques"),
    code(prep_code),
    md("""
Le script lit un grand raster à la fois afin de limiter la mémoire. Pour chaque arbre, il calcule moyenne, écart-type, percentiles 10/90 de l'intensité sur une fenêtre 5×5, variations absolues/relatives entre années, et deux patches RGB 16×16 (2019 et 2021). Les patches sont stockés en `uint8` dans `data/processed/` et ne sont convertis/normalisés qu'au chargement.
"""),
    code("""
feature_cols = [c for c in features.columns if c.startswith(('intensity_', 'delta_mean_', 'relative_delta_'))]
print('Caractéristiques numériques:', len(feature_cols)); display(pd.Series(feature_cols, name='variable').to_frame())

means = features.groupby('target')[[f'intensity_mean_{y}' for y in (2019,2020,2021)]].mean().T
means.index = [2019, 2020, 2021]
means.plot(marker='o', figsize=(7,4), title='Trajectoire moyenne d’intensité par classe')
plt.xlabel('Année'); plt.ylabel('Intensité 8 bits moyenne'); plt.show()
"""),
    md("## 4 — Découpage Train / Validation / Test"),
    code("""
split_table = features.groupby('split').target.agg(effectif='size', positifs='sum', prevalence='mean')
split_table['blocs'] = features.groupby('split').spatial_block.nunique()
display(split_table)
assert features.groupby('tree_id').split.nunique().max() == 1
assert features.groupby('spatial_block').split.nunique().max() == 1
assert all(features.groupby('split').target.nunique() == 2)
print('Contrôle réussi : aucun arbre ni bloc spatial partagé entre partitions.')
"""),
    md("""
Les blocs spatiaux mesurent 250 m. Sept folds groupés et stratifiés sont construits avec une graine fixe ; le fold Test est gelé. Un fold Validation à prévalence comparable est choisi **avant entraînement**, sans consulter les performances Test. Train sert à ajuster modèles et normalisations, Validation aux hyperparamètres/seuils, Test à l'évaluation finale unique.
"""),
    md("## 5 — Métriques et sélection du seuil sur Validation"),
    code(metric_code),
    md("La PR-AUC est le critère principal de sélection car les positifs ne représentent que 2–3 % des arbres. Accuracy, précision, rappel, F1, ROC-AUC et matrice de confusion complètent l'analyse. Le seuil maximisant F1 est calculé sur Validation uniquement."),
    md("## 6 — Baseline par seuil de variation"),
    code(change_code),
    code("""
validation = pd.read_csv(ROOT/'outputs/results/validation_experiments.csv')
display(validation.query("model == 'threshold'")[['config','pr_auc','roc_auc','f1','threshold']].sort_values('pr_auc', ascending=False))
"""),
    md("""
La baseline teste la baisse 2019→2021, la plus forte baisse annuelle et la baisse relative. Elle constitue un repère explicable, mais ne modélise ni textures ni dépendances spatiales. Sa faible PR-AUC Test confirme que l'intensité seule discrimine peu la classe rare.
"""),
    md("## 7 — CUSUM : détection statistique de rupture"),
    code("""
# La fonction cusum_score définie ci-dessus est exactement celle exécutée.
display(validation.query("model == 'cusum'")[['config','pr_auc','roc_auc','f1','threshold']].sort_values('pr_auc', ascending=False))
"""),
    md("Avec seulement trois dates, le CUSUM est une statistique courte de baisse cumulée, pas une datation précise de rupture. Six configurations combinent intensité absolue/relative et dérive `k`; la sélection repose exclusivement sur la PR-AUC Validation."),
    md("## 8 — Isolation Forest"),
    code("""
ISO_GRID = [
    {'n_estimators':150, 'max_samples':256,  'max_features':0.7},
    {'n_estimators':250, 'max_samples':512,  'max_features':1.0},
    {'n_estimators':350, 'max_samples':1024, 'max_features':0.8},
]

def ajuster_isolation_forest_train_seulement(x_train, y_train, configuration):
    normales = x_train[y_train == 0]
    return IsolationForest(**configuration, contamination='auto', random_state=SEED, n_jobs=-1).fit(normales)

display(validation.query("model == 'isolation_forest'")[['config','params','pr_auc','roc_auc','f1']].sort_values('pr_auc', ascending=False))
"""),
    md("Les variables numériques sont standardisées avec un `StandardScaler` ajusté sur Train. Isolation Forest apprend seulement les observations normales de Train. Une anomalie radiométrique n'étant pas nécessairement un retrait OQDS, cette approche présente beaucoup de faux positifs."),
    md("## 9 — Siamese CNN : données et architecture"),
    code(siamese_code),
    md("""
Les deux branches partagent exactement le même encodeur convolutionnel. Les embeddings 2019 et 2021 sont combinés par différence absolue et produit élément par élément, puis classifiés par un petit MLP. Les moyennes/écarts-types RGB sont calculés sur les patches Train seulement. `pos_weight` compense le déséquilibre sans dupliquer le Test.
"""),
    code("""
model_demo = SiameseCNN(width=16, dropout=.35)
print(model_demo)
print('Paramètres entraînables :', sum(p.numel() for p in model_demo.parameters() if p.requires_grad))
"""),
    md("## 10 — Entraînement commun des réseaux et optimisation Siamese"),
    code(training_code),
    code("""
SIAMESE_GRID = [
    {'lr':1e-3, 'width':8,  'dropout':.15},
    {'lr':4e-4, 'width':12, 'dropout':.25},
    {'lr':7e-4, 'width':16, 'dropout':.35},
]
display(validation.query("model == 'siamese_cnn'")[['config','params','fit_time_s','pr_auc','roc_auc','f1']].sort_values('pr_auc', ascending=False))
siamese_hist = pd.concat([pd.read_csv(p) for p in sorted((ROOT/'outputs/results').glob('history_siamese_*.csv'))])
display(siamese_hist)
"""),
    md("Chaque configuration est entraînée six époques avec Adam, pénalisation L2 et BCE pondérée. À chaque époque sont sauvegardées perte Train, perte Validation, ROC-AUC et PR-AUC Validation. Le checkpoint à meilleure PR-AUC est conservé."),
    md("## 11 — TCN : architecture temporelle"),
    code(tcn_code),
    code("""
tcn_demo = TCN(channels=12, kernel=2, dropout=.10)
print(tcn_demo)
print('Paramètres entraînables :', sum(p.numel() for p in tcn_demo.parameters() if p.requires_grad))
"""),
    md("""
Chaque arbre est représenté par quatre canaux — moyenne, écart-type, percentile 10 et percentile 90 — sur trois pas annuels. Le TCN utilise des convolutions causales dilatées, un résidu par bloc et la représentation du dernier pas. Trois dates restent une série très courte : l'architecture est testée conformément à l'énoncé, sans prétendre modéliser une dynamique longue.
"""),
    md("## 12 — Optimisation du TCN"),
    code("""
TCN_GRID = [
    {'lr':1e-3, 'channels':12, 'dropout':.10, 'kernel':2},
    {'lr':4e-4, 'channels':20, 'dropout':.25, 'kernel':2},
    {'lr':7e-4, 'channels':16, 'dropout':.35, 'kernel':3},
]
display(validation.query("model == 'tcn'")[['config','params','fit_time_s','pr_auc','roc_auc','f1']].sort_values('pr_auc', ascending=False))
tcn_hist = pd.concat([pd.read_csv(p) for p in sorted((ROOT/'outputs/results').glob('history_tcn_*.csv'))])
display(tcn_hist)
"""),
    md("Le TCN est entraîné huit époques avec le même protocole anti-fuite. Largeur, dropout, noyau et taux d'apprentissage varient entre les trois configurations ; la meilleure PR-AUC Validation fixe le checkpoint et le seuil."),
    md("## 13 — Recherche et comparaison globale des hyperparamètres"),
    code("""
selected = json.loads((ROOT/'outputs/results/selected_configurations.json').read_text(encoding='utf-8'))
selected_table = pd.DataFrame([{
    'modèle': name, 'configuration': rec['config'], 'paramètres': rec['params'],
    'PR-AUC Validation': rec['pr_auc'], 'ROC-AUC Validation': rec['roc_auc'], 'seuil': rec['threshold']
} for name, rec in selected.items()]).sort_values('PR-AUC Validation', ascending=False)
display(selected_table)
display(Image(filename=str(ROOT/'outputs/figures/learning_curves_neural.png')))
"""),
    md("## 14 — Évaluation finale sur le Test spatial figé"),
    code("""
test = pd.read_csv(ROOT/'outputs/results/model_comparison_test.csv')
predictions = pd.read_csv(ROOT/'outputs/results/test_predictions_all_models.csv', dtype={'tree_id':str})
display(test)

test_ids = set(features.loc[features.split == 'test','tree_id'])
assert all(set(z.tree_id) == test_ids for _, z in predictions.groupby('model'))
print(f'Mêmes {len(test_ids)} arbres Test pour les cinq modèles ; prédictions individuelles = {len(predictions):,}.')
"""),
    md("""
Le Siamese CNN obtient la meilleure discrimination (ROC-AUC ≈ 0,874 ; PR-AUC ≈ 0,220) et le meilleur F1 (≈ 0,322). Le TCN possède un rappel élevé mais une précision très faible au seuil Validation. L'accuracy seule est trompeuse dans ce contexte : prédire presque toujours la classe normale donnerait déjà environ 97,7 %.
"""),
    md("## 15 — Courbes ROC, précision-rappel et matrices de confusion"),
    code("""
display(Image(filename=str(ROOT/'outputs/figures/roc_curves_test.png')))
display(Image(filename=str(ROOT/'outputs/figures/precision_recall_curves_test.png')))
for model in test.model:
    display(Markdown(f'### Matrice de confusion — {model}'))
    display(Image(filename=str(ROOT/f'outputs/figures/confusion_matrix_{model}.png')))
"""),
    md("La courbe précision-rappel est la plus informative ici : elle tient compte de la rareté des positifs. Le Siamese conserve un avantage clair sur une partie utile de la courbe, sans que son seuil actuel rende le problème résolu."),
    md("## 16 — Analyse des erreurs et temps d'exécution"),
    code("""
errors = pd.read_csv(ROOT/'outputs/results/error_analysis.csv')
display(errors)
display(test[['model','training_tuning_time_s','inference_time_s','inference_ms_per_tree']].sort_values('training_tuning_time_s'))
display(Image(filename=str(ROOT/'outputs/figures/training_times.png')))
"""),
    md("""
Le Siamese est le plus coûteux à optimiser mais reste très rapide en inférence batch CPU. Les 155 faux négatifs du Test montrent qu'il ne doit pas être utilisé comme preuve phytosanitaire autonome. Le taux de pixels noirs est comparé entre erreurs et bonnes décisions, mais cette association descriptive ne démontre aucune causalité.
"""),
    md("## 17 — Analyse post-hoc des pixels noirs du Siamese CNN"),
    code("""
black = pd.read_csv(ROOT/'outputs/results/black_pixels_analysis.csv', dtype={'tree_id':str})
black_groups = pd.read_csv(ROOT/'outputs/results/black_pixels_group_metrics.csv')
display(black_groups)

label_black = black.groupby('y_true')[['black_fraction_2019','black_fraction_2021',
    'black_fraction_delta_2021_minus_2019','invalid_alpha_fraction_2019','invalid_alpha_fraction_2021']].agg(['mean','median'])
error_black = black.groupby('error_type')[['black_fraction_max','invalid_alpha_fraction_2019',
    'invalid_alpha_fraction_2021']].agg(['count','mean','median'])
display(label_black); display(error_black)
print('Décisions modifiées après neutralisation alpha=0 :', int(black.prediction_changed_after_neutralization.sum()))
print('Variation absolue moyenne du score :', black.score_change_after_neutralization.abs().mean())
display(Image(filename=str(ROOT/'outputs/figures/black_pixels_analysis.png')))
"""),
    md("""
Le noir strict est défini dans la préparation par `R=G=B=0`. Cette définition mélange potentiellement nodata, bordures, ombres très profondes et véritables valeurs noires. Le canal alpha des rendus RGBA alignés fournit un proxy plus spécifique d'invalidité (`alpha=0`), mais pas un masque physique certifié.

Les groupes sont définis sans utiliser les performances : aucun noir, puis moitié basse et moitié haute des proportions non nulles (seuil médian 0,242188). Le modèle est meilleur dans le groupe sans noir et ne détecte aucun des 13 positifs du groupe le plus noir. Les faux positifs ont presque zéro noir ; les faux négatifs n'en ont pas davantage que les prédictions correctes. Neutraliser les seuls pixels `alpha=0` par la moyenne RGB Train ne modifie aucune décision. Ces observations ne soutiennent pas un raccourci simple fondé sur les pixels noirs invalides, mais elles ne prouvent pas causalement que le réseau apprend la couronne : texture du sol, ombres valides, défauts d'enregistrement et contexte spatial restent possibles.
"""),
    md("## 18 — Contribution réelle des datasets complémentaires"),
    code(hyperspectral_code),
    code("""
complementary = json.loads((ROOT/'outputs/results/complementary_experiments.json').read_text(encoding='utf-8'))
display(pd.DataFrame({k: pd.Series(v) for k,v in complementary.items()}))
display(pd.read_csv(ROOT/'outputs/results/hyperspectral_cv_predictions.csv').head(10))
display(Image(filename=str(ROOT/'outputs/figures/hyperspectral_cv_scores.png')))
"""),
    md("""
- **NEON** : les identifiants répétés ont été contrôlés, mais aucune trajectoire multi-annuelle par individu n'est exploitable dans le fichier local ; aucun modèle temporel n'est annoncé.
- **Hyperspectral** : une régression logistique avec imputation, standardisation et PCA dans chaque pli produit de vraies prédictions hors-fold. Avec seulement six positifs, le résultat reste exploratoire et non comparable au Test OQDS.
- **CrownsDb** : les 33 479 boîtes ont été validées et leur échelle mesurée. Ce dataset peut soutenir une future détection/pré-segmentation de couronnes, mais pas la classification temporelle actuelle.
"""),
    md("## 19 — Conclusion et sélection du modèle final"),
    code("""
report_path = ROOT/'outputs/results/FINAL_SCIENTIFIC_REPORT.md'
display(Markdown(report_path.read_text(encoding='utf-8')))
"""),
    md("""
### Décision

Le **Siamese CNN** est retenu pour le prototype JEE, derrière un service Python REST versionné recevant deux patches RGB cohérents et retournant score, décision et version du modèle. Le seuil d'intensité reste un fallback explicable. Une validation externe sur une autre exploitation, avec vérité terrain indépendante et idéalement NIR/NDVI calibré, est indispensable avant usage opérationnel.
"""),
    md("## 20 — Carte des fichiers nécessaires et vérification de reproductibilité"),
    code("""
required_inputs = [
    'scripts/audit_datasets.py', 'scripts/prepare_oqds.py', 'scripts/train_evaluate.py',
    'scripts/complementary_report.py', 'requirements.txt',
    'data/raw/repository', 'data/neon/neon_crowns_dta.gpkg',
    'data/hyperspectral/merged_hs_olive_gr_tr.xlsx', 'data/crownsdb',
]
required_consultation = [
    'data/processed/oqds_features_splits.csv', 'outputs/results/validation_experiments.csv',
    'outputs/results/model_comparison_test.csv', 'outputs/results/test_predictions_all_models.csv',
    'outputs/results/complementary_experiments.json', 'outputs/models/siamese_cnn_best.pt',
    'outputs/models/tcn_best.pt', 'outputs/figures/roc_curves_test.png',
    'outputs/figures/precision_recall_curves_test.png', 'outputs/results/FINAL_SCIENTIFIC_REPORT.md',
]
for label, files in [('Reproduction complète', required_inputs), ('Consultation', required_consultation)]:
    missing = [p for p in files if not (ROOT/p).exists()]
    print(label, '— OK' if not missing else f'— manquants: {missing}')
    assert not missing

# Recalcul indépendant des métriques à partir des prédictions individuelles.
comparison_index = test.set_index('model')
for model, z in predictions.groupby('model'):
    recalculated = {
        'accuracy': accuracy_score(z.y_true, z.y_pred),
        'precision': precision_score(z.y_true, z.y_pred, zero_division=0),
        'recall': recall_score(z.y_true, z.y_pred, zero_division=0),
        'f1': f1_score(z.y_true, z.y_pred, zero_division=0),
        'roc_auc': roc_auc_score(z.y_true, z.score),
        'pr_auc': average_precision_score(z.y_true, z.score),
    }
    for metric_name, value in recalculated.items():
        assert np.isclose(value, comparison_index.loc[model, metric_name], atol=1e-12)

print('Vérification interne PASS : métriques recalculées, aucune fuite arbre/bloc.')
print('Le notebook courant a été exécuté en mode', EXECUTION_MODE)
display(Markdown((ROOT/'outputs/results/reproducibility_check.md').read_text(encoding='utf-8')))
"""),
    md("""
En mode consultation, les métriques sont relues depuis les prédictions sauvegardées et les contrôles de fuite sont recalculés, mais aucun poids n'est réoptimisé. En reproduction complète, les quatre scripts repartent des données brutes dans l'ordre scientifique et recréent les artefacts. Le manifeste `outputs/results/artifact_manifest.csv` fournit les empreintes SHA-256 des livrables contrôlés.
"""),
]

nbf.write(nb, ROOT / "OLV6_Comparaison_Complete.ipynb")
print(f"Notebook pédagogique construit : {len(nb.cells)} cellules ({sum(c.cell_type == 'code' for c in nb.cells)} code).")
