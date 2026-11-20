"""Rapport PDF complémentaire : réponses aux remarques prioritaires (1, 5, 8, 10, 13, 16, 19, 21, 22, 25, 27, 30 + 3, 20, 23)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Spacer

import build_pdf_report as B
from build_pdf_report import P, bullets, boxed, f, fig, table

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "runs" / "improved_v1"
AN = RUN / "experiments" / "analyses"
AB = RUN / "experiments" / "ablation"
FIG = AN / "figures"
OUT = ROOT / "outputs" / "OLV6_Reponses_Remarques.pdf"
W = 17 * cm


def pct(x, d=1): return f"{x:.{d}f}".replace(".", ",") + " %"
def n(x): return f"{int(x):,}".replace(",", " ")
def pm(m, s, d=3): return f"{f(m, d)} ± {f(s, d)}"


def ablation_chart(summ):
    order = ["ref", "three_dates", "no_share", "gray", "only_2021", "only_2019"]
    lab = {"ref": "Référence\n2019+2021", "three_dates": "3 dates\n2019+20+21", "no_share": "Sans partage\ndes poids",
           "gray": "Niveaux\nde gris", "only_2021": "2021\nseule", "only_2019": "2019 seule\n(contrôle)"}
    s = summ.set_index("variant").loc[[o for o in order if o in set(summ.variant)]]
    fig_, ax = plt.subplots(figsize=(7.6, 3.1))
    x = np.arange(len(s))
    ax.bar(x - 0.2, s.val_pr_auc_mean, 0.4, yerr=s.val_pr_auc_std, capsize=3, color="#8FB59A", label="Validation")
    ax.bar(x + 0.2, s.test_pr_auc_mean, 0.4, yerr=s.test_pr_auc_std, capsize=3, color="#2F6B3A", label="Test")
    ax.axhline(237 / 10205, ls="--", color="k", lw=.8)
    ax.set_xticks(x, [lab[i] for i in s.index], fontsize=7.8); ax.set_ylabel("PR-AUC (moy. ± é.-t., 3 graines)")
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Ablation du Siamese v2 (patch 32×32, mêmes graines)", fontsize=10)
    fig_.tight_layout(); p = FIG / "ablation_chart.png"; fig_.savefig(p, dpi=200); plt.close(fig_)
    return p


def main():
    S = json.loads((AN / "analyses_summary.json").read_text(encoding="utf-8"))
    excl = pd.read_csv(AN / "exclusions_by_class.csv")
    imb = pd.read_csv(AN / "imbalance_by_split.csv").set_index("split").loc[["train", "val", "test"]]
    yr = pd.read_csv(AN / "positives_by_removal_year.csv").set_index("split")
    leak = pd.read_csv(AN / "leakage_buffer.csv")
    ms = pd.read_csv(AN / "multiseed_summary.csv").set_index("model")
    msp = pd.read_csv(AN / "multiseed_per_seed.csv")
    ci = pd.read_csv(AN / "bootstrap_ci_all.csv").set_index("model")
    paired = pd.read_csv(AN / "paired_comparisons.csv")
    op = pd.read_csv(AN / "operating_points.csv")
    comp = pd.read_csv(RUN / "results" / "model_comparison_test.csv")
    abl = pd.read_csv(AB / "ablation_summary.csv") if (AB / "ablation_summary.csv").exists() else None
    abl_seed = pd.read_csv(AB / "ablation_per_seed.csv")
    L = S["leakage_distances"]; E = S["exclusions"]
    orig = {m: comp[comp.model == f"original_{m}"].iloc[0] for m in ("siamese_cnn", "tcn", "cusum", "threshold", "isolation_forest")}
    v2 = comp[comp.model == "siamese_v2_ensemble@f1_opt"].iloc[0]

    st = [P("Réponses aux remarques prioritaires", "title"),
          P("Projet OLV-6 — compléments expérimentaux au rapport final · novembre 2026", "sub")]
    st.append(boxed(
        "<b>Objet.</b> Ce document répond point par point aux remarques jugées prioritaires (1, 5, 8, 10, 13, 16, 19, 21, 22, 25, 27, 30), "
        "complétées par les points 3, 20 et 23 traités au passage. Toutes les expériences réutilisent la partition spatiale du rapport, "
        "sélectionnent sur Validation et n'écrasent aucun résultat initial (dossier <font face='Arial-Bold'>outputs/runs/improved_v1/</font>). "
        "« Siamese v2 » désigne le modèle amélioré (encodeur BatchNorm à 3 blocs, patchs 32×32, augmentation, arrêt anticipé, ensemble de 5 graines)."))
    st.append(Spacer(1, 6))
    st.append(table([["Remarque", "Réponse apportée", "Section"],
                     ["1 — Choix d'OQDS et limites de la cible", "Justification + preuves que la cible est surtout un arrachage de parcelles", "1"],
                     ["3 — Exclusions (bonus)", "Critères détaillés et effet sur les classes (test χ²)", "2"],
                     ["5 — Fuite spatiale", "Distances Test–Train, zones tampons, contrôle « 2019 seule »", "3"],
                     ["8 — Déséquilibre", "Prévalence, concentration par blocs, années de retrait", "4"],
                     ["10 — « NDVI »", "Nature des fichiers, options multispectrales", "5"],
                     ["13, 27 — 3 dates, ablation", "6 variantes × 3 graines", "6"],
                     ["16 — TCN sur 3 dates", "Champ récepteur, comparaison à un MLP", "7"],
                     ["19, 20 — Graines, IC95", "5 graines par modèle, bootstrap par blocs, tests appariés", "8"],
                     ["21, 22, 23 — Rappel et seuils", "5 stratégies de seuil, budget d'inspection", "9"],
                     ["25 — Exemples visuels", "VP / FP / FN / VN sur 2019–2020–2021", "10"],
                     ["30 — Validation terrain", "Protocole d'échantillonnage et d'annotation", "11"]],
                    [5.6 * cm, 9.6 * cm, 1.8 * cm]))

    # 1 ---------------------------------------------------------------------------
    st += [P("1. Choix d'OQDS et portée réelle de la cible (remarque 1)", "h1"),
           P("<b>Pourquoi OQDS est la source principale.</b> C'est la seule des quatre sources qui réunit simultanément (i) des oliviers, "
             "(ii) une identité d'arbre géoréférencée, (iii) plusieurs dates co-enregistrées sur la même emprise (2019, 2020, 2021, 0,5 m) "
             "et (iv) une étiquette de devenir par arbre. NEON ne contient pas d'oliviers ni de suivi multi-annuel exploitable, "
             "le jeu hyperspectral n'a que 76 observations sans géolocalisation commune, et CrownsDb n'a ni date ni état sanitaire."),
           P("<b>Ce que mesure vraiment l'étiquette.</b> Un positif est un arbre présent en 2019 et absent ensuite. Les nouvelles analyses montrent "
             "que cette cible correspond surtout à des <b>retraits groupés</b> plutôt qu'à la mort isolée d'arbres :"),
           *bullets([
               f"la distance médiane entre un positif Test et le positif le plus proche est de <b>{f(L['test_pos_nn_pos_median_m'], 1)} m</b>, "
               "soit l'arbre voisin dans le rang : les positifs se suivent ;",
               f"les 237 positifs Test se trouvent dans <b>{int(imb.loc['test','blocs_avec_positif'])} blocs sur {int(imb.loc['test','blocs'])}</b> (section 4) ;",
               "sur les exemples visuels (section 10), les vrais positifs les plus sûrs correspondent à des parcelles entièrement défrichées entre 2020 et 2021 ;",
               "à l'inverse, certains faux négatifs montrent une couronne encore visible en 2021 : l'étiquette peut être bruitée "
               "(co-enregistrement, arbre voisin, retrait partiel).",
           ]),
           P("<b>Conséquence.</b> Le modèle apprend à détecter la <i>disparition d'une couronne</i>, typiquement lors d'un arrachage. "
             "Trois problèmes doivent être distingués : (a) la détection de disparition, traitée ici ; (b) l'identification du dépérissement "
             "(couronne présente mais dégradée), pour laquelle OQDS ne fournit aucune étiquette ; (c) le diagnostic d'une maladie, "
             "qui exige une confirmation de laboratoire. Les résultats ne doivent pas être présentés comme une détection de maladie.")]

    # 3 ---------------------------------------------------------------------------
    rr = excl.set_index("invalid_reason")
    st += [P("2. Critères d'exclusion et effet sur les classes (remarque 3)", "h1"),
           P("Les règles sont appliquées <b>avant</b> le partitionnement. Une observation est exclue si l'un des critères suivants est vrai :"),
           table([["Critère", "Règle", "Négatifs", "Positifs", "Prévalence"],
                  ["Patch RGB majoritairement noir", "> 80 % de pixels R=G=B=0 en 2019 ou 2021", n(rr.loc['mostly_black_rgb','negatifs']), n(rr.loc['mostly_black_rgb','positifs']), pct(rr.loc['mostly_black_rgb','prevalence_%'], 2)],
                  ["Couverture alpha insuffisante", "< 80 % de pixels alpha > 0 (voisinage 5×5, une des 3 dates)", n(rr.loc['low_alpha_coverage','negatifs']), n(rr.loc['low_alpha_coverage','positifs']), pct(rr.loc['low_alpha_coverage','prevalence_%'], 2)],
                  ["Bordure d'image", "patch 16×16 hors emprise", n(rr.loc['outside_or_patch_border','negatifs']), n(rr.loc['outside_or_patch_border','positifs']), pct(rr.loc['outside_or_patch_border','prevalence_%'], 2)],
                  ["Conservés", "—", n(rr.loc['conservé','negatifs']), n(rr.loc['conservé','positifs']), pct(rr.loc['conservé','prevalence_%'], 2)]],
                 [4.3 * cm, 6.2 * cm, 2 * cm, 2 * cm, 2.5 * cm]),
           Spacer(1, 4),
           P(f"Les exclusions retirent {E['excluded_positives']} positifs sur {n(E['total_positives'])} (après dédoublonnage). La prévalence des exclus "
             f"({pct(E['excluded_prevalence_%'], 2)}) est inférieure à celle des conservés ({pct(E['after_prevalence_%'], 2)}) ; la différence est "
             f"significative (χ² = {f(E['chi2'], 1)}, p = {E['p_value']:.1e}) mais son effet est faible : la prévalence passe de "
             f"{pct(E['before_prevalence_%'], 2)} à {pct(E['after_prevalence_%'], 2)}. Les zones sans données (noir/transparent) sont donc "
             "légèrement moins associées aux retraits ; les performances ne s'appliquent pas à ces zones, qui devront être signalées "
             "« non évaluables » par l'application plutôt que classées.")]

    # 5 ---------------------------------------------------------------------------
    st += [P("3. Vérification de l'absence de fuite spatiale (remarque 5)", "h1"),
           P("La partition garantit qu'aucun arbre ni bloc de 250 m n'est partagé, mais des arbres Test peuvent jouxter des arbres Train "
             "de l'autre côté d'une frontière de bloc. Trois contrôles ont été ajoutés."),
           P("<b>(a) Proximité.</b> " +
             f"Distance médiane d'un arbre Test au plus proche arbre Train : {f(L['test_median_m'], 0)} m (Validation : {f(L['val_median_m'], 0)} m). "
             f"{pct(L['test_within_16m_%'])} des arbres Test sont à moins de 16 m d'un arbre Train (patchs 32×32 qui se chevauchent), "
             f"{pct(L['test_within_50m_%'])} à moins de 50 m."),
           P("<b>(b) Zones tampons.</b> Les métriques du Siamese v2 sont recalculées en retirant les arbres Test proches d'un arbre Train :"),
           table([["Tampon", "Arbres", "Positifs", "Prévalence", "PR-AUC", "PR-AUC / prévalence", "ROC-AUC"],
                  *[[f"{int(r.buffer_m)} m", n(r.arbres), int(r.positifs), pct(r["prevalence_%"], 2), f(r.pr_auc), f(r.pr_auc / (r["prevalence_%"] / 100), 1),
                     f(r.roc_auc)] for _, r in leak.iterrows()]],
                 [2 * cm, 2.2 * cm, 2 * cm, 2.3 * cm, 2.2 * cm, 3.6 * cm, 2.3 * cm]),
           Spacer(1, 4),
           P("La ROC-AUC, insensible à la prévalence, reste stable (0,955–0,965) jusqu'à 50 m. La PR-AUC baisse parce que la prévalence baisse "
             "(les positifs sont regroupés près des frontières) ; rapportée à la prévalence, elle reste ~21–24 fois le hasard. Au-delà de 100 m, "
             "il ne reste que 13 positifs, trop peu pour conclure. Rien n'indique que les performances proviennent du voisinage Train."),
           P("<b>(c) Contrôle « 2019 seule ».</b> Un Siamese identique entraîné sur la seule image de 2019 (avant tout retrait) ne peut réussir "
             "qu'en exploitant le contexte du lieu (type de parcelle, sol, position). Il obtient une PR-AUC Test de "
             f"<b>{pm(abl_seed[abl_seed.variant=='only_2019'].test_pr_auc.mean(), abl_seed[abl_seed.variant=='only_2019'].test_pr_auc.std())}</b> "
             "(hasard : 0,023), contre 0,43–0,45 pour le modèle bi-temporel avec les mêmes graines. Le modèle s'appuie donc sur le "
             "<b>changement</b> entre dates, pas sur un raccourci contextuel."),
           fig(FIG / "leakage_distance.png", 10.5, "Figure 1 — Distance de chaque arbre Test au plus proche arbre Train.")]

    # 8 ---------------------------------------------------------------------------
    st += [P("4. Analyse du déséquilibre des classes (remarque 8)", "h1"),
           table([["Partition", "Arbres", "Positifs", "Prévalence", "Négatifs / positif", "Blocs", "Blocs avec positif", "Positifs dans 10 % des blocs"],
                  *[[s.capitalize(), n(r.arbres), int(r.positifs), pct(r["prevalence_%"], 2), f(r.ratio_neg_pos, 1), int(r.blocs), int(r.blocs_avec_positif),
                     pct(r["part_positifs_top10%_blocs"], 0)] for s, r in imb.iterrows()]],
                 [1.9 * cm, 1.8 * cm, 1.6 * cm, 1.9 * cm, 2.3 * cm, 1.4 * cm, 2.6 * cm, 3.5 * cm]),
           Spacer(1, 4),
           P("Le déséquilibre a deux dimensions. <b>Global :</b> 2,3 à 3,0 % de positifs, soit 32 à 42 négatifs par positif. Un classifieur constant "
             "« sain » atteint 97,7 % d'accuracy en Test, d'où le choix de la PR-AUC (valeur du hasard 0,023). <b>Spatial :</b> les positifs sont "
             f"concentrés dans quelques blocs ; en Test, <b>tous les positifs sont dans {int(imb.loc['test','blocs_avec_positif'])} blocs</b>, dont certains "
             "ont une prévalence locale supérieure à 50 %. L'évaluation Test repose donc en pratique sur 5 foyers, ce qui explique la largeur des "
             "intervalles de confiance et l'écart Validation/Test (PR-AUC 0,29 contre 0,49)."),
           P(f"<b>Années de retrait :</b> {int(yr.loc['test','2020.0'])} des 237 positifs Test sont absents dès 2020 ; la comparaison 2019→2021 "
             "les voit donc toujours, mais l'image 2020 contient déjà l'information (voir section 6)."),
           P("<b>Traitement retenu :</b> pondération de la perte (BCE pondérée, poids √(négatifs/positifs) pour le Siamese v2), seuils choisis "
             "sur Validation et métriques insensibles à la classe majoritaire. La focal loss donne un résultat équivalent (PR-AUC Validation "
             "0,292 contre 0,293). Le sur-échantillonnage aléatoire n'a pas été retenu : avec des positifs groupés, il dupliquerait surtout les "
             "patchs de quelques parcelles et renforcerait le sur-apprentissage local."),
           fig(FIG / "imbalance_blocks.png", 14.5, "Figure 2 — Concentration des positifs : quelques blocs contiennent presque tous les retraits.")]

    # 10 --------------------------------------------------------------------------
    st += [P("5. Statut des images « NDVI » et options multispectrales (remarque 10)", "h1"),
           *bullets([
               "<b>Constat.</b> Les fichiers <font face='Arial-Bold'>*_WV2_NDVI.tif</font> sont des rasters RGBA 8 bits dont les trois bandes "
               "couleur sont identiques, sans métadonnée de calibration ni bande proche infrarouge. Ce sont des <i>rendus</i> en niveaux de gris "
               "d'un NDVI (ou d'une luminance), pas des réflectances : les valeurs ne peuvent pas être converties en NDVI physique. "
               "Toutes les variables qui en sont tirées sont donc nommées « intensité » et aucune conclusion physiologique n'en est déduite.",
               "<b>Effet mesuré.</b> Les méthodes fondées sur ces intensités (seuil, CUSUM, Isolation Forest, TCN) restent proches du hasard "
               "(PR-AUC 0,03–0,06), alors que le Siamese, qui exploite la forme et la texture RGB, atteint 0,22 puis 0,49.",
               "<b>Pour un vrai NDVI.</b> (i) Produit multispectral brut WorldView-2 (8 bandes dont deux proche infrarouge, ~2 m) sur les mêmes "
               "dates, à acquérir auprès du fournisseur : NDVI calculable après correction atmosphérique ; (ii) Pléiades Neo (multispectral ~1,2 m) "
               "pour de nouvelles campagnes ; (iii) Sentinel-2 (10 m, gratuit, revisite ~5 jours) : trop grossier pour une couronne isolée (4–6 m) "
               "mais utile à l'échelle de la parcelle et pour la saisonnalité ; (iv) drone multispectral pour la validation terrain (section 11).",
               "<b>Recommandation.</b> Ajouter un canal NIR/NDVI calibré est la piste la plus prometteuse pour passer de la détection de "
               "disparition à celle du <i>dépérissement</i> (couronne présente mais affaiblie), invisible en RGB 8 bits.",
           ])]

    # 13 / 27 ---------------------------------------------------------------------
    st += [P("6. Ablation et apport de la troisième date (remarques 13 et 27)", "h1"),
           P("Chaque variante est entraînée avec les mêmes hyperparamètres que le Siamese v2 et les <b>mêmes trois graines</b> ; la référence "
             "réutilise les checkpoints existants (scores reproduits à 10<super>-7</super> près).")]
    if abl is not None:
        lab = {"ref": "Référence (2019+2021, poids partagés, RGB)", "three_dates": "3 dates (2019+2020+2021)", "no_share": "Sans partage des poids",
               "gray": "Niveaux de gris", "only_2021": "Image 2021 seule", "only_2019": "Image 2019 seule (contrôle)"}
        a = abl.set_index("variant")
        rows = [["Variante", "Param.", "PR-AUC Val.", "PR-AUC Test", "ROC-AUC Test", "PR-AUC Test (ens. 3)"]]
        for v in ["ref", "three_dates", "no_share", "gray", "only_2021", "only_2019"]:
            if v in a.index:
                r = a.loc[v]
                rows.append([lab[v], n(r.n_params), pm(r.val_pr_auc_mean, r.val_pr_auc_std), pm(r.test_pr_auc_mean, r.test_pr_auc_std),
                             pm(r.test_roc_auc_mean, r.test_roc_auc_std), f(r.test_pr_auc_ens3)])
        st += [table(rows, [5.2 * cm, 1.9 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.4 * cm], highlight_rows=(1,)), Spacer(1, 4),
               fig(ablation_chart(abl), 14.5, "Figure 3 — Ablation : PR-AUC Validation et Test (moyenne ± écart-type sur 3 graines).")]
        st += ABLATION_TEXT(a)
    else:
        st.append(P("<i>Ablation en cours d'exécution — résultats à insérer.</i>"))

    # 16 --------------------------------------------------------------------------
    mlp, tcn = ms.loc["mlp_temporel"], ms.loc["tcn"]
    st += [P("7. Pertinence d'un TCN sur trois dates (remarque 16)", "h1"),
           P("Le TCN retenu comporte deux blocs résiduels (noyau 2, dilatations 1 et 2), soit un champ récepteur de 7 pas de temps pour une "
             "séquence de 3 : chaque sortie voit toute la série dès le premier bloc. Avec 4 statistiques × 3 dates = 12 valeurs en entrée, le TCN "
             "se réduit en pratique à une petite fonction non linéaire de ces 12 nombres, sans dépendance temporelle longue à exploiter."),
           P("Pour le vérifier, un <b>MLP</b> (12 → 32 → 1) a été entraîné sur exactement les mêmes entrées, avec le même protocole et 5 graines :"),
           table([["Modèle", "Paramètres", "PR-AUC Test", "ROC-AUC Test", "F1 Test"],
                  ["TCN (config. retenue)", n(msp[msp.model=='tcn'].n_params.iloc[0]), pm(tcn.pr_auc_mean, tcn.pr_auc_std), pm(tcn.roc_auc_mean, tcn.roc_auc_std), pm(tcn.f1_mean, tcn.f1_std)],
                  ["MLP de contrôle", n(msp[msp.model=='mlp_temporel'].n_params.iloc[0]), pm(mlp.pr_auc_mean, mlp.pr_auc_std), pm(mlp.roc_auc_mean, mlp.roc_auc_std), pm(mlp.f1_mean, mlp.f1_std)]],
                 [4.6 * cm, 2.4 * cm, 3.3 * cm, 3.3 * cm, 3.4 * cm]),
           Spacer(1, 4),
           P("Les deux modèles sont statistiquement équivalents : la convolution temporelle n'apporte rien sur trois dates. Le TCN est conservé "
             "dans le benchmark parce qu'il était imposé par le sujet et qu'il deviendra pertinent avec des séries plus longues (par exemple "
             "Sentinel-2 tous les 5 jours, soit ~70 dates par an). Sa faible précision (5,5 %) s'explique vraisemblablement par ses entrées : des statistiques "
             "d'intensité moyennées sur 5×5 pixels ne distinguent pas un arrachage d'un changement d'éclairage ou de sol.")]

    # 19 / 20 ---------------------------------------------------------------------
    names = {"siamese_v2": "Siamese v2", "siamese_cnn_initial": "Siamese initial", "tcn": "TCN", "mlp_temporel": "MLP (contrôle)", "isolation_forest": "Isolation Forest"}
    st += [P("8. Stabilité sur 5 graines et intervalles de confiance (remarques 19 et 20)", "h1"),
           P("Chaque modèle stochastique est ré-entraîné 5 fois avec sa configuration retenue et le même protocole (seuil F1 choisi sur "
             "Validation pour chaque graine). Seuil et CUSUM sont déterministes : pas de variance due à la graine."),
           table([["Modèle", "PR-AUC Test", "ROC-AUC Test", "F1 Test", "Rappel Test", "Précision Test"],
                  *[[names[m], pm(r.pr_auc_mean, r.pr_auc_std), pm(r.roc_auc_mean, r.roc_auc_std), pm(r.f1_mean, r.f1_std), pm(r.recall_mean, r.recall_std),
                     pm(r.precision_mean, r.precision_std)] for m, r in ms.loc[["siamese_v2", "siamese_cnn_initial", "tcn", "mlp_temporel", "isolation_forest"]].iterrows()]],
                 [3.3 * cm, 2.75 * cm, 2.75 * cm, 2.75 * cm, 2.75 * cm, 2.7 * cm], highlight_rows=(1,)),
           Spacer(1, 4),
           P("Le Siamese initial est instable : son rappel varie de ± 0,15 selon la graine, ce qui confirme que la valeur 0,346 du rapport n'était "
             "qu'un tirage. Le Siamese v2 est meilleur sur chaque graine et plus stable sur le rappel."),
           P("<b>IC95 par bootstrap de blocs spatiaux</b> (1 000 tirages de blocs avec remise, qui respectent l'autocorrélation spatiale) :"),
           table([["Modèle", "PR-AUC [IC95]", "ROC-AUC [IC95]", "F1 [IC95]"],
                  ["Siamese v2", f"{f(v2.pr_auc)} [{f(ci.loc['siamese_v2','pr_auc_lo'])} ; {f(ci.loc['siamese_v2','pr_auc_hi'])}]", f"{f(v2.roc_auc)} [{f(ci.loc['siamese_v2','roc_auc_lo'])} ; {f(ci.loc['siamese_v2','roc_auc_hi'])}]", f"{f(v2.f1)} [{f(ci.loc['siamese_v2','f1_lo'])} ; {f(ci.loc['siamese_v2','f1_hi'])}]"],
                  *[[{"siamese_cnn": "Siamese initial", "tcn": "TCN", "cusum": "CUSUM", "threshold": "Seuil", "isolation_forest": "Isolation Forest"}[m],
                     f"{f(o.pr_auc)} [{f(ci.loc[m,'pr_auc_lo'])} ; {f(ci.loc[m,'pr_auc_hi'])}]", f"{f(o.roc_auc)} [{f(ci.loc[m,'roc_auc_lo'])} ; {f(ci.loc[m,'roc_auc_hi'])}]",
                     f"{f(o.f1)} [{f(ci.loc[m,'f1_lo'])} ; {f(ci.loc[m,'f1_hi'])}]"] for m, o in orig.items()]],
                 [3.4 * cm, 4.5 * cm, 4.5 * cm, 4.6 * cm], highlight_rows=(1,)),
           Spacer(1, 4)]
    pp = paired[paired.metric == "pr_auc"]
    lab2 = {"siamese_v2": "Siamese v2", "siamese_cnn": "Siamese initial", "tcn": "TCN", "cusum": "CUSUM", "threshold": "Seuil", "isolation_forest": "Isolation Forest"}
    st += [P("<b>Comparaisons appariées</b> (mêmes tirages bootstrap pour les deux modèles ; p = part des tirages où A n'est pas meilleur que B) :"),
           table([["A", "B", "Δ PR-AUC [IC95]", "p"],
                  *[[lab2[r.A], lab2[r.B], f"{'+' if r.delta_mean>0 else ''}{f(r.delta_mean)} [{f(r.lo)} ; {f(r.hi)}]", f"{r.p_A_not_better:.3f}".replace(".", ",")]
                    for _, r in pp.iterrows() if (r.A, r.B) in {("siamese_v2", "siamese_cnn"), ("siamese_v2", "tcn"), ("siamese_cnn", "tcn"),
                                                              ("siamese_cnn", "cusum"), ("tcn", "cusum"), ("cusum", "threshold")}]],
                 [3.6 * cm, 3.6 * cm, 6 * cm, 3.8 * cm]),
           Spacer(1, 4),
           P("Toutes les différences entre le Siamese (initial ou v2) et les autres méthodes sont significatives. Les écarts entre les baselines "
             "faibles (TCN, CUSUM, seuil) sont significatifs mais minimes en valeur absolue (≤ 0,02 de PR-AUC).")]

    # 21-23 -----------------------------------------------------------------------
    def oprows(m):
        g = op[op.model == m]
        return [[lab2[m], r.strategie, f"{n(r.inspectes)} ({pct(r['inspectes_%'])})", f"{int(r.tp)}/237", int(r.fp), pct(100 * r.rappel, 0), pct(100 * r.precision, 0),
                 f(r.inspections_par_vrai_positif, 1)] for _, r in g.iterrows()]
    st += [P("9. Faible rappel et stratégies de seuil (remarques 21, 22 et 23)", "h1"),
           P("<b>Diagnostic du rappel de 34,6 %.</b> Trois causes sont identifiées : (i) un modèle sous-entraîné (6 époques, PR-AUC Validation "
             "encore croissante) ; (ii) un seuil très conservateur (0,88) choisi sur une courbe Validation instable — le rappel varie de 0,32 "
             "± 0,15 entre graines ; (iii) un contexte de 16×16 pixels trop étroit. Le Siamese v2 corrige (i) et (iii) ; (ii) est traité "
             "ci-dessous en rendant le seuil explicite selon l'objectif."),
           P("Cinq stratégies sont fixées <b>sur Validation</b>, puis appliquées une seule fois au Test : F1 optimal, rappel ≥ 60 %, rappel ≥ 80 %, "
             "et deux <b>budgets d'inspection</b> (signaler 2 % ou 5 % des arbres, ce qui borne le travail des agents)."),
           table([["Modèle", "Stratégie", "Inspectés", "Trouvés", "FP", "Rappel", "Précision", "Inspections / VP"],
                  *oprows("siamese_v2"), *oprows("siamese_cnn"), *oprows("tcn")],
                 [2.4 * cm, 3.0 * cm, 2.5 * cm, 1.8 * cm, 1.3 * cm, 1.5 * cm, 1.8 * cm, 2.7 * cm], highlight_rows=(1, 4)),
           Spacer(1, 4),
           P("<b>Lecture opérationnelle.</b> Avec un budget de 2 % d'arbres inspectés, le Siamese v2 retrouve 55 % des retraits avec une alerte sur "
             "deux correcte ; avec 5 %, 76 % des retraits pour environ 3 inspections par vrai positif. Pour viser 80 % de rappel, le Siamese "
             "initial imposerait d'inspecter 31 % des arbres (16 inspections par vrai positif) et le TCN 55 %, contre 13 % pour le Siamese v2. "
             "Recommandation : mode <b>« budget 5 % »</b> par défaut, et mode <b>« rappel 80 % »</b> lorsqu'une campagne de terrain dense est possible.")]

    # 25 --------------------------------------------------------------------------
    st += [P("10. Exemples visuels VP / FP / FN / VN (remarques 25 et 4)", "h1"),
           fig(FIG / "examples_confusion_compact.png", 10.5,
               "Figure 4 — Siamese v2, seuil F1. Fenêtres de 64×64 pixels (32 m) ; le cadre jaune est le patch 32×32 vu par le modèle. "
               "Les images ne permettent pas de confirmer l'état sanitaire réel des arbres."),
           *bullets([
               "<b>Vrai positif :</b> rangées d'oliviers présentes en 2019 et 2020, puis parcelle défrichée en 2021 — cas typique d'arrachage.",
               "<b>Faux positif :</b> les arbres sont toujours présents, mais la couleur du sol et l'éclairage changent fortement entre les dates "
               "(sol rose en 2020–2021, ombres plus marquées). Les différences radiométriques entre acquisitions sont une source d'erreur fréquente parmi les faux positifs examinés.",
               "<b>Faux négatif :</b> une couronne semble encore visible en 2021 à l'emplacement étiqueté : erreur d'étiquette, retrait partiel ou "
               "décalage de géolocalisation possibles — seule une vérification terrain peut trancher (section 11).",
               "<b>Vrai négatif :</b> parcelle stable malgré des variations de teinte.",
           ]),
           P("Une planche plus large (3 exemples par catégorie) est disponible dans "
             "<font face='Arial-Bold'>experiments/analyses/figures/examples_confusion.png</font>. Aucune image d'arbre « dépérissant » ne peut être "
             "montrée de façon fiable : OQDS n'étiquette pas cet état.")]

    # 30 --------------------------------------------------------------------------
    st += [P("11. Protocole de validation terrain proposé (remarque 30)", "h1"),
           *bullets([
               "<b>Objectif :</b> estimer, pour les arbres signalés, la part réellement morts, arrachés, dépérissants ou sains (artefact d'acquisition), "
               "et mesurer le taux de retraits manqués.",
               "<b>Échantillonnage stratifié :</b> ~200 alertes réparties par classe de score (0,5–0,7 / 0,7–0,9 / &gt; 0,9) et par bloc, plus ~200 arbres "
               "non signalés tirés au hasard (dont une part près des alertes). Une proportion estimée sur 200 arbres a une marge d'environ ± 7 points à 95 %.",
               "<b>Relevé :</b> visite au GPS (précision sub-métrique) ou vol de drone RGB/multispectral, photo géolocalisée, statut parmi "
               "{présent sain, présent dépérissant, mort sur pied, arraché/souche, remplacé, introuvable}, cause apparente si connue.",
               "<b>Annotation indépendante :</b> deux annotateurs ne connaissant pas le score du modèle ; accord mesuré par le κ de Cohen, "
               "désaccords tranchés par un agronome.",
               "<b>Calendrier :</b> visite au plus tôt après la dernière acquisition, en notant le délai : un arbre arraché après la date de l'image "
               "ne doit pas être compté comme une erreur.",
               "<b>Exploitation :</b> précision et rappel « terrain » avec IC95 ; recalibrage du seuil ; ajout éventuel d'une classe « dépérissant » "
               "pour un futur modèle ; retour des cas d'artefacts radiométriques dans l'ensemble d'entraînement.",
               "<b>Extension (remarque 29) :</b> le même protocole, appliqué à une oliveraie marocaine avec des images de même résolution, constituerait "
               "le test de généralisation externe qui manque aujourd'hui.",
           ])]

    st += [P("Fichiers produits", "h2"),
           P("Scripts : <font face='Arial-Bold'>analyses_remarques.py</font>, <font face='Arial-Bold'>experiments_ablation.py</font>, "
             "<font face='Arial-Bold'>build_report_remarques.py</font>. Résultats : <font face='Arial-Bold'>outputs/runs/improved_v1/experiments/</font> "
             "(tableaux CSV, figures, checkpoints). Graine de base 20261106 ; calcul sur CPU.")]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.8 * cm,
                            title="OLV-6 — Réponses aux remarques", author="Projet OLV-6")
    doc.build(st, onFirstPage=B.on_page, onLaterPages=B.on_page)
    print(OUT)


def ABLATION_TEXT(a):
    g = lambda v, c="test_pr_auc_mean": a.loc[v, c] if v in a.index else float("nan")
    ref = g("ref")
    def cmp(v):
        d = round(g(v) - ref, 3) + 0.0
        return f"{f(g(v))} ({'+' if d >= 0 else ''}{f(d)})"
    gv = lambda v: g(v, "val_pr_auc_mean")
    vref = gv("ref")
    def dv(v):
        d = round(gv(v) - vref, 3) + 0.0
        return f"val. {f(gv(v))} ({'+' if d >= 0 else ''}{f(d)})"
    out = [P("<b>Lecture.</b> PR-AUC moyenne (3 graines) et écart à la référence (val. " + f(vref) + ", test " + f(ref) + "). "
             "La validation (11 blocs, positifs mieux répartis) est le critère principal ; le test (positifs concentrés dans 5 blocs) "
             "est plus bruité, d'où des écarts-types de 0,05 à 0,08 entre graines.")]
    items = []
    if "only_2019" in a.index:
        items.append(f"<b>2019 seule</b> : test {cmp('only_2019')}, {dv('only_2019')}. Au niveau de la prévalence (~0,04) : l'image "
                     "initiale seule ne permet pas de prédire la disparition, donc <b>aucun raccourci contextuel</b> (sol, parcelle, bord d'image).")
    if "only_2021" in a.index:
        items.append(f"<b>2021 seule</b> : test {cmp('only_2021')}, {dv('only_2021')}. L'image finale porte déjà une partie du signal "
                     "(absence de couronne), mais la <b>comparaison avec 2019 apporte environ +0,12 à +0,15</b> de PR-AUC : la formulation "
                     "bi-temporelle (siamoise) est justifiée.")
    if "gray" in a.index:
        items.append(f"<b>Niveaux de gris</b> : test {cmp('gray')}, {dv('gray')}. La couleur aide nettement en validation (-0,06 sans elle) "
                     "mais pas sur le test, dont la variance est forte : l'apport de la couleur est <b>probable mais modéré</b> ; "
                     "l'essentiel du signal est structurel (forme et ombre de la couronne).")
    if "no_share" in a.index:
        items.append(f"<b>Sans partage des poids</b> : test {cmp('no_share')}, {dv('no_share')}, avec "
                     f"{n(a.loc['no_share', 'n_params'])} paramètres contre {n(a.loc['ref', 'n_params'])}. Performance équivalente pour "
                     "deux fois plus de paramètres : le <b>partage des poids est justifié</b> (même capteur, même résolution, "
                     "et un encodeur commun rend les embeddings directement comparables).")
    if "three_dates" in a.index:
        d3v, d3t = gv("three_dates") - vref, g("three_dates") - ref
        if d3v > 0.02 and d3t > 0:
            concl = ("l'image intermédiaire <b>améliore la détection</b> sur les deux partitions : elle aide à distinguer une disparition "
                     "réelle d'un artefact ponctuel (ombre, angle de prise de vue) et justifie l'usage des 3 dates disponibles.")
        elif d3v > 0.02:
            concl = ("gain en validation mais pas confirmé sur le test : apport de 2020 <b>plausible, non démontré</b> avec 3 graines.")
        else:
            concl = ("pas de gain significatif : sur ce jeu, l'information utile est contenue dans le couple 2019/2021, "
                     "ce qui rejoint le constat TCN ≈ MLP de la section 7.")
        items.append(f"<b>Trois dates</b> (remarque 13) : test {cmp('three_dates')}, {dv('three_dates')} ; {concl}")
    return out + bullets(items)


if __name__ == "__main__":
    main()
