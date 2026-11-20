"""Génère le rapport PDF OLV-6 (run initial + run amélioré) à partir des résultats sauvegardés."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "outputs" / "results"
RUN = ROOT / "outputs" / "runs" / "improved_v1"
RES, FIG = RUN / "results", RUN / "figures"
OUT = ROOT / "outputs" / "RAPPORT_OLV6.pdf"

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Arial", str(FONTS / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONTS / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", str(FONTS / "ariali.ttf")))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-Bold")

ACCENT = colors.HexColor("#2F6B3A")
LIGHT = colors.HexColor("#EEF4EF")
GRID = colors.HexColor("#C9D3CB")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Arial-Bold", fontSize=19, leading=23, textColor=ACCENT, spaceAfter=4),
    "sub": ParagraphStyle("s", parent=ss["Normal"], fontName="Arial", fontSize=10, textColor=colors.HexColor("#555555"), alignment=TA_CENTER, spaceAfter=10),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Arial-Bold", fontSize=13, leading=16, textColor=ACCENT, spaceBefore=10, spaceAfter=5),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Arial-Bold", fontSize=11, leading=14, spaceBefore=6, spaceAfter=3),
    "p": ParagraphStyle("p", parent=ss["Normal"], fontName="Arial", fontSize=9.5, leading=13.2, spaceAfter=5),
    "b": ParagraphStyle("b", parent=ss["Normal"], fontName="Arial", fontSize=9.5, leading=13, leftIndent=12, bulletIndent=2, spaceAfter=2),
    "cap": ParagraphStyle("c", parent=ss["Normal"], fontName="Arial-Italic", fontSize=8.3, leading=10.5, textColor=colors.HexColor("#555555"), alignment=TA_CENTER, spaceAfter=8),
    "cell": ParagraphStyle("cell", parent=ss["Normal"], fontName="Arial", fontSize=8.2, leading=10),
    "box": ParagraphStyle("box", parent=ss["Normal"], fontName="Arial", fontSize=9.5, leading=13.2),
}


def P(t, s="p"): return Paragraph(t, S[s])
def bullets(items): return [Paragraph(i, S["b"], bulletText="•") for i in items]


def table(rows, widths, highlight_rows=()):
    data = [[Paragraph(f"<b>{c}</b>" if r == 0 else str(c), S["cell"]) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, GRID), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for r in highlight_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#E3EFE5")))
    t.setStyle(TableStyle(style))
    return t


def boxed(text):
    t = Table([[Paragraph(text, S["box"])]], colWidths=[17 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def fig(path, width_cm, caption):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return KeepTogether([Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w), P(caption, "cap")])


def f(x, d=3): return f"{x:.{d}f}".replace(".", ",")


def make_charts(val_sel, comp, ci):
    names = {"ensemble": "Siamese v2\n(5 graines)", "original_siamese_cnn": "Siamese\ninitial", "original_tcn": "TCN",
             "original_cusum": "CUSUM", "original_threshold": "Seuil", "original_isolation_forest": "Isolation\nForest"}
    pr = {"ensemble": comp.loc[comp.model.str.startswith("siamese_v2"), "pr_auc"].iloc[0]}
    for m in names:
        if m != "ensemble":
            pr[m] = comp.loc[comp.model == m, "pr_auc"].iloc[0]
    ci = ci.set_index("model")
    order = list(names)
    fig1, ax = plt.subplots(figsize=(7.5, 3.2))
    vals = [pr[m] for m in order]
    err = [[pr[m] - ci.loc[m, "pr_auc_lo"] for m in order], [ci.loc[m, "pr_auc_hi"] - pr[m] for m in order]]
    cols = ["#2F6B3A"] + ["#8FB59A"] + ["#B8C4BA"] * 4
    ax.bar(range(len(order)), vals, color=cols, yerr=err, capsize=4, ecolor="#444")
    for i, v in enumerate(vals):
        ax.text(i + 0.1, v + 0.012, f"{v:.3f}", fontsize=8, ha="left")
    ax.axhline(237 / 10205, ls="--", color="k", lw=0.8)
    ax.text(-0.45, 237 / 10205 + 0.012, "hasard", fontsize=7.5, ha="left")
    ax.set_xticks(range(len(order)), [names[m] for m in order], fontsize=8)
    ax.set_ylabel("PR-AUC Test"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("PR-AUC Test et IC95 (bootstrap de blocs spatiaux)", fontsize=10)
    fig1.tight_layout(); p1 = FIG / "report_pr_auc_ci.png"; fig1.savefig(p1, dpi=200); plt.close(fig1)

    lab = {"orig_longer": "Réseau initial\n+ plus d'époques", "v2_p16_bce": "v2 + augmentation\n16×16",
           "v2_p32_bce": "v2 + augmentation\n32×32 (retenu)", "v2_p32_focal": "v2 32×32\nfocal loss"}
    seq = ["orig_longer", "v2_p16_bce", "v2_p32_bce", "v2_p32_focal"]
    v = val_sel.set_index("name").loc[seq, "val_pr_auc"].tolist()
    fig2, ax = plt.subplots(figsize=(7.5, 2.9))
    ax.bar(0, 0.1665, color="#B8C4BA"); ax.text(0, 0.1665 + 0.006, "0.166", ha="center", fontsize=8)
    for i, (s, x) in enumerate(zip(seq, v), start=1):
        ax.bar(i, x, color="#2F6B3A" if s == "v2_p32_bce" else "#8FB59A"); ax.text(i, x + 0.006, f"{x:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(5), ["Run initial\n(6 époques)"] + [lab[s] for s in seq], fontsize=7.8)
    ax.set_ylabel("PR-AUC Validation"); ax.set_ylim(0, 0.34); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Effet de chaque amélioration (graine de base, Validation)", fontsize=10)
    fig2.tight_layout(); p2 = FIG / "report_ablation.png"; fig2.savefig(p2, dpi=200); plt.close(fig2)
    return p1, p2


def on_page(c, doc):
    c.saveState(); c.setFont("Arial", 7.5); c.setFillColor(colors.HexColor("#777777"))
    c.drawString(2 * cm, 1.2 * cm, "OliveSmart OLV-6 — Détection d'oliviers infectés par imagerie satellite")
    c.drawRightString(19 * cm, 1.2 * cm, f"Page {doc.page}")
    c.restoreState()


def main():
    prep = json.loads((ORIG / "preparation_summary.json").read_text(encoding="utf-8"))
    comp = pd.read_csv(RES / "model_comparison_test.csv")
    val_sel = pd.read_csv(RES / "validation_selection.csv")
    per_seed = pd.read_csv(RES / "test_per_seed.csv")
    ci = pd.read_csv(RES / "bootstrap_ci_test.csv")
    summ = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    frozen = json.loads((RES / "frozen_choices.json").read_text(encoding="utf-8"))
    comp_hyp = json.loads((ORIG / "complementary_experiments.json").read_text(encoding="utf-8"))
    p_ci, p_abl = make_charts(val_sel, comp, ci)

    ens_f1 = comp[comp.model == "siamese_v2_ensemble@f1_opt"].iloc[0]
    ens_rc = comp[comp.model == "siamese_v2_ensemble@recall_60"].iloc[0]
    o = {m: comp[comp.model == f"original_{m}"].iloc[0] for m in ("siamese_cnn", "tcn", "cusum", "threshold", "isolation_forest")}
    d = summ["delta_vs_original"]
    sc = prep["split_counts"]

    story = [P("Détection d'oliviers infectés par imagerie satellite", "title"),
             P("Projet OLV-6 — comparaison de cinq approches et amélioration du Siamese CNN · novembre 2026", "sub")]

    story.append(boxed(
        "<b>Résumé.</b> Cinq pipelines (seuil d'intensité, CUSUM, Isolation Forest, Siamese CNN, TCN) ont été comparés sur "
        f"{prep['usable_points']:_} oliviers OQDS avec une partition spatiale stricte. ".replace("_", " ") + f"Le Siamese CNN bi-temporel est de loin le meilleur. "
        "Une seconde campagne d'entraînement corrigeant ses faiblesses (sous-entraînement, contexte trop petit, absence d'augmentation, "
        f"graine unique) porte sa PR-AUC Test de <b>{f(o['siamese_cnn'].pr_auc)}</b> à <b>{f(ens_f1.pr_auc)}</b> et son ROC-AUC de "
        f"{f(o['siamese_cnn'].roc_auc)} à <b>{f(ens_f1.roc_auc)}</b>. Il détecte désormais {int(ens_f1.tp)} arbres infectés sur 237 "
        f"(contre {int(o['siamese_cnn'].tp)}). Le gain est statistiquement robuste : +{f(d['delta_pr_auc_mean'])} de PR-AUC, IC95 "
        f"[+{f(d['lo'])} ; +{f(d['hi'])}]."))

    story += [P("1. Objectif et données", "h1"),
              P("L'objectif est de repérer automatiquement, à partir d'images satellites prises à plusieurs dates, les oliviers probablement "
                "infectés, afin d'alimenter une application JEE de suivi des vergers. Quatre sources ont été auditées :"),
              *bullets([
                  "<b>OQDS</b> — trois images WorldView-2 co-enregistrées (2019, 2020, 2021, 0,5 m/pixel) et la position de chaque arbre. "
                  "Un arbre est étiqueté « infecté » s'il est présent en 2019 puis absent ensuite (retrait après infection probable). "
                  "C'est la seule source compatible avec la tâche ; elle sert au benchmark principal.",
                  "<b>NEON Tree Crowns</b> — autres essences, aucune étiquette de maladie, aucun individu suivi sur deux années : non utilisable pour le benchmark.",
                  f"<b>Hyperspectral (stress de l'olivier)</b> — {comp_hyp['Hyperspectral']['observations']} observations dont "
                  f"{comp_hyp['Hyperspectral']['positives']} positives ; expérience exploratoire séparée (PR-AUC hors-pli "
                  f"{f(comp_hyp['Hyperspectral']['pr_auc_cv'])}), non comparable.",
                  f"<b>OliveTreeCrownsDb</b> — {comp_hyp['CrownsDb']['boxes']:,} boîtes de couronnes sans information sanitaire ni temporelle ; "
                  "utile pour la détection de couronnes, pas pour la classification.".replace(",", " "),
              ]),
              P("<b>Point important :</b> les fichiers OQDS nommés « NDVI » sont en réalité des rendus RGBA 8 bits sans calibration. "
                "Les variables qui en sont tirées sont donc traitées comme des <i>intensités</i>, pas comme un NDVI physique.")]

    story += [P("2. Protocole expérimental", "h1"),
              *bullets([
                  f"<b>Nettoyage :</b> {prep['original_points']:,} points → {prep['duplicate_coordinate_rows_removed']} doublons et "
                  f"{sum(prep['invalid_counts'].values()):,} observations invalides (patches majoritairement noirs, couverture insuffisante) retirés → "
                  f"<b>{prep['usable_points']:,} arbres</b>.".replace(",", " "),
                  "<b>Partition spatiale :</b> le territoire est découpé en blocs de 250 m ; chaque bloc entier va en Train, Validation ou Test. "
                  "Aucun arbre ni bloc n'est partagé, ce qui évite qu'un modèle « reconnaisse » un voisinage déjà vu.",
                  "<b>Anti-fuite :</b> normalisations et scalers ajustés sur Train uniquement ; hyperparamètres et seuils choisis sur Validation ; "
                  "Test calculé une seule fois après gel des choix.",
                  "<b>Métrique principale : PR-AUC.</b> Avec ~2,3 % d'arbres infectés, un modèle qui répond toujours « sain » obtient 97,7 % "
                  "d'accuracy : l'accuracy n'est donc pas informative. La PR-AUC d'un classement aléatoire vaut 0,023.",
              ])]
    story.append(table([
        ["Partition", "Arbres sains", "Arbres infectés", "Prévalence", "Blocs spatiaux"],
        *[[s.capitalize(), f"{sc[s]['0']:,}".replace(",", " "), sc[s]["1"], f"{100*sc[s]['1']/(sc[s]['0']+sc[s]['1']):.1f} %".replace(".", ","),
           prep["split_blocks"][s]] for s in ("train", "val", "test")]],
        [3.4 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm]))
    story.append(Spacer(1, 4))

    story += [P("3. Benchmark initial des cinq approches", "h1"),
              P("Chaque méthode a été réglée sur Validation (3 à 6 configurations) puis évaluée une seule fois sur Test.")]
    names = {"siamese_cnn": "Siamese CNN (patches RGB 2019/2021)", "tcn": "TCN (séries d'intensité 3 dates)",
             "cusum": "CUSUM (rupture d'intensité)", "threshold": "Seuil de baisse d'intensité", "isolation_forest": "Isolation Forest"}
    story.append(KeepTogether(table([["Modèle", "PR-AUC", "ROC-AUC", "F1", "Détectés / 237", "Fausses alertes"],
                        *[[names[m], f(o[m].pr_auc), f(o[m].roc_auc), f(o[m].f1), int(o[m].tp), f"{int(o[m].fp):,}".replace(",", " ")] for m in names]],
                       [6.2 * cm, 2 * cm, 2 * cm, 1.8 * cm, 2.6 * cm, 2.4 * cm], highlight_rows=(1,))))
    story += [Spacer(1, 5),
              P("Le Siamese CNN, qui compare directement les patches d'image des deux dates, domine nettement. Les méthodes fondées sur une "
                "seule intensité moyenne (seuil, CUSUM) sont à peine meilleures que le hasard : un changement de luminosité n'est pas spécifique "
                "au retrait d'un arbre. L'Isolation Forest est inadaptée (une anomalie radiométrique n'est pas une infection) et le TCN dispose "
                "de trop peu de dates pour exploiter une vraie dynamique temporelle."),
              P("<b>Diagnostic du Siamese initial.</b> Quatre faiblesses ont été identifiées : (1) seulement 6 époques alors que la PR-AUC "
                "Validation progressait encore (0,121 → 0,166) ; (2) un contexte de 16×16 pixels (8 m) ; (3) aucune augmentation ni "
                "normalisation par lots ; (4) un résultat issu d'une graine unique, sans intervalle de confiance.")]

    story += [P("4. Campagne d'amélioration du Siamese CNN", "h1"),
              P("La campagne réutilise exactement la même partition et le même ensemble Test. Elle est isolée dans "
                "<font face='Arial-Bold'>outputs/runs/improved_v1/</font> et n'écrase aucun résultat initial."),
              *bullets([
                  "<b>Entraînement :</b> jusqu'à 40 époques, arrêt anticipé (patience 7) sur la PR-AUC Validation, réduction automatique du taux d'apprentissage, AdamW.",
                  "<b>Architecture v2 :</b> encodeur à trois blocs convolutifs avec BatchNorm, partagé entre les deux dates ; fusion |z<sub>a</sub>−z<sub>b</sub>|, z<sub>a</sub>·z<sub>b</sub>, z<sub>b</sub>−z<sub>a</sub>.",
                  "<b>Augmentation :</b> rotations de 90° et symétries appliquées identiquement aux deux dates, variation radiométrique indépendante par date (robustesse aux différences d'acquisition).",
                  "<b>Contexte :</b> patches 16×16 (identiques au run initial, vérifié pixel à pixel) contre 32×32 (16 m).",
                  "<b>Déséquilibre :</b> BCE pondérée (poids √(sains/infectés)) contre focal loss.",
                  "<b>Variance :</b> la configuration retenue est ré-entraînée avec 5 graines ; le modèle final moyenne leurs probabilités. IC95 par bootstrap de blocs spatiaux (1 000 tirages).",
              ]),
              fig(p_abl, 15.5, "Figure 1 — PR-AUC Validation de chaque configuration. Le simple allongement de l'entraînement confirme le "
                  "sous-entraînement (0,166 → 0,228) ; l'architecture v2 avec augmentation et le contexte 32×32 apportent le reste.")]
    story.append(table([["Configuration", "Patch", "Perte", "PR-AUC Val.", "Meilleure époque", "Temps (s)"],
                        *[[r["name"], f"{r['patch']}×{r['patch']}", r["loss"].replace("bce_sqrtpw", "BCE √pondérée").replace("bce_pw", "BCE pondérée").replace("focal", "focal loss"),
                           f(r["val_pr_auc"]), int(r["best_epoch"]), int(r["train_time_s"])] for _, r in val_sel.iterrows()]],
                       [3.6 * cm, 1.8 * cm, 3.6 * cm, 2.4 * cm, 3 * cm, 2.4 * cm], highlight_rows=(1,)))
    story += [Spacer(1, 4), P("La configuration <b>v2_p32_bce</b> est retenue sur Validation. La focal loss donne un score quasi identique (0,292) : "
                              "le choix de la perte n'est pas décisif, contrairement au contexte spatial et à l'architecture.")]

    story += [P("5. Résultats finaux sur Test", "h1")]
    story.append(table([["Modèle", "PR-AUC", "ROC-AUC", "F1", "Précision", "Rappel", "Détectés", "Fausses alertes"],
                        ["Siamese v2 — seuil F1", f(ens_f1.pr_auc), f(ens_f1.roc_auc), f(ens_f1.f1), f(ens_f1.precision), f(ens_f1.recall), f"{int(ens_f1.tp)}/237", int(ens_f1.fp)],
                        ["Siamese v2 — seuil surveillance", f(ens_rc.pr_auc), f(ens_rc.roc_auc), f(ens_rc.f1), f(ens_rc.precision), f(ens_rc.recall), f"{int(ens_rc.tp)}/237", int(ens_rc.fp)],
                        ["Siamese initial", f(o["siamese_cnn"].pr_auc), f(o["siamese_cnn"].roc_auc), f(o["siamese_cnn"].f1), f(o["siamese_cnn"].precision), f(o["siamese_cnn"].recall), f"{int(o['siamese_cnn'].tp)}/237", int(o["siamese_cnn"].fp)]],
                       [4.5 * cm, 1.6 * cm, 1.9 * cm, 1.3 * cm, 1.8 * cm, 1.5 * cm, 1.8 * cm, 2.6 * cm], highlight_rows=(1,)))
    story += [Spacer(1, 4),
              P(f"Les deux seuils ont été fixés sur Validation avant le calcul Test : le seuil F1 optimal ({f(frozen['ensemble_thresholds']['f1_opt'])}) "
                f"et un seuil « surveillance » visant au moins 60 % de rappel ({f(frozen['ensemble_thresholds']['recall_60'])}). "
                f"Au seuil F1, le modèle retrouve <b>{100*ens_f1.recall:.0f} %</b> des arbres infectés avec environ une alerte sur trois correcte ; "
                f"au seuil surveillance, il en retrouve {100*ens_rc.recall:.0f} % au prix de {int(ens_rc.fp)} fausses alertes sur 9 968 arbres sains."),
              P(f"<b>Stabilité :</b> sur les 5 graines, PR-AUC Test {f(per_seed.pr_auc.mean())} ± {f(per_seed.pr_auc.std())}, "
                f"ROC-AUC {f(per_seed.roc_auc.mean())} ± {f(per_seed.roc_auc.std())}. Chaque graine individuelle dépasse largement le Siamese initial ; "
                "l'ensemble est légèrement meilleur que la moyenne des graines."),
              fig(p_ci, 15.5, "Figure 2 — PR-AUC Test avec intervalles de confiance à 95 % (bootstrap de blocs spatiaux). La ligne pointillée est le niveau du hasard."),
              fig(FIG / "pr_curves_original_vs_improved.png", 10.5, "Figure 3 — Courbes précision–rappel sur Test : à rappel égal, le Siamese v2 offre une précision bien supérieure.")]

    story += [P("6. Limites", "h1"), *bullets([
        "<b>Écart Validation/Test :</b> la PR-AUC Test (0,49) dépasse la PR-AUC Validation (0,29), comme dans le run initial (0,22 contre 0,17). "
        "Les cas infectés sont regroupés dans peu de blocs, d'où des intervalles très larges (PR-AUC Test entre 0,11 et 0,73). "
        "La performance attendue sur une nouvelle zone se situe plutôt entre ces deux valeurs.",
        "<b>Réutilisation du Test :</b> l'ensemble Test avait déjà été consulté pour le run initial. Tous les choix du run amélioré ont été faits sur "
        "Validation seule, mais une confirmation sur une exploitation externe reste indispensable.",
        "<b>Cible indirecte :</b> « infecté » signifie « arbre retiré après 2019 », pas un diagnostic de laboratoire individuel. Le modèle détecte d'abord une disparition de couronne.",
        "<b>Radiométrie :</b> RGB 8 bits non calibré, trois dates seulement, pas de vrai NDVI/proche infrarouge.",
    ])]

    story += [P("7. Recommandation pour l'application JEE", "h1"), *bullets([
        "Déployer le <b>Siamese v2 (ensemble de 5 graines)</b> — <font face='Arial-Bold'>siamese_ensemble_best.pt</font> — dans un service Python "
        "d'inférence séparé, exposé à l'application JEE par une API REST.",
        "Entrée : deux patches RGB <b>32×32</b> géoréférencés (dates t<sub>0</sub> et t<sub>1</sub>) ; normalisation sauvegardée appliquée par le service ; "
        "sortie : score, classe, seuil utilisé et version du modèle.",
        "Proposer deux modes : <b>précision</b> (seuil F1) pour des alertes fiables, <b>surveillance</b> (seuil rappel 60 %) pour un dépistage large suivi d'une vérification terrain.",
        "Conserver la règle de seuil d'intensité uniquement comme indicateur explicable, jamais comme substitut du modèle.",
        "Avant production : test sur une nouvelle région, contrôle de dérive à chaque nouvelle zone, et idéalement rasters NIR/NDVI calibrés et vérité terrain indépendante.",
    ])]

    story += [P("Reproductibilité", "h2"),
              P("Graine 20261106 ; PyTorch CPU ; durée du run amélioré : "
                f"{summ['total_minutes']:.0f} min. Scripts : <font face='Arial-Bold'>prepare_oqds.py</font>, <font face='Arial-Bold'>train_evaluate.py</font> "
                "(benchmark initial), <font face='Arial-Bold'>train_improved.py</font> (campagne d'amélioration), "
                "<font face='Arial-Bold'>build_pdf_report.py</font> (ce rapport). Configurations, seuils figés, historiques d'entraînement et prédictions "
                "par arbre sont enregistrés dans <font face='Arial-Bold'>outputs/</font>.")]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.8 * cm,
                            title="Rapport OLV-6 — Détection d'oliviers infectés", author="Projet OLV-6")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(OUT)


if __name__ == "__main__":
    main()
