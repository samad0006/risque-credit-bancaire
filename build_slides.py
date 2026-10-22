"""Génère reports/slides_soutenance.pdf (11 slides, noir et blanc) à partir de results.json."""
import json, os
from PIL import Image as PI
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(ROOT, "reports", "figures")
R = json.load(open(os.path.join(ROOT, "reports", "results.json"), encoding="utf-8"))
FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", FD + "DejaVuSans.ttf")); pdfmetrics.registerFont(TTFont("DV-B", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV", boldItalic="DV-B")
st = {"t": ParagraphStyle("t", fontName="DV-B", fontSize=22, leading=27, spaceAfter=10),
      "b": ParagraphStyle("b", fontName="DV", fontSize=15, leading=21, leftIndent=14, bulletIndent=2, spaceAfter=5),
      "m": ParagraphStyle("m", fontName="DV-B", fontSize=14, leading=19, spaceBefore=8),
      "c": ParagraphStyle("c", fontName="DV", fontSize=10.5, leading=13), "ch": ParagraphStyle("ch", fontName="DV-B", fontSize=10.5, leading=13),
      "big": ParagraphStyle("big", fontName="DV-B", fontSize=30, leading=38, spaceAfter=14)}
fr = lambda x, d=0: f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
pct = lambda x, d=0: fr(x * 100, d) + " %"
mm = lambda x, d=1: fr(x / 1e6, d) + " M$"
T = R["test"]; M = T["modele"]; RU = T["regle_notes_DG"]; C = {x["scenario"]: x for x in R["roi"]}; cen = C["Central"]
def T_(t): return Paragraph(t, st["t"])
def Bl(t): return Paragraph(t, st["b"], bulletText="•")
def img(n, h):
    w, hh = PI.open(os.path.join(FIG, n)).size
    return Image(os.path.join(FIG, n), width=h * cm * w / hh, height=h * cm)
def tb(rows, widths):
    d = [[Paragraph(str(c), st["ch"] if i == 0 else st["c"]) for c in r] for i, r in enumerate(rows)]
    t = Table(d, colWidths=[w * cm for w in widths])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.black), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDDDDD")),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t
S = []
S += [Spacer(1, 2.2 * cm), Paragraph("Prédire le risque de défaut d'un prêt personnel", st["big"]),
      Paragraph("De la donnée à la décision · Projet final S7MOI1 · ECAM-EPMI · 2AE", st["m"]),
      Paragraph("Abdessamad EL IDRISSI · Octobre 2026", st["b"]), PageBreak()]
S += [T_("1. Le problème"), Bl("Banque de détail : certains emprunteurs ne remboursent pas, refuser à tort un bon client fait aussi perdre de l'argent."),
      Bl("<b>Décision</b> : pour chaque demande, accorder ou refuser (aide à la décision de l'analyste crédit)."),
      Bl("<b>Situation actuelle (hypothèse)</b> : refus des notes internes D à G."),
      Bl("<b>Critère de succès</b> : détecter au moins 80 % des défauts, avec au plus 1 refus à tort pour 1 défaut évité, et réduire le coût d'erreur d'au moins 30 %."),
      PageBreak()]
S += [T_("2. Les données"), Bl(f"Credit Risk Dataset : <b>{fr(R['raw']['n'])} demandes</b>, 11 variables, cible binaire (défaut = {pct(R['default_rate'],0)})."),
      Bl(f"Qualité : valeurs manquantes (ancienneté, taux) imputées dans le pipeline ; {R['clean']['doublons']} doublons et {R['clean']['valeurs_impossibles']} valeurs impossibles (âge 144 ans) supprimés."),
      Bl("Audit de fuite : aucune variable de l'après-octroi. <b>Note interne et taux</b> : connus à la décision mais circulaires, donc testés par ablation."),
      Bl("Split stratifié 80/20 fait <b>avant</b> toute préparation ; le test n'est utilisé qu'une fois."), PageBreak()]
S += [T_("3. Ce que montrent les données"), img("fig2_eda.png", 10.5), PageBreak()]
S += [T_("4. La démarche"), img("fig1_pipeline.png", 3.8),
      Bl("Classification binaire supervisée (régression, clustering et prévision écartés)."),
      Bl("Pipeline scikit-learn : imputation, standardisation, one-hot, tout ajusté sur le train."),
      Bl("6 modèles en validation croisée 5 plis, réglage d'hyperparamètres, baseline obligatoire."), PageBreak()]
rows = [["Modèle", "PR-AUC train", "PR-AUC CV (± σ)", "Lecture"]]
rd = ["Plancher", "Stable mais limité", "Sur-apprentissage net", "Gain avec max_depth", "Écart train/CV marqué", "<b>Retenu : meilleur et stable</b>"]
for r, l in zip(R["models"], rd): rows.append([r["modele"], fr(r["ap_train"], 2), f"{fr(r['ap_cv'],3)} ± {fr(r['ap_cv_std'],3)}", l])
S += [T_("5. Modèles comparés (validation croisée)"), tb(rows, [8, 3.6, 4.6, 7.5]), Spacer(1, 8),
      Paragraph("L'arbre libre mémorise le train (1,00) mais généralise mal. Le gradient boosting bat la baseline et les autres modèles.", st["m"]), PageBreak()]
S += [T_("6. Métrique et seuil choisis par le coût"),
      Bl("Métrique principale : <b>Recall</b> sur les défauts ; garde-fou Precision ≥ 50 %. Accuracy écartée (tout accorder donne déjà 78 %)."),
      Bl("Coûts (hypothèses) : défaut non détecté = 60 % du montant ; bon client refusé = 12 % du montant."),
      img("fig5_threshold.png", 6.2), Paragraph(f"Seuil retenu : <b>{fr(R['thresholds']['t_final'],2)}</b> (et non 0,5), choisi sur le train.", st["m"]), PageBreak()]
rows = [["Stratégie (test)", "Recall", "Precision", "Refus", "Coût d'erreur"],
        [f"<b>Modèle, seuil {fr(R['thresholds']['t_final'],2)}</b>", pct(M["recall"]), pct(M["precision"]), pct(M["refus"]), mm(T["cout"]["modele"], 2)],
        ["Règle actuelle (notes D-G)", pct(RU["recall"]), pct(RU["precision"]), pct(RU["refus"]), mm(T["cout"]["regle"], 2)],
        ["Tout accorder", "0 %", "—", "0 %", mm(T["cout"]["accorder_tout"], 2)]]
S += [T_("7. Résultats sur des données jamais vues"), tb(rows, [8.5, 3.4, 3.4, 3.2, 4.5]), Spacer(1, 6), img("fig6_test.png", 7.4),
      Paragraph(f"« Sur 100 défauts, le modèle en anticipe {fr(M['recall']*100,0)}, contre {fr(RU['recall']*100,0)} pour la règle actuelle. »", st["m"]), PageBreak()]
rows = [["Scénario", "Part du gain réalisée", "Coût du projet", "Gain annuel", "ROI an 1"]]
for k in ["Pessimiste", "Central", "Optimiste", "Dérive (scores bruités)"]:
    x = C[k]; rows.append([k, pct(x["part_realisee"]), fr(x["cout_projet"] / 1e3) + " k$", mm(x["gain"]), fr(x["roi"]) + " %"])
S += [T_("8. La valeur : ROI et scénarios"), tb(rows, [6.5, 4.5, 4, 4, 4]), Spacer(1, 8),
      Bl("Hypothèses explicites : 32 000 demandes/an, LGD 60 %, marge 12 %, gain mesuré décoté de 30 % dans le central."),
      Bl(f"Point mort : il suffit que {pct(R['break_even_part_realisee'],1)} du gain mesuré se réalise. <b>Ce ROI n'est pas une prévision</b> : la LGD et la qualité de la référence sont à valider."), PageBreak()]
S += [T_("9. Risques et limites"),
      Bl("<b>Biais de sélection</b> : l'historique ne contient que des prêts accordés (pilote en parallèle)."),
      Bl("<b>Circularité</b> de la note interne ; sans elle PR-AUC " + fr(R["ablation"]["Sans note interne ni taux (grade, loan_int_rate)"]["ap"], 2) + " (reste très au-dessus de la baseline 0,22)."),
      Bl("<b>Équité et réglementation</b> : retirer l'âge, auditer les refus par groupe (logement), humain dans la boucle (RGPD art. 22)."),
      Bl("<b>Dérive</b> : suivi mensuel du Recall, réentraînement trimestriel. <b>Jeu très propre</b>, possiblement synthétique : performances probablement optimistes."), PageBreak()]
S += [T_("10. Recommandation"), Bl("Remplacer la règle seule par un <b>score de risque en aide à la décision</b>."),
      Bl("Lancer un <b>pilote de 3 mois en parallèle</b> de la règle actuelle pour mesurer le coût réel (LGD, marge) avant déploiement."),
      Bl("Valider les hypothèses avec la direction des risques, tester un modèle sans l'âge, calibrer les probabilités.")]
doc = SimpleDocTemplate(os.path.join(ROOT, "reports", "slides_soutenance.pdf"), pagesize=(28 * cm, 15.75 * cm),
                        leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.1 * cm, bottomMargin=1 * cm, title="Soutenance S7MOI1")
doc.build(S); print("slides OK")
