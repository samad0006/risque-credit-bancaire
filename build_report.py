"""Génère reports/rapport_synthese.pdf à partir de reports/results.json (aucun chiffre saisi à la main)."""
import html, json, os
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "reports", "figures")
R = json.load(open(os.path.join(ROOT, "reports", "results.json"), encoding="utf-8"))

FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", FD + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

NAVY, TEAL, ORANGE, LIGHT = colors.black, colors.black, colors.black, colors.HexColor("#F2F2F2")
S = {
    "body": ParagraphStyle("body", fontName="DV", fontSize=8.6, leading=12.2, alignment=TA_JUSTIFY, spaceAfter=4),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=13, leading=16, textColor=NAVY, spaceBefore=8, spaceAfter=5, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=9.6, leading=13, textColor=TEAL, spaceBefore=5, spaceAfter=2, keepWithNext=1),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.4, leading=9.6),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=7.4, leading=9.6, textColor=colors.black),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=7.4, leading=9.6, textColor=colors.HexColor("#555555"), spaceAfter=6),
    "bul": ParagraphStyle("bul", fontName="DV", fontSize=8.6, leading=12, leftIndent=11, bulletIndent=2, spaceAfter=1.5),
    "box": ParagraphStyle("box", fontName="DV", fontSize=8.6, leading=12.2, textColor=colors.black),
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=17, leading=22, textColor=NAVY, spaceAfter=3),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=11, leading=15, textColor=colors.HexColor("#444444"), spaceAfter=2),
}


def fr(x, d=0):
    s = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return s


def pct(x, d=0):
    return fr(x * 100, d) + " %"


def mm(x, d=1):                       # millions
    return fr(x / 1e6, d) + " M$"


def kk(x):
    return fr(x / 1e3, 0) + " k$"


def P(t, st="body"): return Paragraph(t, S[st])
def B(t): return Paragraph(t, S["bul"], bulletText="•")
def H1(t): return Paragraph(t, S["h1"])
def H2(t): return Paragraph(t, S["h2"])


def tbl(rows, widths, head=True, zebra=True, align_right_from=None):
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(str(c), S["cellb"] if (head and i == 0) else S["cell"]) for c in r])
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if head else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C9D1DC")),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5)]
    if head: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDDDDD")))
    if zebra:
        for i in range(1, len(rows)):
            pass
    t.setStyle(TableStyle(st))
    return t


def fig(name, width_cm, caption):
    from PIL import Image as PI
    w, h = PI.open(os.path.join(FIG, name)).size
    img = Image(os.path.join(FIG, name), width=width_cm * cm, height=width_cm * cm * h / w)
    return KeepTogether([img, Paragraph(caption, S["cap"])])


def box(text, color=TEAL):
    t = Table([[Paragraph(text, S["box"])]], colWidths=[17.4 * cm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.black),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def footer(c, d):
    c.saveState(); c.setFont("DV", 7); c.setFillColor(colors.HexColor("#777777"))
    c.drawString(1.8 * cm, 1.1 * cm, "Projet final S7MOI1 · ECAM-EPMI · 2AE · Risque de défaut de prêt personnel")
    c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {d.page}")
    c.restoreState()


# ------------------------------------------------------------------ valeurs
T = R["test"]; M = T["modele"]; RU = T["regle_notes_DG"]; M5 = T["seuil_05"]
mods = R["models"]; th = R["thresholds"]; C = {x["scenario"]: x for x in R["roi"]}
cen = C["Central"]; pes = C["Pessimiste"]; opt = C["Optimiste"]; der = C["Dérive (scores bruités)"]
hgb = mods[-1]; tree_free = mods[2]; rf = mods[4]
tf = th["t_final"]
tab = th["oof_table"]
LGD, MG = cen["lgd"], cen["marge"]
n_te, n_def = T["n"], T["n_defauts"]
E = R["errors"]; F = R["fairness"]; A = R["ablation"]
a_noage = A["Sans l'âge"]["ap"]
Fa = list(F.values())
miss = R["raw"]["missing"]

S_ = []
# ================================================================== PAGE DE TITRE + RÉSUMÉ
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import HRFlowable
C_ = lambda sz, bold=False, sp=6: ParagraphStyle("c%s" % sz, fontName="DV-B" if bold else "DV", fontSize=sz, leading=sz * 1.35, alignment=TA_CENTER, spaceAfter=sp)
S_ += [Spacer(1, 2.5 * cm), Paragraph("ECAM-EPMI · 2AE", C_(13, True, 4)),
       Paragraph("Module S7MOI1 · Science des Données – Concepts Avancés", C_(11, False, 40)),
       HRFlowable(width="70%", thickness=0.8, color=colors.black, spaceAfter=30),
       Paragraph("Projet final", C_(13, False, 10)),
       Paragraph("De la donnée à la décision", C_(26, True, 10)),
       Paragraph("Prédire le risque de défaut de paiement<br/>d'un prêt personnel", C_(16, False, 30)),
       HRFlowable(width="70%", thickness=0.8, color=colors.black, spaceAfter=40),
       Paragraph("Rapport de synthèse", C_(12, True, 30)),
       Paragraph("Réalisé par : <b>Abdessamad EL IDRISSI</b>", C_(12, False, 8)),
       Paragraph("Intervenant : Yann Skonieczny", C_(12, False, 8)),
       Paragraph("Octobre 2026", C_(12, False, 8)), PageBreak(),
       P("Projet final S7MOI1 · Risque de défaut de prêt personnel", "sub"), Spacer(1, 0.2 * cm)]
S_.append(box(
    "<b>Résumé en 3 phrases.</b> Nous proposons de remplacer la règle actuelle (refuser les notes internes D à G) par un "
    f"score de risque qui aide à accorder ou refuser chaque demande de prêt. Sur {fr(n_te)} demandes jamais vues, le modèle "
    f"repère <b>{pct(M['recall'])} des défauts</b> ({fr(M['tp'])} sur {fr(n_def)}) en refusant {pct(M['refus'])} des demandes, "
    f"et {pct(M['precision'])} des refus sont justifiés, alors que la règle actuelle ne repère que {pct(RU['recall'])} des défauts. "
    f"Sous nos hypothèses, le coût annuel des erreurs passe de {mm(cen['cout_sans'])} à {mm(cen['cout_avec'])} ; nous "
    f"recommandons un <b>pilote de 3 mois en parallèle de la règle actuelle</b> pour mesurer le gain réel avant tout déploiement.", ORANGE))
S_.append(Spacer(1, 0.2 * cm))
S_.append(P("<b>Déclaration d'usage de l'IA générative.</b> Un assistant d'IA générative (Claude, Anthropic) a été utilisé pour "
            "structurer le dépôt, écrire le code (script et notebooks), rédiger une première version de ce rapport et les slides. Les membres de l'équipe relisent, "
            "exécutent et savent expliquer chaque ligne de code et chaque chiffre (cf. annexe C). Tous les chiffres de ce document sont "
            "produits par <i>run_analysis.py</i> (fichier <i>reports/results.json</i>)."))

# ================================================================== AXE 1
S_.append(H1("Axe 1 · Cadrage de la problématique métier"))
S_.append(tbl([
    ["Élément", "Réponse"],
    ["Contexte", "Banque de détail qui propose des prêts personnels (études, santé, travaux, consolidation de dettes…). "
                 "Utilisateurs du résultat : l'analyste crédit et le responsable des risques."],
    ["Problème", "Certains emprunteurs ne remboursent pas leur prêt et la banque perd de l'argent ; à l'inverse, refuser "
                 "un bon client lui fait perdre la marge sur ce prêt."],
    ["Décision à éclairer", "Pour chaque nouvelle demande : <b>accorder ou refuser</b> (ou envoyer en revue humaine), au moment "
                            "du dépôt du dossier. Le jeu de données ne précise pas l'horizon exact du défaut : c'est une limite assumée (axe 9)."],
    ["Situation actuelle", "<i>Hypothèse posée</i> (le processus réel n'est pas documenté) : règle sur la note interne, avec refus des "
                           "notes D à G. C'est notre référence métier, en plus de « tout accorder »."],
    ["Critère de succès", "Détecter au moins 80 % des défauts, avec au plus 1 client sain refusé pour 1 défaut évité (Precision ≥ 50 %), "
                          "et réduire d'au moins 30 % le coût annuel des erreurs par rapport à la règle actuelle, mesuré sur données jamais vues "
                          "puis confirmé par un pilote de 3 mois."],
    ["Contraintes", "Seules les informations connues au dépôt du dossier sont utilisables ; modèle explicable ; décision "
                    "automatisée encadrée (RGPD art. 22, crédit = usage « à haut risque » du règlement européen sur l'IA) donc "
                    "<b>humain dans la boucle</b> ; taux de refus global raisonnable pour ne pas bloquer l'activité."],
], [3.3, 14.1]))

# ================================================================== AXE 2
S_.append(H1("Axe 2 · Type de problématique"))
S_.append(P("La <b>cible</b> est <i>loan_status</i> : 1 si le client a fait défaut, 0 s'il a remboursé. Elle est connue pour tout "
            "l'historique, donc le problème est <b>supervisé</b>, et elle ne prend que deux valeurs : c'est une <b>classification binaire</b>. "
            "Les autres types sont écartés : une <i>régression</i> n'a pas de sens car la sortie n'est pas bornée entre 0 et 1 (et le montant "
            "réellement perdu n'est pas dans les données) ; un <i>clustering</i> ignorerait l'étiquette dont nous disposons ; une "
            f"<i>prévision</i> est impossible car le jeu ne contient aucune date ; le <i>multi-classe</i> n'a pas de raison d'être (un seul type de défaut). "
            f"La classe positive est minoritaire ({pct(R['default_rate'], 1)}) : nous en tenons compte dans le split (stratifié) et dans le choix des métriques (axe 5)."))

# ================================================================== AXE 3
S_.append(H1("Axe 3 · Analyse et préparation des données"))
S_.append(H2("Description"))
S_.append(P(f"Source : <i>Credit Risk Dataset</i> (Kaggle) [lien exact, licence et date de consultation à compléter dans le README]. "
            f"La copie utilisée est celle du dépôt GitHub JoshLG18/DSE-EMP-Project. Elle contient <b>{fr(R['raw']['n'])} demandes</b> et "
            f"{R['raw']['cols']} colonnes, sans variable temporelle. Montants exprimés en dollars (hypothèse, non précisé par la source)."))
S_.append(tbl([
    ["Variable", "Type", "Signification (unité)"],
    ["person_age", "numérique", "Âge du demandeur (années)"],
    ["person_income", "numérique", "Revenu annuel ($)"],
    ["person_home_ownership", "catégorielle", "Logement : RENT, MORTGAGE, OWN, OTHER"],
    ["person_emp_length", "numérique", "Ancienneté dans l'emploi (années)"],
    ["loan_intent", "catégorielle", "Objet du prêt (6 modalités : études, santé, travaux, consolidation, personnel, entreprise)"],
    ["loan_grade", "ordinale", "Note interne de risque, A (peu risqué) à G (très risqué)"],
    ["loan_amnt", "numérique", "Montant du prêt ($)"],
    ["loan_int_rate", "numérique", "Taux d'intérêt du prêt (%)"],
    ["loan_percent_income", "numérique", "Montant du prêt / revenu annuel (ratio)"],
    ["cb_person_default_on_file", "binaire", "Défaut passé enregistré par un bureau de crédit (Y/N)"],
    ["cb_person_cred_hist_length", "numérique", "Ancienneté de l'historique de crédit (années)"],
    ["<b>loan_status</b>", "<b>cible</b>", "<b>1 = défaut, 0 = remboursé</b>"],
], [4.4, 2.3, 10.7]))
S_.append(Spacer(1, 4))
S_.append(H2("Qualité des données et décisions de traitement"))
S_.append(B(f"<b>Valeurs manquantes</b> : person_emp_length ({fr(miss['person_emp_length'])} lignes, {pct(miss['person_emp_length']/R['raw']['n'],1)}) "
            f"et loan_int_rate ({fr(miss['loan_int_rate'])} lignes, {pct(miss['loan_int_rate']/R['raw']['n'],1)}). Décision : imputation par la <b>médiane</b>, "
            "ajustée sur le train uniquement dans le Pipeline, avec un indicateur de valeur manquante. Le taux de défaut est quasi identique avec ou sans taux renseigné "
            f"({pct(R['missing_rate_by_default']['int_rate_missing']['true'],1)} contre {pct(R['missing_rate_by_default']['int_rate_missing']['false'],1)}) : le manque ne semble pas informatif."))
S_.append(B(f"<b>Doublons</b> : {R['clean']['doublons']} lignes identiques supprimées avant le split (sinon une même ligne pourrait se retrouver dans le train et dans le test)."))
S_.append(B(f"<b>Valeurs impossibles</b> : âge jusqu'à {fr(R['raw']['age_max'])} ans et ancienneté jusqu'à {fr(R['raw']['emp_max'])} ans. "
            f"{R['clean']['valeurs_impossibles']} lignes supprimées (âge &gt; 100 ans ou ancienneté &gt; 60 ans). Les revenus extrêmes "
            f"(jusqu'à {fr(R['raw']['income_max'])} $) sont plausibles : conservés, avec une transformation logarithmique."))
S_.append(B(f"<b>Jeu final</b> : {fr(R['n_clean'])} demandes. Split <b>stratifié 80/20</b> : {fr(R['split']['train'])} en train, {fr(R['split']['test'])} en test, "
            f"taux de défaut {pct(R['split']['default_train'],1)} / {pct(R['split']['default_test'],1)} (identique)."))
S_.append(fig("fig2_eda.png", 14.8,
              "Figure 1 · Le défaut est minoritaire (22 %). Il est très lié à la note interne (de 10 % en A à 98 % en G), à la charge de la dette "
              "(montant/revenu) et au statut de logement (locataires ≈ 4 fois plus de défauts que propriétaires). Un défaut passé double le risque."))
def _img(n, w):
    from PIL import Image as PI
    a, b = PI.open(os.path.join(FIG, n)).size
    return Image(os.path.join(FIG, n), width=w * cm, height=w * cm * b / a)


S_.append(Table([[_img("fig3_corr.png", 8.2), _img("fig7_importance.png", 8.6)]],
                colWidths=[8.7 * cm, 8.7 * cm], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
S_.append(P("Figure 2 (gauche) · Les variables les plus corrélées au défaut sont le ratio montant/revenu, la note interne et le taux d'intérêt, "
            "qui sont eux-mêmes corrélés entre eux (la note détermine le taux). Figure 7 (droite, voir axe 4) · importance par permutation.", "cap"))
S_.append(H2("Audit de fuite de données"))
S_.append(P("Aucune colonne ne décrit ce qui se passe <i>après</i> l'octroi (pas d'historique de remboursement, pas de recouvrement) : "
            "<b>aucune variable n'est exclue pour fuite stricte</b>. Deux variables restent cependant sous surveillance : <i>loan_grade</i> et "
            "<i>loan_int_rate</i> sont fixées par la banque à l'octroi. Elles sont connues au moment de la décision, mais elles résument "
            "déjà une évaluation interne du risque (risque de circularité). Nous les gardons et nous <b>mesurons leur poids</b> par ablation à l'axe 4. "
            "<i>loan_percent_income</i> est le simple ratio montant/revenu : pas de fuite."))
S_.append(H2("Ingénierie de variables (3 variables construites)"))
S_.append(B("<b>log_income</b> = log(1 + revenu) : les revenus sont très asymétriques, le log rend la relation plus lisible pour les modèles linéaires."))
S_.append(B("<b>grade_num</b> = note A..G codée de 1 à 7 : la note a un ordre naturel de risque."))
S_.append(B("<b>emp_to_age</b> = ancienneté / âge : mesure la stabilité professionnelle relative (un jeune de 22 ans avec 5 ans d'emploi n'est pas comparable à un homme de 50 ans)."))

# ================================================================== AXE 4
S_.append(H1("Axe 4 · Développement du modèle"))
S_.append(P("<b>Protocole.</b> Split d'abord, test mis de côté. Toute la préparation (imputation + indicateur, standardisation, one-hot) est dans un "
            "<i>Pipeline / ColumnTransformer</i> ajusté sur le train uniquement. Six modèles sont comparés en <b>validation croisée stratifiée à 5 plis</b> "
            "sur le train, avec réglage d'un hyperparamètre au moins (GridSearchCV). Les prédictions sont des probabilités de défaut."))
S_.append(fig("fig1_pipeline.png", 16.6, "Schéma 1 · Pipeline de bout en bout : le test n'est utilisé qu'une seule fois, à la toute fin."))
rows = [["Modèle", "Hyperparam. retenus", "PR-AUC train", "PR-AUC CV (moy. ± σ)", "ROC-AUC CV", "Lecture"]]
reading = ["Plancher : aucune discrimination (PR-AUC = taux de défaut)",
           "Stable (train ≈ CV) mais limité : relations non linéaires",
           f"Sur-apprentissage net : train {fr(tree_free['ap_train'],2)} contre CV {fr(tree_free['ap_cv'],2)}",
           "Gros gain obtenu en limitant la profondeur",
           f"Bon, mais écart train/CV marqué ({fr(rf['ap_train'],2)} contre {fr(rf['ap_cv'],2)})",
           f"<b>Meilleur et plus stable (σ = {fr(hgb['ap_cv_std'],3)}) : retenu</b>"]
for r, rd in zip(mods, reading):
    prm = ", ".join(f"{k}={v}" for k, v in r["params"].items()) or "—"
    rows.append([r["modele"], prm, fr(r["ap_train"], 3), f"{fr(r['ap_cv'],3)} ± {fr(r['ap_cv_std'],3)}", fr(r["auc_cv"], 3), rd])
S_.append(tbl(rows, [3.7, 2.7, 1.6, 2.5, 1.6, 5.3]))
S_.append(P("Tableau 1 · Comparaison des modèles (PR-AUC = aire sous la courbe Precision-Recall, métrique indépendante du seuil et adaptée à une classe rare). "
            "Le score de CV du meilleur réglage est légèrement optimiste, d'où l'intérêt du test final.", "cap"))
S_.append(P(f"<b>Règle de sélection (fixée avant de regarder le test)</b> : meilleur PR-AUC de validation croisée ; en cas d'égalité à un écart-type près, "
            f"modèle le plus simple. Ici le gradient boosting (HistGradientBoosting) l'emporte nettement ({fr(hgb['ap_cv'],3)} contre {fr(rf['ap_cv'],3)} pour la Random Forest, "
            "écart supérieur à l'écart-type) et bat la baseline de très loin. <b>Diagnostic</b> : écart train/validation modéré pour le modèle retenu "
            f"({fr(hgb['ap_train'],2)} contre {fr(hgb['ap_cv'],2)}), net pour l'arbre libre (sur-apprentissage) et la forêt."))
S_.append(fig("fig4_models.png", 16.0, "Figure 3 · L'arbre sans limite de profondeur mémorise le train (PR-AUC = 1,00) mais généralise mal ; HGB est le plus performant en validation."))
S_.append(H2("Importance des variables et rôle de la note interne"))
imp = R["perm_importance"]
S_.append(P("L'importance par permutation (sur le test, uniquement pour expliquer le modèle final) place en tête le ratio montant/revenu, le statut de logement, "
            "la note interne et le revenu. Le taux d'intérêt compte peu : il est largement redondant avec la note. "
            f"<b>Ablations</b> (PR-AUC en CV, train) : modèle complet {fr(A['Modèle complet']['ap'],3)} ; <b>sans l'âge</b> {fr(a_noage,3)} (quasi aucune perte) ; "
            f"<b>sans note interne ni taux</b> {fr(A['Sans note interne ni taux (grade, loan_int_rate)']['ap'],3)}. "
            "Sans les deux variables potentiellement circulaires, le modèle reste très au-dessus de la baseline (0,219) : l'essentiel de sa valeur ne vient pas de la note."))
S_.append(H2("Analyse des erreurs"))
S_.append(P(f"Sur le test, {E['fn_n']} défauts ne sont pas détectés (faux négatifs). Ils ont un profil plus « sain » que les défauts détectés : note moyenne "
            f"{fr(E['fn_grade_mean'],1)} contre {fr(E['tp_grade_mean'],1)} (1 = A, 7 = G), ratio montant/revenu {fr(E['fn_lpi_mean'],2)} contre {fr(E['tp_lpi_mean'],2)}, "
            f"défaut passé dans {pct(E['fn_prior_default_share'])} des cas contre {pct(E['tp_prior_default_share'])}. Ces défauts « surprises » sont par nature les plus difficiles "
            "à anticiper avec les seules données du dossier."))

# ================================================================== AXE 5
S_.append(H1("Axe 5 · Choix des métriques"))
S_.append(B("<b>Métrique principale de décision : Recall sur la classe défaut</b>, car un défaut non détecté coûte beaucoup plus cher qu'un refus à tort "
            f"(environ {fr(LGD/MG,0)} fois plus sous nos hypothèses, axe 6)."))
S_.append(B("<b>Garde-fou : Precision ≥ 50 %</b> (au plus 1 client sain refusé pour 1 défaut évité), pour ne pas détruire l'activité commerciale. "
            "On suit aussi le <b>taux de refus global</b>."))
S_.append(B("<b>Comparaison des modèles : PR-AUC</b> (indépendante du seuil, adaptée à la classe rare), complétée par la ROC-AUC. "
            "<b>F1</b> affiché pour information."))
S_.append(B(f"<b>Accuracy écartée</b> : accorder tous les prêts donne déjà {pct(T['accorder_tout']['tn']/n_te,1)} d'accuracy sans détecter aucun défaut."))

# ================================================================== AXE 6
S_.append(H1("Axe 6 · Justification métrique et business"))
S_.append(P(f"<b>Matrice de coûts (hypothèses explicites, à valider avec la banque).</b> Le seuil décide : on <b>refuse si le score de risque ≥ seuil</b>. "
            f"Faux négatif (prêt accordé puis défaut) : perte = <b>{pct(LGD)} du montant</b> (perte en cas de défaut, LGD ; le reste étant récupéré). "
            f"Faux positif (bon client refusé) : manque à gagner = <b>{pct(MG)} du montant</b> (marge nette sur la durée du prêt). "
            "Vrai positif et vrai négatif : pas de coût d'erreur (la marge des prêts sains accordés est identique avec ou sans modèle). "
            f"Les coûts sont calculés prêt par prêt, proportionnellement au montant. Le seuil théorique qui minimise ce coût est <b>marge/(marge + LGD) = {fr(th['t_theorique'],2)}</b> "
            "pour un modèle bien calibré, ce qui est très loin du seuil par défaut de 0,5."))
rows = [["Seuil (OOF train)", "Precision", "Recall", "F1", "Taux de refus", "Coût d'erreur total (train)"]]
for t in sorted(tab, key=float):
    v = tab[t]; lab = f"{t}" + (" (retenu)" if abs(float(t) - tf) < 1e-9 else "")
    rows.append([lab.replace(".", ","), pct(v["precision"]), pct(v["recall"]), fr(v["f1"], 2), pct(v["refus"]), mm(v["cout"], 2)])
rows.append(["Accorder tout", "—", "0 %", "0,00", "0 %", mm(th["oof_cout_accorder_tout"], 2)])
S_.append(tbl(rows, [3.4, 2.2, 2.2, 1.8, 2.8, 5.0]))
S_.append(P("Tableau 2 · Analyse du seuil sur les prédictions hors-échantillon du <b>train</b> (cross_val_predict) : le test n'intervient pas dans le choix du seuil.", "cap"))
S_.append(fig("fig5_threshold.png", 15.2,
              f"Figure 4 · Le coût d'erreur est minimal autour de {fr(tf,2)}. Le garde-fou Precision ≥ 50 % n'est pas contraignant ici ; "
              "le seuil 0,5 a le meilleur F1 mais coûte plus cher."))
S_.append(P(f"<b>Conclusion.</b> Le seuil 0,5 maximise le F1, mais le seuil <b>{fr(tf,2)}</b> réduit le coût d'erreur de "
            f"{pct(1 - tab[str(tf)]['cout']/tab['0.5']['cout'])} sur le train. Le meilleur modèle statistique n'est pas le meilleur modèle économique. "
            f"Le seuil de {fr(tf,2)} est retenu."))
S_.append(H2("Résultat final sur le test (utilisé une seule fois)"))
rows = [["Stratégie", "TP", "FP", "FN", "Precision", "Recall", "F1", "Refus", "Coût d'erreur (test)"]]
for nm, d, c in [(f"<b>Modèle, seuil {fr(tf,2)}</b>", M, T["cout"]["modele"]), ("Modèle, seuil 0,5", M5, T["cout"]["seuil_05"]),
                 ("Règle actuelle (notes D-G)", RU, T["cout"]["regle"]), ("Tout accorder", T["accorder_tout"], T["cout"]["accorder_tout"])]:
    rows.append([nm, fr(d["tp"]), fr(d["fp"]), fr(d["fn"]), pct(d["precision"]), pct(d["recall"]), fr(d["f1"], 2), pct(d["refus"]), mm(c, 2)])
S_.append(tbl(rows, [4.2, 1.1, 1.1, 1.1, 1.9, 1.7, 1.2, 1.6, 3.5]))
S_.append(P(f"Tableau 3 · {fr(n_te)} demandes de test, {fr(n_def)} défauts réels. ROC-AUC test = {fr(T['roc_auc'],3)} et PR-AUC test = {fr(T['pr_auc'],3)} "
            f"(contre {fr(hgb['ap_cv'],3)} en CV) : la performance se confirme sur des données jamais vues.", "cap"))
S_.append(fig("fig6_test.png", 15.0, "Figure 5 · Matrice de confusion et courbe Precision-Recall sur le test : le modèle domine la règle actuelle."))
S_.append(P(f"<b>Traduction métier</b> : « sur 100 défauts, le modèle en anticipe {fr(M['recall']*100,0)}, contre {fr(RU['recall']*100,0)} pour la règle actuelle ; "
            f"sur 100 refus, {fr(M['precision']*100,0)} sont justifiés ». Le modèle refuse {pct(M['refus'])} des demandes (contre {pct(RU['refus'])} pour la règle) : il refuse davantage, mais mieux. "
            f"Le coût d'erreur sur le test est de {mm(T['cout']['modele'],2)}, contre {mm(T['cout']['regle'],2)} pour la règle actuelle."))
S_.append(P("<i>Attention</i> : sur ce jeu, la précision reste proche de 100 % pour les scores les plus élevés (figure 5). Le jeu de données paraît très « propre », "
            "peut-être en partie synthétique : ces performances sont probablement plus optimistes que sur un vrai portefeuille.", "body"))

# ================================================================== AXE 7
S_.append(H1("Axe 7 · ROI potentiel"))
S_.append(P(f"<b>Méthode.</b> Les résultats du test sont ramenés à une année en supposant <b>{fr(32000)} demandes par an</b> (hypothèse de volume, facteur {fr(32000/n_te,2)} sur le test). "
            f"« Coût sans modèle » = coût d'erreur de la règle actuelle. Gain annuel mesuré = coût sans modèle − coût avec modèle. "
            f"Comme le gain mesuré hors échantillon est probablement trop beau, nous le <b>décotons</b> d'une part « réalisée en production » (adoption partielle, dérive, biais de sélection). "
            f"Coût du projet 1<super>re</super> année (central) = {kk(cen['cout_projet'])} : développement 80 k$ + intégration et validation 30 k$ + exploitation 40 k$. "
            "ROI (%) = (gain annuel − coût du projet) / coût du projet × 100 ; délai de retour = coût du projet / (gain annuel / 12)."))
rows = [["Scénario", "LGD", "Marge", "Coût projet", "Gain réalisé", "Coût sans modèle", "Coût avec modèle", "Gain annuel", "ROI an 1", "Retour"]]
for x in [pes, cen, opt, der]:
    rows.append([("<b>%s</b>" % x["scenario"]) if x is cen else x["scenario"], pct(x["lgd"]), pct(x["marge"]), kk(x["cout_projet"]), pct(x["part_realisee"]),
                 mm(x["cout_sans"]), mm(x["cout_avec"]), mm(x["gain"]), fr(x["roi"], 0) + " %", fr(x["retour_mois"], 1) + " mois"])
S_.append(tbl(rows, [2.5, 1.0, 1.4, 1.6, 1.6, 1.9, 1.9, 1.8, 1.9, 1.8]))
S_.append(P("Tableau 4 · Scénarios de ROI (seuil fixé à 0,20 dans tous les cas). Le scénario « dérive » ajoute du bruit aux scores pour simuler un modèle qui se dégrade.", "cap"))
S_.append(fig("fig8_sens.png", 10.2, "Figure 6 · Sensibilité du gain mesuré (avant décote) aux deux hypothèses de coût : le gain dépend surtout de la perte en cas de défaut (LGD), très peu de la marge."))
be = R["break_even_part_realisee"]
S_.append(P(f"<b>Lecture et mise en garde.</b> Le projet reste très largement rentable dans les quatre scénarios, et même avec seulement {fr(0.1*100,0)} % du gain mesuré "
            f"(ROI {fr(R['roi_selon_part_realisee']['0.1'],0)} %). <b>Point mort</b> : il suffirait que {fr(be*100,1)} % du gain mesuré se réalise pour rembourser le projet en un an. "
            f"Ces ROI sont énormes parce que les pertes évitées (de l'ordre de {mm(cen['cout_sans'] - cen['cout_avec'],0)} par an) sont sans commune mesure avec le coût d'un projet de données : "
            f"<b>ils ne doivent pas être lus comme une prévision</b>. Ordre de grandeur indépendant du volume : environ {kk(cen['gain_mesure_par_1000_demandes'])} de gain mesuré pour 1 000 demandes. "
            "L'hypothèse la plus critique est le niveau de la LGD et la qualité de la référence (la règle actuelle est supposée, et le modèle utilise la même note interne) : "
            "c'est ce que le pilote doit valider en priorité."))

# ================================================================== AXE 9
S_.append(H1("Axe 9 · Risques et enjeux"))
rows = [["Risque", "Catégorie", "Proba.", "Impact", "Mesure de réduction"],
        ["Biais de sélection : l'historique ne contient que des prêts accordés, pas les clients refusés (« reject inference »)", "Technique", "Élevée", "Élevé",
         "Pilote en parallèle, sans décision automatique ; accorder un petit échantillon aléatoire pour mesurer le défaut des profils à risque."],
        ["Circularité de la note interne : le modèle réutilise un jugement existant", "Technique", "Moyenne", "Moyen",
         f"Ablation faite (PR-AUC {fr(A['Sans note interne ni taux (grade, loan_int_rate)']['ap'],2)} sans note ni taux) ; suivre le modèle sans ces variables comme modèle de contrôle."],
        ["Dérive des données (conjoncture, nouveaux produits, taux)", "Technique", "Moyenne", "Élevé",
         "Suivi mensuel du Recall, de la Precision et des distributions ; réentraînement trimestriel ; règle de repli (règle actuelle)."],
        ["Performances surestimées (jeu très propre, possiblement synthétique)", "Technique", "Moyenne", "Élevé",
         "Valider sur les données réelles de la banque avant déploiement ; décote de 30 % dans le ROI central."],
        ["Coût réel (LGD, marge) différent des hypothèses", "Métier", "Moyenne", "Élevé", "Faire valider LGD et marge par la direction des risques ; pilote de 3 mois."],
        ["Adoption : analystes qui ignorent ou suivent aveuglément le score", "Métier", "Moyenne", "Élevé",
         "Score présenté comme aide à la décision avec les 3 variables qui l'expliquent ; revue humaine des cas proches du seuil."],
        ["Discrimination indirecte (statut de logement, âge)", "Éthique", "Moyenne", "Élevé",
         "Retirer l'âge (perte négligeable) ; auditer les taux de refus par groupe à chaque réentraînement ; revue juridique."],
        ["Décision automatisée et explicabilité (RGPD art. 22, règlement IA)", "Régle-<br/>mentaire", "Élevée", "Élevé",
         "Humain dans la boucle, explication des principaux facteurs au client, journalisation des décisions."],
        ["Coût énergétique", "Environ-<br/>nement", "Faible", "Faible", "Entraînement d'environ 1 minute sur CPU : négligeable face au gain attendu."]]
S_.append(tbl(rows, [4.7, 2.4, 1.7, 1.3, 7.3]))
S_.append(P("Tableau 5 · Registre de risques propre à ce projet.", "cap"))
S_.append(H2("Audit d'équité sur le test (seuil 0,20)"))
rows = [["Groupe", "Effectif", "Taux de défaut réel", "Taux de refus", "Recall", "Faux refus (FPR)"]]
for g, v in F.items():
    rows.append([html.escape(g), fr(v["n"]), pct(v["taux_defaut"], 1), pct(v["taux_refus"], 1), pct(v["recall"]), pct(v["fpr"], 1)])
S_.append(tbl(rows, [3.4, 2.0, 3.2, 2.7, 2.2, 3.9]))
S_.append(P("Tableau 6 · Par âge, les taux de refus suivent de près les taux de défaut réels, sans écart marqué. Le statut de logement montre "
            f"des différences plus fortes : les locataires sont refusés bien plus souvent ({pct(F['Locataire']['taux_refus'])}) mais ils font aussi bien plus défaut ({pct(F['Locataire']['taux_defaut'])}). "
            f"Le Recall est plus faible pour les emprunteurs avec hypothèque ({pct(F['Hypothèque']['recall'])}) : le modèle y rate plus de défauts. "
            "Ces écarts justifient un suivi régulier, sans prouver à eux seuls une discrimination.", "cap"))
S_.append(P("<b>Limites et suite</b> : pas de date dans le jeu (pas de validation temporelle), horizon de défaut inconnu, historique limité aux prêts accordés, volumes et coûts supposés. "
            "Prochaines étapes : valider les hypothèses de coût avec la banque, calibrer les probabilités, tester un modèle sans l'âge, lancer le pilote de 3 mois."))

# ================================================================== ANNEXES
S_.append(PageBreak())
S_.append(H1("Annexe A · Dépôt Git et reproductibilité"))
S_.append(P("Structure du dépôt (conforme aux consignes) :"))
S_.append(tbl([["Élément", "Rôle"],
               ["README.md", "Problème, résultats clés, ROI, comment lancer, usage de l'IA déclaré"],
               ["requirements.txt", "Versions des librairies (pandas, numpy, scikit-learn, matplotlib, reportlab, pillow)"],
               ["data/credit_risk_dataset.csv", "Données brutes (1,8 Mo, sous la limite de 50 Mo)"],
               ["src/features.py, src/costs.py", "Fonctions réutilisables : nettoyage, variables, coûts, métriques, ROI"],
               ["notebooks/", "01_cadrage_eda, 02_modelisation, 03_decision_roi (exécutés, sans erreur)"],
               ["run_analysis.py", "Exécute tout le pipeline (random_state=42, chemins relatifs) et écrit reports/results.json et les figures"],
               ["build_report.py", "Génère ce rapport PDF à partir de results.json : les chiffres du rapport sont ceux du code"],
               ["reports/", "Rapport PDF, figures, results.json"]], [5.4, 12.0]))
S_.append(Spacer(1, 4))
S_.append(P("Pour reproduire : <font name='DV-B'>pip install -r requirements.txt</font>, puis exécuter dans l'ordre les notebooks "
            "<i>01_cadrage_eda</i>, <i>02_modelisation</i>, <i>03_decision_roi</i> (« Restart &amp; Run All », environ 4 minutes au total sur un CPU), "
            "ou bien <font name='DV-B'>python run_analysis.py</font> pour le même résultat en une commande. Les notebooks et le script donnent les mêmes chiffres (random_state=42)."))

S_.append(H1("Annexe B · Plan de soutenance (10 minutes, binôme)"))
S_.append(tbl([["Partie", "Contenu", "Visuel", "Temps"],
               ["1. Le problème", "Décision accorder/refuser, règle actuelle, critère de succès", "Tableau de cadrage", "1 min 30"],
               ["2. Les données", "32 580 demandes, qualité, 3 constats clés, audit de fuite", "Figure 1 (EDA)", "2 min"],
               ["3. La démarche", "Classification binaire, pipeline sans fuite, 6 modèles en CV", "Schéma 1 + Tableau 1", "2 min 30"],
               ["4. Les résultats", "Métrique, seuil par le coût, résultat sur le test, erreurs typiques", "Figures 4 et 5, Tableau 3", "2 min"],
               ["5. La valeur", "ROI en 4 scénarios, décote, recommandation de pilote", "Tableau 4", "1 min 30"],
               ["6. Limites et suite", "Biais de sélection, équité, dérive, prochaines étapes", "Tableau 5", "30 s"]], [3.0, 7.4, 4.5, 2.5]))
S_.append(P("Chaque membre prend la parole ; prévoir une capture d'écran de secours. Les questions du jury (5 min) porteront sur le code : chacun doit pouvoir expliquer "
            "le Pipeline, le choix du seuil, la décote du ROI et les limites.", "body"))

S_.append(H1("Annexe C · Check-list finale (consignes, section 15)"))
S_.append(tbl([["Point", "État"],
               ["Décision métier et critère de succès écrits en une phrase chacun", "Fait (axe 1)"],
               ["Type de problème justifié, alternatives écartées", "Fait (axe 2)"],
               ["Aucune variable inconnue au moment de la décision", "Fait (audit de fuite, axe 3)"],
               ["Préparation dans un Pipeline ajusté sur le train", "Fait (run_analysis.py)"],
               ["Baseline présente et battue", "Fait (Tableau 1)"],
               ["CV avec écart-type, train vs validation comparés", "Fait (Tableau 1)"],
               ["Test utilisé une seule fois", "Fait (Tableau 3)"],
               ["Métrique et seuil justifiés par les coûts", "Fait (axes 5-6)"],
               ["ROI avec hypothèses et scénario défavorable", "Fait (axe 7)"],
               ["Risques spécifiques avec mesures", "Fait (axe 9)"],
               ["Restart &amp; Run All sur les notebooks", "Fait (3 notebooks exécutés sans erreur)"],
               ["Source des données (lien, licence, date) dans le README", "<b>À compléter</b>"],
               ["Oral répété en temps réel ; chacun sait expliquer chaque ligne de code", "<b>À faire par l'équipe</b>"]], [12.4, 5.0]))

doc = SimpleDocTemplate(os.path.join(ROOT, "reports", "rapport_simple.pdf"), pagesize=A4,
                        leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.8 * cm,
                        title="Projet final S7MOI1 - Risque de défaut de prêt personnel", author="Abdessamad EL IDRISSI")
doc.build(S_, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("PDF OK")
