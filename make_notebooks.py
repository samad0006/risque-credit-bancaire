"""Génère les 3 notebooks à partir de run_analysis.py (mêmes cellules de code, mêmes résultats)."""
import nbformat as nbf
L = open("run_analysis.py", encoding="utf-8").read().split("\n")
def sl(a, b): return "\n".join(L[a - 1:b]).strip("\n")

setup = sl(5, 49).replace('sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))', 'sys.path.insert(0, os.path.abspath("../src"))') \
    .replace('ROOT = os.path.dirname(os.path.abspath(__file__))', 'ROOT = os.path.abspath("..")') + "\n\n" + sl(273, 275)
data = sl(51, 96)
def md(t): return nbf.v4.new_markdown_cell(t.strip())
def code(t): return nbf.v4.new_code_cell(t)
def nb(cells, name):
    n = nbf.v4.new_notebook(); n.cells = cells
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(n, f"notebooks/{name}.ipynb")

nb([md("""# 01 · Cadrage métier et analyse des données (axes 1, 2, 3)
**Décision** : pour chaque demande de prêt personnel, accorder ou refuser (aide à la décision de l'analyste crédit).
**Cible** : `loan_status` (1 = défaut, 0 = remboursé), connue pour tout l'historique → classification binaire supervisée.
**Situation actuelle (hypothèse)** : refus des notes internes D à G. **Critère de succès** : Recall ≥ 80 % avec Precision ≥ 50 %, et coût d'erreur réduit d'au moins 30 % vs la règle actuelle.
Détails complets dans le rapport (axes 1 et 2)."""),
    code(setup), md("## Chargement, nettoyage, variables construites, split\nLe nettoyage (doublons, valeurs impossibles) est déterministe : il ne dépend d'aucune statistique apprise, donc sans fuite. Le split est fait **avant** toute préparation apprise."),
    code(data), code('print(R["raw"]); print(R["clean"]); print("taux de défaut :", round(R["default_rate"], 3)); print(R["split"])'),
    md("## Audit de fuite\nAucune colonne ne décrit l'après-octroi. `loan_grade` et `loan_int_rate` sont fixées par la banque à l'octroi (connues à la décision mais circulaires) : leur poids est mesuré par ablation dans le notebook 03."),
    md("## Graphiques (chaque figure est interprétée dans le rapport)"), code(sl(294, 319)), code(sl(320, 333))], "01_cadrage_eda")

nb([md("""# 02 · Modélisation (axe 4)
Six modèles comparés en validation croisée stratifiée à 5 plis sur le **train** : baseline, régression logistique, arbre libre, arbre réglé, Random Forest, HistGradientBoosting. Toute la préparation est dans un `Pipeline`. Le test n'est pas touché ici."""),
    code(setup), code(data), md("## Comparaison des modèles et sélection\nRègle fixée à l'avance : meilleur PR-AUC en CV ; si égalité à un écart-type près, modèle le plus simple."),
    code(sl(97, 141)), code(sl(277, 293)), code(sl(334, 349)),
    md("## Sauvegarde du modèle retenu (utilisé par le notebook 03)"),
    code('import joblib\nos.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)\njoblib.dump({"model": model, "chosen": chosen, "table": table}, os.path.join(ROOT, "reports", "model.joblib"))\nprint("modèle retenu :", chosen)')], "02_modelisation")

nb([md("""# 03 · Seuil, résultat sur le test, ROI, équité (axes 5, 6, 7, 9)
Le seuil est choisi par le coût sur les prédictions hors-échantillon du **train**. Le **test** n'est utilisé qu'une fois, pour le modèle et le seuil finaux.
Hypothèses (explicites, testées en sensibilité) : perte en cas de défaut = 60 % du montant, marge perdue sur un refus à tort = 12 %, 32 000 demandes/an, coût du projet 150 k$, 70 % du gain mesuré réalisé en production."""),
    code(setup), code(data),
    code('import joblib\nd = joblib.load(os.path.join(ROOT, "reports", "model.joblib"))\nmodel, chosen, table = d["model"], d["chosen"], d["table"]\nprint(chosen)'),
    md("## Seuil par le coût (prédictions OOF du train)"), code(sl(142, 163)), code(sl(350, 364)),
    md("## Résultat final sur le test (une seule fois)"), code(sl(164, 182)), code(sl(365, 384)),
    md("## ROI et analyse de sensibilité"), code(sl(183, 219)), code('import pandas as pd\npd.DataFrame(R["roi"]).round(2)'), code(sl(396, 405)),
    md("## Importance des variables, équité, ablations, erreurs"), code(sl(220, 268)), code(sl(385, 395)),
    code('print(R["perm_importance"]); print(R["ablation"]); print(R["errors"])\npd.DataFrame(R["fairness"]).T.round(3)')], "03_decision_roi")
print("notebooks écrits")
