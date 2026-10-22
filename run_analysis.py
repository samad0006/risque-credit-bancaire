"""Projet final S7MOI1 - Risque de défaut de prêt personnel.
Exécution de bout en bout :  python run_analysis.py
Produit reports/results.json et reports/figures/*.png (les chiffres du rapport viennent d'ici).
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from features import CAT_COLS, NUM_COLS, TARGET, add_features, clean, load_raw
from costs import confusion, metrics, roi, total_cost

warnings.filterwarnings("ignore")
RS = 42
ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)

# ---------------- hypothèses business (axes 6-7), toutes testées en sensibilité -------------
LGD, MARGIN = 0.60, 0.12          # perte si défaut = 60 % du montant ; marge nette perdue si refus = 12 %
N_ANNUAL = 32_000                 # nombre de demandes traitées par an (hypothèse = taille du jeu)
COUT_PROJET = 150_000             # 1re année : développement 80 k + intégration/validation 30 k + exploitation 40 k
GAIN_REALISE = 0.70               # décote : seule une part du gain mesuré hors échantillon est réalisée en production
GUARDRAIL_PRECISION = 0.50        # au plus 1 client sain refusé pour 1 défaut évité

NAVY, TEAL, ORANGE, GREY, RED = "#1F3A5F", "#2A9D8F", "#E76F51", "#8D99AE", "#C0392B"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 150})
R = {}                            # tous les résultats
t0 = time.time()

# ------------------------------------------------------------------ 1. données
raw = load_raw(os.path.join(ROOT, "data", "credit_risk_dataset.csv"))
R["raw"] = {"n": len(raw), "cols": raw.shape[1],
            "missing": raw.isna().sum()[lambda s: s > 0].to_dict(),
            "age_max": float(raw.person_age.max()), "emp_max": float(raw.person_emp_length.max()),
            "income_max": float(raw.person_income.max())}
df, qc = clean(raw)
R["clean"] = qc
df = add_features(df)
R["default_rate"] = float(df[TARGET].mean())
R["n_clean"] = len(df)
R["rate_by"] = {
    "grade": df.groupby("loan_grade")[TARGET].mean().round(3).to_dict(),
    "prior_default": df.groupby("cb_person_default_on_file")[TARGET].mean().round(3).to_dict(),
    "home": df.groupby("person_home_ownership")[TARGET].mean().round(3).to_dict(),
    "intent": df.groupby("loan_intent")[TARGET].mean().round(3).to_dict(),
}
df["lpi_bin"] = pd.qcut(df["loan_percent_income"], 8, duplicates="drop")
R["rate_by"]["lpi_bin_first_last"] = [float(df.groupby("lpi_bin", observed=True)[TARGET].mean().iloc[0]),
                                      float(df.groupby("lpi_bin", observed=True)[TARGET].mean().iloc[-1])]
R["missing_rate_by_default"] = {
    "int_rate_missing": df.assign(m=df.loan_int_rate.isna()).groupby("m")[TARGET].mean().round(3).to_dict()}

# ------------------------------------------------------------------ 2. split (AVANT toute préparation)
X = df[NUM_COLS + CAT_COLS]
y = df[TARGET]
idx_tr, idx_te = train_test_split(df.index, test_size=0.2, stratify=y, random_state=RS)
X_tr, X_te, y_tr, y_te = X.loc[idx_tr], X.loc[idx_te], y.loc[idx_tr], y.loc[idx_te]
amt_tr, amt_te = df.loc[idx_tr, "loan_amnt"].values, df.loc[idx_te, "loan_amnt"].values
R["split"] = {"train": len(X_tr), "test": len(X_te), "default_train": float(y_tr.mean()),
              "default_test": float(y_te.mean())}


def make_prep(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)])


def make_pipe(clf, num_cols=NUM_COLS, cat_cols=CAT_COLS):
    return Pipeline([("prep", make_prep(num_cols, cat_cols)), ("clf", clf)])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

# ------------------------------------------------------------------ 3. modèles (axe 4)
specs = {
    "Baseline (probabilité a priori)": (DummyClassifier(strategy="prior"), {}, 0),
    "Régression logistique": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                              {"clf__C": [0.01, 0.1, 1, 10]}, 1),
    "Arbre sans max_depth": (DecisionTreeClassifier(random_state=RS), {}, 2),
    "Arbre (max_depth réglé)": (DecisionTreeClassifier(random_state=RS, min_samples_leaf=20),
                                {"clf__max_depth": [3, 5, 7, 10]}, 2),
    "Random Forest": (RandomForestClassifier(n_estimators=150, random_state=RS, n_jobs=-1,
                                             min_samples_leaf=3),
                      {"clf__max_depth": [6, 10, 16]}, 3),
    "Gradient Boosting (HGB)": (HistGradientBoostingClassifier(max_iter=250, random_state=RS,
                                                              early_stopping=True),
                                {"clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1]}, 4),
}
scoring = {"ap": "average_precision", "auc": "roc_auc"}
table, fitted, complexity = [], {}, {}
for name, (clf, grid, cx) in specs.items():
    pipe = make_pipe(clf)
    gs = GridSearchCV(pipe, grid if grid else {"clf": [clf]}, scoring=scoring, refit="ap", cv=cv,
                      return_train_score=True, n_jobs=1)
    gs.fit(X_tr, y_tr)
    i = gs.best_index_
    cr = gs.cv_results_
    table.append({"modele": name, "params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()
                                             if k != "clf"},
                  "ap_train": float(cr["mean_train_ap"][i]), "ap_cv": float(cr["mean_test_ap"][i]),
                  "ap_cv_std": float(cr["std_test_ap"][i]), "auc_cv": float(cr["mean_test_auc"][i]),
                  "auc_cv_std": float(cr["std_test_auc"][i])})
    fitted[name] = gs.best_estimator_
    complexity[name] = cx
    print(f"{name:34s} AP cv {table[-1]['ap_cv']:.3f} ± {table[-1]['ap_cv_std']:.3f} "
          f"(train {table[-1]['ap_train']:.3f})  [{time.time()-t0:.0f}s]", flush=True)
R["models"] = table

# règle de sélection a priori : meilleur PR-AUC en CV ; parmi les modèles à moins d'un écart-type
# du meilleur, on retient le plus simple (règle de l'écart-type)
best_row = max(table, key=lambda r: r["ap_cv"])
ok = [r for r in table if r["ap_cv"] >= best_row["ap_cv"] - best_row["ap_cv_std"]]
chosen = min(ok, key=lambda r: (complexity[r["modele"]], -r["ap_cv"]))["modele"]
R["chosen_model"] = chosen
R["best_by_ap"] = best_row["modele"]
model = fitted[chosen]
print("Modèle retenu :", chosen, flush=True)

# ------------------------------------------------------------------ 4. seuil par le coût (axe 6) -> OOF sur le TRAIN
oof = cross_val_predict(clone(model), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
grid_t = np.round(np.arange(0.05, 0.91, 0.01), 2)
rows = []
for t in grid_t:
    ref = oof >= t
    m = metrics(y_tr.values, ref)
    m["t"] = float(t)
    m["cout"] = total_cost(y_tr.values, ref, amt_tr, LGD, MARGIN)
    rows.append(m)
thr = pd.DataFrame(rows)
t_cost_free = float(thr.loc[thr.cout.idxmin(), "t"])
feas = thr[thr.precision >= GUARDRAIL_PRECISION]
t_final = float(feas.loc[feas.cout.idxmin(), "t"])
R["thresholds"] = {
    "t_theorique": MARGIN / (MARGIN + LGD),
    "t_cout_min_sans_contrainte": t_cost_free, "t_final": t_final,
    "oof_table": {str(t): {k: float(thr.loc[thr.t == t, k].iloc[0]) for k in
                           ["precision", "recall", "f1", "refus", "cout"]}
                  for t in sorted({0.5, round(t_cost_free, 2), round(t_final, 2), 0.3})},
    "oof_cout_accorder_tout": total_cost(y_tr.values, np.zeros(len(y_tr), bool), amt_tr, LGD, MARGIN)}

# ------------------------------------------------------------------ 5. TEST (utilisé une seule fois)
p_te = model.predict_proba(X_te)[:, 1]
ref_model = p_te >= t_final
ref_all = np.zeros(len(y_te), bool)
ref_rule = (X_te["grade_num"].values >= 4)          # règle actuelle supposée : refus des notes D à G
ref_t05 = p_te >= 0.5
yt = y_te.values
R["test"] = {
    "roc_auc": float(roc_auc_score(yt, p_te)), "pr_auc": float(average_precision_score(yt, p_te)),
    "baseline_pr_auc": float(yt.mean()),
    "modele": metrics(yt, ref_model), "seuil_05": metrics(yt, ref_t05),
    "regle_notes_DG": metrics(yt, ref_rule), "accorder_tout": metrics(yt, ref_all),
    "cout": {"modele": total_cost(yt, ref_model, amt_te, LGD, MARGIN),
             "seuil_05": total_cost(yt, ref_t05, amt_te, LGD, MARGIN),
             "regle": total_cost(yt, ref_rule, amt_te, LGD, MARGIN),
             "accorder_tout": total_cost(yt, ref_all, amt_te, LGD, MARGIN)},
    "n_defauts": int(yt.sum()), "n": int(len(yt))}
print("TEST", json.dumps(R["test"]["modele"]), flush=True)

# ------------------------------------------------------------------ 6. ROI + sensibilité (axe 7)
scale = N_ANNUAL / len(yt)


def annual_costs(lgd, margin, p=p_te, t=t_final):
    c_m = total_cost(yt, p >= t, amt_te, lgd, margin) * scale
    c_r = total_cost(yt, ref_rule, amt_te, lgd, margin) * scale
    c_a = total_cost(yt, ref_all, amt_te, lgd, margin) * scale
    return c_m, c_r, c_a


rng = np.random.default_rng(RS)
p_noisy = np.clip(p_te + rng.normal(0, 0.15, len(p_te)), 0, 1)
scen_def = [("Pessimiste", 0.40, 0.15, 250_000, p_te, 0.40),
            ("Central", LGD, MARGIN, COUT_PROJET, p_te, GAIN_REALISE),
            ("Optimiste", 0.75, 0.10, 120_000, p_te, 1.00),
            ("Dérive (scores bruités)", LGD, MARGIN, COUT_PROJET, p_noisy, GAIN_REALISE)]
scen = []
for nm, lgd, mg, cp, p, frac in scen_def:
    c_m, c_r, c_a = annual_costs(lgd, mg, p=p)
    g_mesure = c_r - c_m
    g = g_mesure * frac
    r_, back = roi(g, cp)
    scen.append({"scenario": nm, "lgd": lgd, "marge": mg, "cout_projet": cp, "part_realisee": frac,
                 "cout_sans": c_r, "cout_avec": c_m, "gain_mesure": g_mesure, "gain": g, "roi": r_,
                 "retour_mois": back, "gain_vs_accorder_tout": c_a - c_m,
                 "gain_mesure_par_1000_demandes": g_mesure / N_ANNUAL * 1000})
R["roi"] = scen
lgds = [0.30, 0.45, 0.60, 0.75, 0.90]
margins = [0.06, 0.09, 0.12, 0.15, 0.18]
sens = np.array([[annual_costs(l, m)[1] - annual_costs(l, m)[0] for m in margins] for l in lgds])
R["sens"] = {"lgds": lgds, "margins": margins, "gain": sens.tolist()}
# part minimale du gain mesuré qui doit se réaliser pour rembourser le projet (scénario central)
g_central = [x for x in scen if x["scenario"] == "Central"][0]["gain_mesure"]
R["break_even_part_realisee"] = COUT_PROJET / g_central
R["roi_selon_part_realisee"] = {str(f): roi(g_central * f, COUT_PROJET)[0] for f in (1.0, 0.7, 0.4, 0.1, 0.05)}

# ------------------------------------------------------------------ 7. diagnostics, équité, ablations
perm = permutation_importance(model, X_te, y_te, scoring="average_precision", n_repeats=5,
                              random_state=RS, n_jobs=1)
imp = pd.Series(perm.importances_mean, index=X_te.columns).sort_values(ascending=False)
R["perm_importance"] = imp.round(4).head(10).to_dict()

age = X_te["person_age"].values
groups = {"Âge < 25": age < 25, "Âge 25-29": (age >= 25) & (age < 30), "Âge ≥ 30": age >= 30,
          "Locataire": X_te.person_home_ownership.values == "RENT",
          "Hypothèque": X_te.person_home_ownership.values == "MORTGAGE",
          "Propriétaire": X_te.person_home_ownership.values == "OWN"}
fair = {}
for g, m in groups.items():
    yy, rr = yt[m], ref_model[m]
    tp, fp, fn, tn = confusion(yy, rr)
    fair[g] = {"n": int(m.sum()), "taux_defaut": float(yy.mean()), "taux_refus": float(rr.mean()),
               "recall": tp / (tp + fn) if tp + fn else 0.0, "fpr": fp / (fp + tn) if fp + tn else 0.0}
R["fairness"] = fair

abl = {}
base_cols = (NUM_COLS, CAT_COLS)
variants = {"Modèle complet": (NUM_COLS, CAT_COLS),
            "Sans l'âge": ([c for c in NUM_COLS if c not in ("person_age", "emp_to_age")], CAT_COLS),
            "Sans note interne ni taux (grade, loan_int_rate)":
                ([c for c in NUM_COLS if c not in ("grade_num", "loan_int_rate")], CAT_COLS)}
for nm, (nc, cc) in variants.items():
    est = clone(model)
    est = make_pipe(clone(model.named_steps["clf"]), nc, cc)
    s = cross_val_score(est, X_tr[nc + cc], y_tr, cv=cv, scoring="average_precision")
    a = cross_val_score(est, X_tr[nc + cc], y_tr, cv=cv, scoring="roc_auc")
    abl[nm] = {"ap": float(s.mean()), "ap_std": float(s.std()), "auc": float(a.mean())}
R["ablation"] = abl

# erreurs typiques : profil moyen des FN (défauts non détectés) vs TP
dte = df.loc[idx_te].copy()
dte["p"] = p_te
fn_mask = (yt == 1) & ~ref_model
tp_mask = (yt == 1) & ref_model
R["errors"] = {
    "fn_n": int(fn_mask.sum()), "tp_n": int(tp_mask.sum()),
    "fn_grade_mean": float(dte.loc[fn_mask, "grade_num"].mean()),
    "tp_grade_mean": float(dte.loc[tp_mask, "grade_num"].mean()),
    "fn_lpi_mean": float(dte.loc[fn_mask, "loan_percent_income"].mean()),
    "tp_lpi_mean": float(dte.loc[tp_mask, "loan_percent_income"].mean()),
    "fn_amnt_mean": float(dte.loc[fn_mask, "loan_amnt"].mean()),
    "tp_amnt_mean": float(dte.loc[tp_mask, "loan_amnt"].mean()),
    "fn_prior_default_share": float((dte.loc[fn_mask, "cb_person_default_on_file"] == "Y").mean()),
    "tp_prior_default_share": float((dte.loc[tp_mask, "cb_person_default_on_file"] == "Y").mean())}

with open(os.path.join(ROOT, "reports", "results.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, indent=2, ensure_ascii=False, default=float)

# ====================================================================== FIGURES
def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close(fig)


# fig 1 : schéma du pipeline
fig, ax = plt.subplots(figsize=(9.2, 2.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 24)
boxes = [("Données\nbrutes\n32 580 lignes", NAVY), ("Nettoyage\ndoublons,\nvaleurs\nimpossibles", NAVY),
         ("Split 80/20\nstratifié", ORANGE), ("Pipeline\nsklearn\nimputation,\néchelle,\none-hot", TEAL),
         ("6 modèles\n+ CV 5 plis", TEAL), ("Seuil par\nle coût\n(OOF du\ntrain)", ORANGE),
         ("Test\n(1 seule\nfois)\n+ ROI", NAVY)]
w, gap = 12.4, 2.0
for i, (txt, c) in enumerate(boxes):
    x = 1 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 5), w, 14, boxstyle="round,pad=0.3", fc=c, ec="none", alpha=.92))
    ax.text(x + w / 2, 12, txt, ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    if i < len(boxes) - 1:
        ax.annotate("", xy=(x + w + gap - .2, 12), xytext=(x + w + .3, 12),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4))
ax.text(1 + 2 * (w + gap) + w / 2, 1.2, "le test est mis de côté ici", ha="center", fontsize=7, color=ORANGE)
save(fig, "fig1_pipeline.png")

# fig 2 : EDA
fig, axs = plt.subplots(2, 2, figsize=(8.6, 6))
ax = axs[0, 0]
vc = df[TARGET].value_counts(normalize=True).sort_index()
ax.bar(["Remboursé (0)", "Défaut (1)"], vc.values * 100, color=[TEAL, ORANGE])
for i, v in enumerate(vc.values): ax.text(i, v * 100 + 1, f"{v*100:.1f} %", ha="center", fontweight="bold")
ax.set_ylabel("% des demandes"); ax.set_title("Classe défaut minoritaire (~22 %)"); ax.set_ylim(0, 95)
ax = axs[0, 1]
g = df.groupby("loan_grade")[TARGET].mean() * 100
ax.bar(g.index, g.values, color=[TEAL if v < 25 else ORANGE for v in g.values])
ax.set_ylabel("taux de défaut (%)"); ax.set_xlabel("note interne (A = peu risqué)")
ax.set_title("Le défaut explose à partir de la note D")
ax = axs[1, 0]
g = df.groupby("lpi_bin", observed=True)[TARGET].mean() * 100
ax.plot(range(len(g)), g.values, marker="o", color=NAVY)
ax.set_xticks(range(len(g))); ax.set_xticklabels([f"{b.mid:.2f}" for b in g.index], fontsize=7)
ax.set_xlabel("charge de la dette : montant / revenu (centre de la tranche)"); ax.set_ylabel("taux de défaut (%)")
ax.set_title("Plus la charge pèse sur le revenu, plus le risque monte")
ax = axs[1, 1]
cat = pd.concat([df.groupby("cb_person_default_on_file")[TARGET].mean(),
                 df.groupby("person_home_ownership")[TARGET].mean().loc[["OWN", "MORTGAGE", "RENT"]]]) * 100
lab = ["Pas de défaut\npassé", "Défaut\npassé", "Proprié-\ntaire", "Hypo-\nthèque", "Locataire"]
ax.bar(lab, cat.values, color=[TEAL, ORANGE, TEAL, GREY, ORANGE]); ax.tick_params(axis="x", labelsize=7)
ax.set_ylabel("taux de défaut (%)"); ax.set_title("Historique de crédit et statut de logement")
fig.tight_layout(); save(fig, "fig2_eda.png")

# fig 3 : corrélations
num_eda = df[["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate",
              "loan_percent_income", "cb_person_cred_hist_length", "grade_num", TARGET]]
cm = num_eda.corr(method="spearman")
lbl = ["âge", "revenu", "ancienneté\nemploi", "montant", "taux\nintérêt", "montant/\nrevenu",
       "hist.\ncrédit", "note\ninterne", "DÉFAUT"]
fig, ax = plt.subplots(figsize=(6.4, 5))
im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(9)); ax.set_xticklabels(lbl, fontsize=6.5); ax.set_yticks(range(9)); ax.set_yticklabels(lbl, fontsize=6.5)
for i in range(9):
    for j in range(9): ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
ax.set_title("Corrélations (Spearman) : le défaut suit note, taux et charge de la dette")
fig.colorbar(im, shrink=.8); save(fig, "fig3_corr.png")

# fig 4 : comparaison des modèles
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
names = [r["modele"] for r in table]; short = ["Baseline", "Rég.\nlogistique", "Arbre\nlibre", "Arbre\nréglé", "Random\nForest", "HGB"]
xs = np.arange(len(names)); wbar = 0.38
axs[0].bar(xs - wbar / 2, [r["ap_train"] for r in table], wbar, label="train", color=GREY)
axs[0].bar(xs + wbar / 2, [r["ap_cv"] for r in table], wbar, yerr=[r["ap_cv_std"] for r in table],
           label="validation croisée (±1 σ)", color=TEAL, capsize=3)
axs[0].set_xticks(xs); axs[0].set_xticklabels(short, fontsize=7); axs[0].set_ylabel("PR-AUC")
axs[0].set_title("L'arbre libre sur-apprend ; RF et HGB généralisent"); axs[0].legend(fontsize=7, frameon=False)
ci = names.index(chosen)
axs[0].annotate("retenu", (ci + wbar / 2, table[ci]["ap_cv"] + table[ci]["ap_cv_std"] + .02), ha="center", color=ORANGE, fontweight="bold")
axs[1].barh(range(len(names)), [r["auc_cv"] for r in table], xerr=[r["auc_cv_std"] for r in table], color=NAVY, capsize=3)
axs[1].set_yticks(range(len(names))); axs[1].set_yticklabels([s.replace("\n", " ") for s in short], fontsize=7)
axs[1].set_xlim(0.4, 1); axs[1].set_xlabel("ROC-AUC (CV)"); axs[1].invert_yaxis(); axs[1].set_title("Même classement en ROC-AUC")
fig.tight_layout(); save(fig, "fig4_models.png")

# fig 5 : seuil
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
ax = axs[0]
ax.plot(thr.t, thr.cout / 1e6, color=NAVY, lw=2)
ax.axvline(t_final, color=ORANGE, ls="--"); ax.axvline(0.5, color=GREY, ls=":")
ax.text(t_final + .01, ax.get_ylim()[1] * .92, f"retenu t={t_final:.2f}", color=ORANGE, fontsize=8)
ax.text(0.51, ax.get_ylim()[1] * .78, "défaut 0,5", color=GREY, fontsize=8)
ax.set_xlabel("seuil de refus (score ≥ seuil)"); ax.set_ylabel("coût d'erreur total, train OOF (M$)")
ax.set_title("Le seuil 0,5 coûte plus que le seuil retenu")
ax = axs[1]
ax.plot(thr.t, thr.precision, label="Precision", color=TEAL, lw=2); ax.plot(thr.t, thr.recall, label="Recall", color=ORANGE, lw=2)
ax.axhline(GUARDRAIL_PRECISION, color=GREY, ls=":"); ax.text(.45, GUARDRAIL_PRECISION + .02, "garde-fou Precision 50 %", fontsize=7, color=GREY)
ax.axvline(t_final, color=ORANGE, ls="--"); ax.set_xlabel("seuil"); ax.set_title("Compromis Precision / Recall"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); save(fig, "fig5_threshold.png")

# fig 6 : test
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
ax = axs[0]
tp, fp, fn, tn = confusion(yt, ref_model)
cmx = np.array([[tn, fp], [fn, tp]])
ax.imshow(cmx, cmap="Blues")
for i in range(2):
    for j in range(2): ax.text(j, i, f"{cmx[i, j]:,}".replace(",", " "), ha="center", va="center",
                               color="white" if cmx[i, j] > cmx.max() / 2 else "black", fontsize=13, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["accordé", "refusé"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["remboursé", "défaut"])
ax.set_xlabel("décision du modèle"); ax.set_ylabel("réalité"); ax.set_title(f"Matrice de confusion (test, seuil {t_final:.2f})")
ax = axs[1]
pr, rc, _ = precision_recall_curve(yt, p_te)
ax.plot(rc, pr, color=NAVY, lw=2, label=f"modèle (PR-AUC {R['test']['pr_auc']:.2f})")
ax.axhline(yt.mean(), color=GREY, ls="--", label=f"baseline ({yt.mean():.2f})")
mm = R["test"]["modele"]; ax.scatter([mm["recall"]], [mm["precision"]], color=ORANGE, zorder=5, s=50, label="point retenu")
rr_ = R["test"]["regle_notes_DG"]; ax.scatter([rr_["recall"]], [rr_["precision"]], color=RED, marker="s", zorder=5, s=45, label="règle actuelle (notes D-G)")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Courbe Precision-Recall sur le test"); ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.42))
fig.tight_layout(); save(fig, "fig6_test.png")

# fig 7 : importance
fig, ax = plt.subplots(figsize=(6.2, 3.2))
top = imp.head(8)[::-1]
nice = {"grade_num": "note interne", "loan_percent_income": "montant / revenu", "loan_int_rate": "taux d'intérêt",
        "person_home_ownership": "logement", "log_income": "revenu (log)", "cb_person_default_on_file": "défaut passé",
        "loan_intent": "objet du prêt", "person_emp_length": "ancienneté emploi", "loan_amnt": "montant",
        "person_age": "âge", "emp_to_age": "ancienneté / âge", "cb_person_cred_hist_length": "historique crédit"}
ax.barh([nice.get(i, i) for i in top.index], top.values, color=TEAL)
ax.set_xlabel("baisse de PR-AUC si la variable est mélangée (test)")
ax.set_title("Variables qui comptent le plus pour le modèle"); save(fig, "fig7_importance.png")

# fig 8 : sensibilité ROI
fig, ax = plt.subplots(figsize=(6, 3.6))
im = ax.imshow(sens / 1e6, cmap="YlGn", aspect="auto", origin="lower")
ax.set_xticks(range(len(margins))); ax.set_xticklabels([f"{m:.0%}" for m in margins])
ax.set_yticks(range(len(lgds))); ax.set_yticklabels([f"{l:.0%}" for l in lgds])
for i in range(len(lgds)):
    for j in range(len(margins)): ax.text(j, i, f"{sens[i, j]/1e6:.2f}", ha="center", va="center", fontsize=8, color="white" if sens[i, j] > 0.6 * sens.max() else "black")
ax.set_xlabel("marge nette perdue sur un refus à tort"); ax.set_ylabel("perte en cas de défaut (LGD)")
ax.set_title("Gain annuel mesuré vs règle actuelle (M$, avant décote)"); fig.colorbar(im, shrink=.8); save(fig, "fig8_sens.png")

print(f"OK en {time.time()-t0:.0f}s")
