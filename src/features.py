"""Nettoyage et ingénierie de variables (déterministes : aucune statistique apprise,
donc aucune fuite possible avant le split)."""
import numpy as np
import pandas as pd

TARGET = "loan_status"          # 1 = défaut de paiement, 0 = remboursé
GRADE_MAP = {g: i + 1 for i, g in enumerate("ABCDEFG")}   # A (peu risqué) -> G (très risqué)

NUM_COLS = ["person_age", "log_income", "person_emp_length", "loan_amnt",
            "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
            "grade_num", "emp_to_age"]
CAT_COLS = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]


def load_raw(path):
    return pd.read_csv(path)


def clean(df):
    """Règles métier de nettoyage (appliquées avant le split car elles ne dépendent
    d'aucune statistique du jeu de données)."""
    n0 = len(df)
    df = df.drop_duplicates().copy()
    n_dup = n0 - len(df)
    # valeurs impossibles : âge > 100 ans, ancienneté > 60 ans (ex. 123 ans d'emploi)
    mask_bad = (df["person_age"] > 100) | (df["person_emp_length"] > 60)
    n_bad = int(mask_bad.sum())
    df = df[~mask_bad].reset_index(drop=True)
    return df, {"lignes_brutes": n0, "doublons": n_dup, "valeurs_impossibles": n_bad,
                "lignes_finales": len(df)}


def add_features(df):
    """3 variables construites (justification métier dans le rapport)."""
    out = df.copy()
    out["log_income"] = np.log1p(out["person_income"])               # revenus très asymétriques
    out["grade_num"] = out["loan_grade"].map(GRADE_MAP)              # note interne ordinale A..G
    out["emp_to_age"] = out["person_emp_length"] / out["person_age"]  # stabilité professionnelle
    return out
