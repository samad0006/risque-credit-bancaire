"""Matrice de coûts et ROI.

Convention : on REFUSE le prêt si score de risque >= seuil.
 - FN (prêt accordé, client fait défaut)  : perte = LGD x montant
 - FP (prêt refusé, client aurait payé)   : manque à gagner = marge nette x montant
 - TP / TN : pas d'erreur -> coût d'erreur nul (le gain de marge des TN est identique avec
   ou sans modèle, il s'annule dans la comparaison).
"""
import numpy as np


def total_cost(y, refuse, amount, lgd, margin):
    y, refuse, amount = map(np.asarray, (y, refuse, amount))
    fn = (y == 1) & (~refuse)
    fp = (y == 0) & refuse
    return float((lgd * amount[fn]).sum() + (margin * amount[fp]).sum())


def confusion(y, refuse):
    y, refuse = np.asarray(y), np.asarray(refuse)
    tp = int(((y == 1) & refuse).sum()); fp = int(((y == 0) & refuse).sum())
    fn = int(((y == 1) & ~refuse).sum()); tn = int(((y == 0) & ~refuse).sum())
    return tp, fp, fn, tn


def metrics(y, refuse):
    tp, fp, fn, tn = confusion(y, refuse)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": f1, "refus": (tp + fp) / len(y)}


def roi(gain_annuel, cout_projet):
    roi_pct = (gain_annuel - cout_projet) / cout_projet * 100
    retour_mois = cout_projet / (gain_annuel / 12) if gain_annuel > 0 else float("inf")
    return roi_pct, retour_mois
