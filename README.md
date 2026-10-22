# Risque de défaut de prêt personnel — Projet final S7MOI1 (ECAM-EPMI)

**Problème** : aider la banque à accorder ou refuser un prêt en prédisant le défaut de paiement.
**Résultat clé (test, 6 482 demandes)** : 83 % des défauts détectés, 76 % des refus justifiés, coût d'erreur 1,70 M$ contre 5,71 M$ pour la règle actuelle supposée (notes D-G).

## Lancer
pip install -r requirements.txt
jupyter nbconvert --execute --inplace notebooks/*.ipynb   # ou ouvrir les 3 notebooks dans l'ordre 01, 02, 03
python run_analysis.py     # même résultat en une commande (~2 min), écrit reports/results.json et les figures
python build_report.py     # génère reports/rapport_simple.pdf

## À compléter par l'équipe
- Source des données : lien Kaggle exact, licence, date de consultation (copie utilisée : GitHub JoshLG18/DSE-EMP-Project).
- Relire le code et savoir expliquer chaque ligne (exigé en soutenance).

## Usage de l'IA générative
Claude (Anthropic) a été utilisé comme assistant : structure du dépôt, code (script et notebooks), première rédaction du rapport et des slides.
