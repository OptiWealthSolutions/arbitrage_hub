# Arbitrage Hub — Horacle Capital

Cours complet et laboratoire de recherche sur l'arbitrage statistique, le pairs trading
et l'arbitrage inter-plateformes, du tableau blanc jusqu'à des stratégies codées,
testées et auditables avec Claude Code.

## 📘 Le cours (un seul PDF)

**[docs/cours/arbitrage_hub_cours_complet.pdf](docs/cours/arbitrage_hub_cours_complet.pdf)** — 19 chapitres, 4 parties, annexes.

| Partie | Chapitres |
|---|---|
| Introduction | 0 — Cartographie de l'arbitrage, méthode, revue de littérature, analyse du moteur `atjte` |
| I. Fondations statistiques | 1 Arbitrage, stationnarité, retour à la moyenne · 2 Régression et estimation · 3 Tests de stationnarité · 4 Cointégration · 5 Ornstein-Uhlenbeck |
| II. Méthodes de pairs trading | 6 Distances · 7 Sélection par cointégration · 8 Kalman · 9 Copules · 10 Multi-actifs (ACP) · 11 Apprentissage automatique |
| III. Arbitrage inter-plateformes | 12 Microstructure et coûts · 13 Spatial et triangulaire · 14 Base, cash-and-carry, funding · 15 Exécution et architecture |
| IV. Validation, risque, production | 16 Backtest rigoureux · 17 Risque et limites de l'arbitrage · 18 Du papier au code avec Claude Code |
| Annexes | A Formulaire · B Notations · C Glossaire · D Reproduire les résultats |

Chaque chapitre suit la même structure : objectifs, intuition (image mentale), démonstrations,
vérification en Python, limites, fiche « tableau blanc », exercices, lectures conseillées.

## Méthode

1. Lire un chapitre, refaire les démonstrations au tableau blanc (photos dans `whiteboard/chXX/`).
2. Relancer `code/chXX/` : chaque chiffre du cours sort d'un script à graine fixée.
3. Faire les exercices, dont ceux « à coder ».
4. Passer au chapitre suivant seulement ensuite.

## Arborescence

```
docs/cours/          le cours : arbitrage_hub_cours_complet.tex/.pdf, chapitres/, figures/, references.bib
docs/specs/          cahiers des charges des stratégies (modèle : TEMPLATE_strategie.md)
code/chXX/           scripts de vérification de chaque chapitre + results.json
code/common/         style des figures, univers d'actions simulé
src/arbhub/          paquet réutilisable
  data/              chargement des données (seul module qui lit le disque)
  stats/             AR(1), demi-vie, z-score
  signals/           stratégie de paire (version recherche + version production)
  backtest/          moteur sans anticipation, Sharpe, PSR, DSR, PBO
  risk/              Kelly, ciblage de volatilité, CUSUM
  models/ execution/ à compléter (Kalman, OU, copules, simulateur d'exécution)
tests/               tests pytest (anticipation, parité, coûts, métriques, risque)
data/case_studies/   500 clôtures réelles GLD, GLDM, KO, PEP (Interactive Brokers)
papers/              liste des articles de référence + liens
whiteboard/          photos datées du tableau blanc
CLAUDE.md            règles de travail pour Claude Code
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # 24 tests
python code/ch16/ch16_backtest.py           # exemple : rejouer un chapitre
```

## Recompiler le PDF

```bash
cd docs/cours && sh build.sh
```

## Avertissement

Support pédagogique, pas un conseil en investissement. L'arbitrage statistique n'est pas sans risque.
Les stratégies inter-plateformes ne doivent pas tourner sur des comptes de prop firms, dont les règles
interdisent en général l'arbitrage de latence et la couverture croisée entre comptes.
