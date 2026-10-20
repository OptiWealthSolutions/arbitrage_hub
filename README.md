# Arbitrage Hub — Horacle Capital

Cours complet et laboratoire de recherche sur l'arbitrage statistique, le pairs trading
et l'arbitrage inter-plateformes, du tableau blanc jusqu'à des stratégies codées,
testées et auditables avec Claude Code.

## Avancement du cours

| # | Document | Statut |
|---|----------|--------|
| 00 | [Plan d'étude, revue de littérature, feuille de route](docs/00_plan/00_plan_du_cours.pdf) | ✅ |
| 01 | [Arbitrage, stationnarité et retour à la moyenne](docs/01_fondations/01_chapitre_fondations.pdf) | ✅ |
| 02 | Régression et estimation | ⏳ |
| 03 | Tests de stationnarité | ⏳ |
| 04 | Cointégration | ⏳ |
| 05 | Processus d'Ornstein-Uhlenbeck | ⏳ |
| 06–11 | Méthodes de pairs trading (distance, cointégration, Kalman, copules, multi-facteurs, ML) | ⏳ |
| 12–15 | Arbitrage inter-plateformes (microstructure, spatial, base/funding, exécution) | ⏳ |
| 16–18 | Backtest rigoureux, risque, mise en production | ⏳ |

## Méthode

1. Lire un chapitre, refaire les démonstrations au tableau blanc (photos dans `whiteboard/chXX/`).
2. Relancer les scripts `code/chXX/` : chaque script teste une hypothèse énoncée dans le chapitre.
3. Faire les exercices, dont ceux « à coder ».
4. Passer au chapitre suivant seulement ensuite.

## Arborescence

```
docs/        chapitres (.tex + .pdf), préambule et bibliographie communs dans docs/shared/
code/chXX/   scripts pédagogiques, un fichier = une hypothèse testée, résultats dans results/
src/arbhub/  paquet réutilisable (data, stats, models, signals, backtest, execution, risk)
tests/       tests pytest, dont tests anti-anticipation
papers/      liste des articles de référence + liens
whiteboard/  photos datées du tableau blanc
data/        données locales (non versionnées)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd code/ch01 && for f in h*.py; do python "$f"; done
pytest
```

## Recompiler les PDF

```bash
cd docs/01_fondations
pdflatex 01_chapitre_fondations && bibtex 01_chapitre_fondations && pdflatex 01_chapitre_fondations && pdflatex 01_chapitre_fondations
```

## Avertissement

Support pédagogique, pas un conseil en investissement. L'arbitrage statistique n'est pas sans risque.
Les stratégies inter-plateformes ne doivent pas tourner sur des comptes de prop firms, dont les règles
interdisent en général l'arbitrage de latence et la couverture croisée entre comptes.
