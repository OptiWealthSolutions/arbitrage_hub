# Règles de travail pour Claude Code — arbitrage_hub

Ce dépôt sert à apprendre puis à produire des stratégies d'arbitrage statistique.
Toute contribution de code doit respecter les règles suivantes.

## Principes
- **Tester au lieu d'affirmer.** Toute propriété statistique utilisée par une stratégie
  (stationnarité, demi-vie, cointégration) est vérifiée par un script ou un test, avec graine fixée.
- **Aucune anticipation.** Une fonction de signal ne lit jamais une donnée datée après l'instant
  de décision. Le P&L s'écrit `pos[t-1] * (S[t] - S[t-1])`. Un test dans `tests/` doit le garantir
  pour chaque nouvelle stratégie (par exemple : modifier les données futures ne change pas le signal passé).
- **Formation / test séparés.** Les paramètres (beta, moyenne, écart-type, seuils) sont estimés
  sur la période de formation uniquement, ou en fenêtre glissante strictement passée.
- **Coûts explicites.** Chaque backtest déclare son modèle de coûts (frais, fourchette, emprunt,
  financement) en points de base par jambe. Pas de résultat publié sans coûts.
- **Tests multiples.** Chaque résultat publié indique combien de variantes ont été essayées.
  Utiliser le Sharpe déflaté (chapitre 16) dès qu'on sélectionne parmi plusieurs variantes.

## Conventions
- Python ≥ 3.11, `numpy`, `pandas`, `statsmodels`, `scipy`, `matplotlib`, `pytest`.
- Code réutilisable dans `src/arbhub/<module>/`, scripts pédagogiques dans `code/chXX/`.
- Fonctions pures autant que possible ; I/O isolées dans `src/arbhub/data/`.
- Docstrings en français : hypothèse, entrées, sorties, référence (article + section).
- Chaque fonction de `src/arbhub` a au moins un test dans `tests/`.

## Sécurité
- Aucune clé API dans le dépôt (`.env` ignoré). Aucun ordre réel sans `LIVE_TRADING = True`
  explicite et validation humaine.
- Les stratégies inter-plateformes ne sont jamais déployées sur des comptes de prop firms.
