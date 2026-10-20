# Cahier des charges — <nom de la stratégie>

> Modèle du chapitre 18. À remplir AVANT d'écrire le code, puis à confier à Claude Code.
> Toute section vide = la stratégie n'est pas prête.

## 1. Hypothèse économique
- Pourquoi l'écart devrait-il revenir à la moyenne ? (ancre contractuelle, même sous-jacent,
  même secteur, arbitrage de frais…) — 3 lignes maximum.
- Qu'est-ce qui casserait la relation ? (événements à surveiller)

## 2. Univers et données
- Instruments, plateformes, contrats (identifiants courtier).
- Fréquence, période de formation, période de test (dates), source des données.
- Traitement des radiations, dividendes, divisions, jours fériés différents.

## 3. Modèle
- Construction de l'écart (prix ou log-prix, ratio statique / glissant / Kalman).
- Paramètres estimés et fenêtres (toutes strictement passées).

## 4. Règles de trading
- Entrée, sortie, stop en z, stop en durée, alarme de rupture.
- Taille : ciblage de volatilité ou fraction de Kelly (préciser), plafonds.

## 5. Coûts déclarés (pb par jambe et par jour)
| Poste | Valeur | Source |
|---|---|---|
| Commission | | |
| Demi-fourchette | | |
| Glissement | | |
| Emprunt / financement / funding | | |

## 6. Variantes essayées
- Liste exhaustive (nombre N) — nécessaire au Sharpe déflaté.

## 7. Critères de passage (chapitre 18)
- G1 cointégration (p < 0,05 sur la formation)
- G2 demi-vie entre 1 et 30 périodes
- G3 DSR > 0,95 avec N variantes
- G4 test hors échantillon net positif
- G5 coût de rentabilité nulle > 2 × coût déclaré

## 8. Tests exigés
- Absence d'anticipation (modifier le futur ne change pas le passé).
- Parité backtest vectorisé / version événementielle.
- Coûts : payés seulement aux changements de position.
- Limites de risque : un ordre qui les violerait est refusé.

## 9. Déploiement
- Paper trading : durée, contrôles prévus (dates), seuil d'arrêt.
- Taille réduite : capital, durée.
- Taille cible : conditions.
- Rappel : jamais sur un compte de prop firm pour une stratégie inter-plateformes.
