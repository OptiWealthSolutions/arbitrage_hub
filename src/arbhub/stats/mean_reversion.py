"""Outils de base du chapitre 1 : estimation AR(1) et demi-vie.

Référence : chapitre 1, proposition 6.2 (demi-vie) ; Hamilton (1994), ch. 3.
"""
from __future__ import annotations

import numpy as np


def ar1_phi(x: np.ndarray) -> float:
    """Coefficient phi estimé par MCO de x_t sur (1, x_{t-1})."""
    x = np.asarray(x, dtype=float)
    return float(np.polyfit(x[:-1], x[1:], 1)[0])


def half_life(x: np.ndarray) -> float:
    """Demi-vie -ln2/ln(phi) en périodes ; +inf si phi n'est pas dans ]0, 1[."""
    phi = ar1_phi(x)
    return float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else float("inf")


def zscore_formation(spread_train: np.ndarray, spread_test: np.ndarray) -> np.ndarray:
    """z-score de la période de test avec moyenne et écart-type de la formation SEULEMENT."""
    m, d = float(np.mean(spread_train)), float(np.std(spread_train, ddof=1))
    return (np.asarray(spread_test, dtype=float) - m) / d
