"""Mesures de performance et de surapprentissage (chapitre 16).

Toutes les fonctions sur le ratio de Sharpe travaillent PAR PÉRIODE (non annualisé),
sauf `annualized_sharpe`. Notations : T observations, g3 = asymétrie, g4 = kurtosis (non excédentaire,
3 pour une loi normale).

Références :
- Lo (2002), The statistics of Sharpe ratios, FAJ 58(4) : erreur-type sous hypothèse i.i.d. normale ;
- Bailey et López de Prado (2014), The deflated Sharpe ratio, JPM 40(5) : PSR, DSR, maximum espéré ;
- Bailey, Borwein, López de Prado et Zhu (2017), The probability of backtest overfitting : CSCV, PBO.
"""
from __future__ import annotations

from itertools import combinations
from math import comb, e, sqrt

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329


def sharpe(r: np.ndarray) -> float:
    """Ratio de Sharpe par période : moyenne / écart-type (ddof=1)."""
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def annualized_sharpe(r: np.ndarray, periods_per_year: int = 252) -> float:
    return sharpe(r) * sqrt(periods_per_year)


def sharpe_se(sr: float, T: int, g3: float = 0.0, g4: float = 3.0) -> float:
    """Erreur-type de l'estimateur du Sharpe par période.
    Normal i.i.d. (g3=0, g4=3) : sqrt((1 + sr^2/2) / (T-1)) (Lo, 2002)."""
    return sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr, 1e-12) / (T - 1))


def probabilistic_sharpe(sr: float, sr_star: float, T: int, g3: float = 0.0, g4: float = 3.0) -> float:
    """PSR : probabilité que le vrai Sharpe dépasse sr_star, vu l'estimation sr sur T périodes."""
    return float(norm.cdf((sr - sr_star) / sharpe_se(sr, T, g3, g4)))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Maximum espéré de n_trials Sharpe estimés de stratégies SANS avantage (vrai Sharpe nul),
    de variance var_sr entre essais (Bailey et López de Prado, 2014)."""
    if n_trials <= 1:
        return 0.0
    a = (1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials)
    b = EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * e))
    return float(sqrt(var_sr) * (a + b))


def deflated_sharpe(r: np.ndarray, n_trials: int, var_trials_sr: float) -> float:
    """DSR : PSR du meilleur essai avec, comme seuil, le maximum espéré sous l'hypothèse nulle."""
    r = np.asarray(r, dtype=float)
    sr = sharpe(r)
    sr0 = expected_max_sharpe(n_trials, var_trials_sr)
    if r.std() == 0:            # variante qui n'a jamais traité : moments d'ordre 3 et 4 indéfinis
        return probabilistic_sharpe(sr, sr0, len(r))
    return probabilistic_sharpe(sr, sr0, len(r), float(skew(r)), float(kurtosis(r, fisher=False)))


def max_drawdown(pnl: np.ndarray) -> float:
    """Perte maximale depuis un sommet de la courbe cumulée (valeur positive)."""
    eq = np.cumsum(np.asarray(pnl, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float(np.max(peak - eq)) if len(eq) else 0.0


def pbo_cscv(returns: np.ndarray, n_blocks: int = 16, max_combos: int | None = None,
             rng: np.random.Generator | None = None) -> tuple[float, np.ndarray]:
    """Probabilité de surapprentissage par validation croisée combinatoire symétrique (CSCV).

    returns : matrice (T, M), une colonne par variante essayée.
    Pour chaque façon de choisir n_blocks/2 blocs « dans l'échantillon » : on retient la variante de
    meilleur Sharpe dans l'échantillon, on calcule son rang relatif w hors échantillon et
    logit = ln(w / (1 - w)). PBO = proportion des logits <= 0 (meilleure variante IS sous la médiane OOS).
    """
    R = np.asarray(returns, dtype=float)
    T, M = R.shape
    L = T // n_blocks
    R = R[: L * n_blocks].reshape(n_blocks, L, M)
    s1 = R.sum(axis=1)           # (blocs, M)
    s2 = (R * R).sum(axis=1)
    combos = list(combinations(range(n_blocks), n_blocks // 2))
    if max_combos is not None and len(combos) > max_combos:
        rng = rng or np.random.default_rng(0)
        combos = [combos[i] for i in rng.choice(len(combos), max_combos, replace=False)]
    full = set(range(n_blocks))
    n_half = (n_blocks // 2) * L

    def sr_of(blocks):
        idx = list(blocks)
        m = s1[idx].sum(axis=0) / n_half
        v = s2[idx].sum(axis=0) / n_half - m * m
        return m / np.sqrt(np.maximum(v, 1e-18))

    logits = []
    for c in combos:
        is_sr = sr_of(c)
        oos_sr = sr_of(sorted(full - set(c)))
        best = int(np.argmax(is_sr))
        # rang moyen en cas d'égalité (variantes qui ne traitent jamais, par exemple)
        rank = (oos_sr < oos_sr[best]).sum() + 0.5 * ((oos_sr == oos_sr[best]).sum() - 1) + 1
        w = rank / (M + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return float(np.mean(logits <= 0)), logits


def n_cscv_combinations(n_blocks: int) -> int:
    return comb(n_blocks, n_blocks // 2)
