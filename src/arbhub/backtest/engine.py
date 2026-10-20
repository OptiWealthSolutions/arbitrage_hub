"""Moteur de backtest vectorisé pour une paire (chapitre 16).

Conventions (voir CLAUDE.md) :
- la position q[t] est décidée à la clôture de t avec l'information disponible en t ;
- elle s'applique à la variation de t à t+1 : gain[t+1] = q[t] * (y[t+1] - beta[t] * x[t+1] - (y[t] - beta[t] * x[t])) ;
- les coûts sont payés en t sur le notionnel échangé : |Δ unités de y| * y[t] + |Δ unités de x| * x[t],
  multiplié par le coût en points de base par jambe.

Référence : chapitre 16, section « Un moteur sans anticipation ».
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def zscore_positions(z: np.ndarray, entry: float = 2.0, exit: float = 0.0,
                     stop: float | None = None) -> np.ndarray:
    """Règle à hystérésis : vendre l'écart si z > entry, l'acheter si z < -entry,
    sortir quand z repasse exit (en direction de la moyenne), stop-loss si |z| > stop.

    Entrée : z (n,), éventuellement avec des NaN au début (aucune position tant que z est NaN).
    Sortie : q (n,) dans {-1, 0, +1}, q[t] ne dépend que de z[0..t].
    """
    z = np.asarray(z, dtype=float)
    q = np.zeros(len(z))
    for t in range(len(z)):
        prev = q[t - 1] if t > 0 else 0.0
        zt = z[t]
        if np.isnan(zt):
            q[t] = 0.0
            continue
        if stop is not None and abs(zt) > stop:
            q[t] = 0.0
        elif prev == 0.0:
            q[t] = -1.0 if zt > entry else (1.0 if zt < -entry else 0.0)
        elif prev > 0 and zt >= -exit:
            q[t] = 0.0
        elif prev < 0 and zt <= exit:
            q[t] = 0.0
        else:
            q[t] = prev
    return q


def rolling_zscore(s: np.ndarray, window: int) -> np.ndarray:
    """z[t] = (s[t] - moyenne(s[t-window+1..t])) / écart-type(idem). NaN avant la première fenêtre.
    N'utilise que des données datées <= t."""
    s = np.asarray(s, dtype=float)
    z = np.full(len(s), np.nan)
    if len(s) < window:
        return z
    c1 = np.cumsum(np.insert(s, 0, 0.0))
    c2 = np.cumsum(np.insert(s * s, 0, 0.0))
    m = (c1[window:] - c1[:-window]) / window
    v = (c2[window:] - c2[:-window]) / window - m * m
    v = v * window / (window - 1)
    sd = np.sqrt(np.maximum(v, 1e-18))
    z[window - 1:] = (s[window - 1:] - m) / sd
    return z


@dataclass
class PairResult:
    gross: np.ndarray      # gain brut par date (en devise)
    costs: np.ndarray      # coûts par date
    net: np.ndarray        # gain net par date
    units_y: np.ndarray    # unités de y détenues après la décision de t
    units_x: np.ndarray    # unités de x détenues après la décision de t
    notional: np.ndarray   # notionnel brut engagé après la décision de t
    n_trades: int          # nombre de changements de position

    @property
    def pnl(self) -> float:
        return float(self.net.sum())


def pair_backtest(y: np.ndarray, x: np.ndarray, beta, q: np.ndarray,
                  cost_bps: float = 0.0) -> PairResult:
    """P&L d'une position q (en unités d'écart y - beta x) sans anticipation.

    beta : scalaire ou tableau (n,) ; beta[t] est le ratio choisi à la clôture de t.
    cost_bps : coût par jambe en points de base du notionnel échangé.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    n = len(y)
    b = np.full(n, float(beta)) if np.isscalar(beta) else np.asarray(beta, dtype=float)
    uy = q.copy()
    ux = -q * b
    gross = np.zeros(n)
    gross[1:] = uy[:-1] * np.diff(y) + ux[:-1] * np.diff(x)
    duy = np.abs(np.diff(np.concatenate([[0.0], uy])))
    dux = np.abs(np.diff(np.concatenate([[0.0], ux])))
    costs = (duy * y + dux * x) * cost_bps / 1e4
    notional = np.abs(uy) * y + np.abs(ux) * x
    n_trades = int(np.count_nonzero(np.diff(np.concatenate([[0.0], q]))))
    return PairResult(gross, costs, gross - costs, uy, ux, notional, n_trades)
