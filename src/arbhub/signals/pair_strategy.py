"""Stratégie de paire en z-score, en deux implémentations qui doivent coïncider (chapitre 18).

- `vectorized_positions` : version de recherche, calcule toutes les positions d'un coup ;
- `PairStrategy.on_bar`   : version « production », reçoit les prix un par un comme un robot.

Le test de parité (tests/test_pair_strategy.py) vérifie qu'elles donnent exactement les mêmes
positions et les mêmes ratios de couverture : c'est la garantie que le code exécuté en paper
trading est celui qui a été backtesté.

Modèle : log y_t = alpha_t + beta_t log x_t + s_t, avec (alpha_t, beta_t) estimés par MCO sur les
`beta_window` dernières séances (passé strict + séance courante), s_t le résidu courant, et z_t
le z-score de s sur les `z_window` derniers résidus calculés avec ce même (alpha_t, beta_t).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PairConfig:
    beta_window: int = 120
    z_window: int = 60
    entry: float = 2.0
    exit: float = 0.0
    stop: float | None = 4.0

    def __post_init__(self):
        if self.z_window > self.beta_window:
            raise ValueError("z_window doit être <= beta_window")
        if not 0 <= self.exit < self.entry:
            raise ValueError("il faut 0 <= exit < entry")
        if self.stop is not None and self.stop <= self.entry:
            raise ValueError("stop doit être > entry")


def _fit(ly: np.ndarray, lx: np.ndarray) -> tuple[float, float]:
    xm, ym = lx.mean(), ly.mean()
    b = float(np.dot(lx - xm, ly - ym) / np.dot(lx - xm, lx - xm))
    return b, float(ym - b * xm)


def _zscore(ly_w: np.ndarray, lx_w: np.ndarray, b: float, a: float, z_window: int) -> float:
    s = ly_w[-z_window:] - a - b * lx_w[-z_window:]
    sd = s.std(ddof=1)
    return float((s[-1] - s.mean()) / sd) if sd > 0 else 0.0


def _step(prev: float, z: float, cfg: PairConfig) -> float:
    if cfg.stop is not None and abs(z) > cfg.stop:
        return 0.0
    if prev == 0.0:
        return -1.0 if z > cfg.entry else (1.0 if z < -cfg.entry else 0.0)
    if prev > 0 and z >= -cfg.exit:
        return 0.0
    if prev < 0 and z <= cfg.exit:
        return 0.0
    return prev


def vectorized_positions(y: np.ndarray, x: np.ndarray, cfg: PairConfig):
    """Positions q[t] (unités de log-écart) et ratios beta[t] décidés à la clôture de t.
    Renvoie (q, beta, z) ; q = 0 et beta = NaN tant que la fenêtre n'est pas remplie."""
    ly, lx = np.log(np.asarray(y, float)), np.log(np.asarray(x, float))
    n = len(ly)
    q, beta, z = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
    W = cfg.beta_window
    for t in range(W - 1, n):
        b, a = _fit(ly[t - W + 1: t + 1], lx[t - W + 1: t + 1])
        beta[t] = b
        z[t] = _zscore(ly[t - W + 1: t + 1], lx[t - W + 1: t + 1], b, a, cfg.z_window)
        q[t] = _step(q[t - 1] if t > 0 else 0.0, z[t], cfg)
    return q, beta, z


@dataclass
class PairStrategy:
    """Version événementielle : un appel à on_bar par clôture."""
    cfg: PairConfig
    _ly: deque = field(init=False)
    _lx: deque = field(init=False)
    position: float = field(init=False, default=0.0)

    def __post_init__(self):
        self._ly = deque(maxlen=self.cfg.beta_window)
        self._lx = deque(maxlen=self.cfg.beta_window)

    def on_bar(self, y_close: float, x_close: float) -> tuple[float, float, float]:
        """Reçoit la clôture du jour ; renvoie (position cible, beta, z). N'a jamais accès au futur."""
        if y_close <= 0 or x_close <= 0:
            raise ValueError("prix non positif")
        self._ly.append(np.log(y_close))
        self._lx.append(np.log(x_close))
        if len(self._ly) < self.cfg.beta_window:
            return 0.0, float("nan"), float("nan")
        ly, lx = np.array(self._ly), np.array(self._lx)
        b, a = _fit(ly, lx)
        z = _zscore(ly, lx, b, a, self.cfg.z_window)
        self.position = _step(self.position, z, self.cfg)
        return self.position, b, z


def log_spread_pnl(y: np.ndarray, x: np.ndarray, q: np.ndarray, beta: np.ndarray,
                   cost_bps: float = 0.0, capital: float = 1.0) -> np.ndarray:
    """Gain net quotidien (en fraction du capital) d'une position en log-écart :
    on détient q * capital en y et -q * beta * capital en x (montants), rééquilibrés chaque jour.
    gain[t+1] = q[t] * (r_y[t+1] - beta[t] * r_x[t+1]) ; coût = |Δ montants| * cost_bps."""
    ry = np.diff(np.log(y))
    rx = np.diff(np.log(x))
    b = np.nan_to_num(beta)
    my, mx = q * capital, -q * b * capital
    gain = np.zeros(len(y))
    gain[1:] = my[:-1] * ry + mx[:-1] * rx
    turn = np.abs(np.diff(np.concatenate([[0.0], my]))) + np.abs(np.diff(np.concatenate([[0.0], mx])))
    return gain - turn * cost_bps / 1e4
