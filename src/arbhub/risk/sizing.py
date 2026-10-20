"""Dimensionnement et détection de rupture (chapitre 17).

Références : Kelly (1956) ; Thorp (2006) pour la version continue f* = mu / sigma^2 ;
Page (1954) pour le CUSUM.
"""
from __future__ import annotations

import numpy as np


def kelly_fraction(mu: float, sigma: float) -> float:
    """Levier de Kelly continu f* = mu / sigma^2 (rendement espéré et volatilité par période).
    Maximise la croissance logarithmique espérée g(f) = f mu - f^2 sigma^2 / 2."""
    return float(mu / sigma**2)


def log_growth(f: float, mu: float, sigma: float) -> float:
    """Taux de croissance logarithmique espéré par période avec un levier f (approximation continue)."""
    return float(f * mu - 0.5 * f * f * sigma**2)


def vol_target_weights(pnl_unit: np.ndarray, target_vol: float, window: int = 60,
                       max_lev: float = 5.0) -> np.ndarray:
    """Taille w[t] (multiplicateur) pour que la volatilité attendue du P&L vise target_vol.

    w[t] = target_vol / vol estimée sur pnl_unit[t-window+1..t] ; appliquée à pnl_unit[t+1].
    Renvoie NaN -> 0 tant que la fenêtre n'est pas remplie. N'utilise que le passé.
    """
    x = np.asarray(pnl_unit, dtype=float)
    w = np.zeros(len(x))
    for t in range(window - 1, len(x)):
        sd = x[t - window + 1: t + 1].std(ddof=1)
        w[t] = min(target_vol / sd, max_lev) if sd > 0 else 0.0
    return w


def cusum_alarm(x: np.ndarray, mu0: float, sd: float, k: float = 0.5, h: float = 5.0) -> int | None:
    """CUSUM bilatéral de Page sur la série standardisée (x - mu0) / sd.

    S+ = max(0, S+ + z - k), S- = max(0, S- - z - k) ; alarme dès que S+ > h ou S- > h.
    Renvoie l'indice de la première alarme, ou None.
    """
    sp = sm = 0.0
    for t, v in enumerate(np.asarray(x, dtype=float)):
        z = (v - mu0) / sd
        sp = max(0.0, sp + z - k)
        sm = max(0.0, sm - z - k)
        if sp > h or sm > h:
            return t
    return None


def rolling_ar1_phi(s: np.ndarray, window: int) -> np.ndarray:
    """phi estimé par MCO de s_t sur s_{t-1} sur la fenêtre [t-window+1, t] ; NaN avant."""
    s = np.asarray(s, dtype=float)
    out = np.full(len(s), np.nan)
    for t in range(window, len(s)):
        a = s[t - window: t]
        b = s[t - window + 1: t + 1]
        am = a - a.mean()
        out[t] = float(np.dot(am, b - b.mean()) / np.dot(am, am))
    return out
