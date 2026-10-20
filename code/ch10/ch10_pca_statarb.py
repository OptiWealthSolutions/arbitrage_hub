"""Chapitre 10 — Arbitrage statistique multi-actifs (Avellaneda et Lee, 2010).

Univers simulé : N = 100 actions, 3 facteurs, résidus cumulés de type Ornstein-Uhlenbeck
(scénario « signal ») ou résidus en bruit blanc (scénario « nul »).

E1  Spectre des valeurs propres de la matrice de corrélation vs borne de Marchenko-Pastur.
E2  Stratégie s-score d'Avellaneda-Lee : ACP (k facteurs) sur 252 jours, régression des rendements
    sur les portefeuilles propres sur 60 jours, résidu cumulé -> AR(1) -> s-score, règles
    (ouvrir à |s| > 1,25 ; fermer court à s < 0,75, long à s > -0,5), filtre de vitesse.
E3  Sensibilité au nombre de facteurs et aux coûts ; scénario nul.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib.pyplot as plt

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, r3

ch = Chapter("ch10")
N, W_PCA, W_REG = 100, 252, 60


def simulate(rng, T=1252, signal=True, half_life=8.0, sd_eq=0.03):
    sig_f = np.array([0.010, 0.006, 0.005])
    F = rng.normal(0, 1, (T, 3)) * sig_f
    B = np.column_stack([rng.normal(1, 0.3, N), rng.normal(0, 1, N), rng.normal(0, 1, N)])
    if signal:
        b = np.exp(-np.log(2) / half_life)
        X = np.zeros((T + 1, N))
        z = rng.normal(0, sd_eq * np.sqrt(1 - b**2), (T, N))
        for t in range(T):
            X[t + 1] = b * X[t] + z[t]
        idio = np.diff(X, axis=0) + rng.normal(0, 0.005, (T, N))
    else:
        b = np.exp(-np.log(2) / half_life)
        sd = np.sqrt(2 * sd_eq**2 * (1 - b) + 0.005**2)   # même variance quotidienne que le scénario signal
        idio = rng.normal(0, sd, (T, N))
    R = F @ B.T + idio
    return R


def marchenko_pastur_edge(n, t):
    q = n / t
    return (1 + np.sqrt(q)) ** 2


def eigen_portfolios(Rw, k):
    mu, sd = Rw.mean(0), Rw.std(0)
    Z = (Rw - mu) / sd
    C = Z.T @ Z / len(Z)
    val, vec = np.linalg.eigh(C)
    order = np.argsort(val)[::-1]
    val, vec = val[order], vec[:, order]
    Q = vec[:, :k] / sd[:, None]          # poids des portefeuilles propres
    return val, Q


def s_scores(Rreg, Freg):
    """Régression des rendements (60 x N) sur les facteurs (60 x k) ; résidus cumulés -> AR(1) -> s-score."""
    X = np.column_stack([np.ones(len(Freg)), Freg])
    coef, *_ = np.linalg.lstsq(X, Rreg, rcond=None)
    resid = Rreg - X @ coef
    Xc = np.cumsum(resid, axis=0)
    x0, x1 = Xc[:-1], Xc[1:]
    m0, m1 = x0.mean(0), x1.mean(0)
    b = ((x0 - m0) * (x1 - m1)).sum(0) / ((x0 - m0) ** 2).sum(0)
    a = m1 - b * m0
    zeta = x1 - a - b * x0
    b = np.clip(b, 1e-6, 0.999999)
    kappa = -np.log(b) * 252
    m = a / (1 - b)
    sig_eq = np.sqrt(zeta.var(0) / (1 - b**2))
    s = (Xc[-1] - m) / sig_eq
    return s, kappa, coef[1:]


def backtest(R, k=3, cost_bps=5.0, kappa_min=252 / 30, open_=1.25, close_short=0.75, close_long=-0.5):
    T = R.shape[0]
    pos = np.zeros(N)
    beta = np.zeros((k, N))
    daily = []
    for t in range(W_PCA, T - 1):
        if (t - W_PCA) % 1 == 0:
            _, Q = eigen_portfolios(R[t - W_PCA:t], k)
        Ft = R[:t + 1] @ Q                                  # rendements des portefeuilles propres
        s, kappa, coef = s_scores(R[t - W_REG + 1:t + 1], Ft[t - W_REG + 1:t + 1])
        new = pos.copy()
        ok = kappa > kappa_min
        new[(pos == 0) & ok & (s > open_)] = -1
        new[(pos == 0) & ok & (s < -open_)] = 1
        new[(pos == -1) & (s < close_short)] = 0
        new[(pos == 1) & (s > close_long)] = 0
        trade_cost = np.abs(new - pos) * (1 + np.abs(coef).sum(0)) * cost_bps / 1e4
        pos, beta = new, coef
        # rendement du lendemain : action moins couverture factorielle
        f_next = R[t + 1] @ Q
        hedged = R[t + 1] - f_next @ beta
        daily.append((pos * hedged).sum() / N - trade_cost.sum() / N)
    d = np.array(daily)
    return {"rendement_annuel_pct": r3(100 * 252 * d.mean(), 2), "volatilite_annuelle_pct": r3(100 * np.sqrt(252) * d.std(), 2),
            "sharpe": r3(np.sqrt(252) * d.mean() / d.std(), 2)}, d


def e1():
    rng = np.random.default_rng(101)
    R = simulate(rng)
    val, _ = eigen_portfolios(R[:W_PCA], 3)
    edge = marchenko_pastur_edge(N, W_PCA)
    ch.record("E1", {"5_plus_grandes_valeurs_propres": [r3(v, 2) for v in val[:5]], "borne_MP": r3(edge, 3),
                     "nb_au_dessus_borne": int((val > edge).sum()), "part_variance_3_facteurs": r3(val[:3].sum() / N, 3)})
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ax.bar(np.arange(1, 21), val[:20], color=[ACCENT if v > edge else MUTED for v in val[:20]])
    ax.axhline(edge, color=INK, ls="--", lw=1, label=f"borne de Marchenko-Pastur ({edge:.2f})")
    ax.set_xlabel("rang de la valeur propre")
    ax.set_ylabel("valeur propre")
    ax.set_title("Spectre de la matrice de corrélation (N = 100, T = 252)")
    ax.legend(frameon=False, fontsize=7)
    ch.save_fig(fig, "spectre_valeurs_propres")


def e2_e3():
    out = {}
    curves = {}
    for scen in ("signal", "nul"):
        rng = np.random.default_rng(102)
        R = simulate(rng, signal=(scen == "signal"))
        for k in (1, 3, 10):
            for c in (0.0, 5.0, 10.0):
                if scen == "nul" and (k != 3):
                    continue
                res, d = backtest(R, k=k, cost_bps=c)
                out[f"{scen}_k={k}_cout={c:g}pb"] = res
                if k == 3 and c == 5.0:
                    curves[scen] = np.cumsum(d)
    ch.record("E2_E3", out)
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ax.plot(100 * curves["signal"], color=GREEN, label="résidus OU (signal)")
    ax.plot(100 * curves["nul"], color=ACCENT, label="résidus bruit blanc (nul)")
    ax.set_xlabel("jour de trading")
    ax.set_ylabel("P&L cumulé (%)")
    ax.set_title("s-score, 3 facteurs, coûts 5 pb")
    ax.legend(frameon=False, fontsize=7)
    ch.save_fig(fig, "pnl_sscore")


if __name__ == "__main__":
    e1(); e2_e3()
    ch.save_results()
