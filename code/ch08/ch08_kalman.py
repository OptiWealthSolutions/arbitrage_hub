"""Chapitre 8 — Modèles espace-état et filtre de Kalman.

E1  Ratio de couverture variable (marche aléatoire) : Kalman vs MCO glissants.
E2  Réglage du bruit d'état delta : erreur de suivi et log-vraisemblance des innovations.
E3  Stratégie fondée sur les innovations de Kalman vs z-score à beta statique (coûts 5 pb).
E4  Données réelles : beta de Kalman pour log KO / log PEP et log GLD / log GLDM.
E5  Cointégration partielle : estimation de (rho, sigma_M, sigma_R) par maximum de vraisemblance de Kalman.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, load_close, ar1, r3

ch = Chapter("ch08")


def kalman_regression(y, x, delta=1e-4, r=1e-3, m0=None, P0=None):
    """Régression à coefficients variables : y_t = a_t + b_t x_t + v_t, (a_t, b_t) marche aléatoire.

    Q = delta/(1-delta) * I (paramétrisation de Chan), R = r.
    Renvoie états filtrés (T x 2), innovations e_t, variances d'innovation F_t, log-vraisemblance.
    """
    T = len(y)
    Q = delta / (1 - delta) * np.eye(2)
    m = np.zeros(2) if m0 is None else np.array(m0, float)
    P = np.eye(2) if P0 is None else P0
    states = np.zeros((T, 2)); e = np.zeros(T); F = np.zeros(T); ll = 0.0
    for t in range(T):
        P = P + Q                                    # prédiction (marche aléatoire : m inchangé)
        H = np.array([1.0, x[t]])
        yhat = H @ m
        F[t] = H @ P @ H + r
        e[t] = y[t] - yhat
        K = P @ H / F[t]                             # gain
        m = m + K * e[t]                             # mise à jour
        P = P - np.outer(K, H @ P)
        states[t] = m
        ll += -0.5 * (np.log(2 * np.pi * F[t]) + e[t] ** 2 / F[t])
    return states, e, F, ll


def rolling_beta(y, x, w):
    b = np.full(len(y), np.nan)
    for t in range(w, len(y)):
        b[t] = np.polyfit(x[t - w:t], y[t - w:t], 1)[0]
    return b


def simulate_dynamic_pair(rng, n=1500, beta_vol=0.003, noise=0.3):
    """Prix (pas de log) : x marche aléatoire autour de 50, y = 5 + beta_t x + s_t, beta_t marche aléatoire."""
    x = 50 + np.cumsum(rng.normal(0, 1.0, n))
    x = np.maximum(x, 5.0)
    beta = 1.0 + np.cumsum(rng.normal(0, beta_vol, n))
    s = ar1(rng, n - 1, 0.9, noise)[0]
    y = 5.0 + beta * x + s
    return y, x, beta


# ---------------------------------------------------------------- E1 / E2
def e1_e2():
    rng = np.random.default_rng(81)
    y, x, beta = simulate_dynamic_pair(rng)
    burn = 250
    res = {}
    for d in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        st, e, F, ll = kalman_regression(y, x, delta=d, r=0.5, m0=[0, 1])
        res[f"delta={d:g}"] = {"rmse_beta": r3(np.sqrt(np.mean((st[burn:, 1] - beta[burn:]) ** 2)), 4),
                               "log_vraisemblance": r3(ll, 1)}
    roll = {}
    for w in (60, 250):
        b = rolling_beta(y, x, w)
        roll[f"MCO_glissant_{w}"] = r3(np.sqrt(np.nanmean((b[burn:] - beta[burn:]) ** 2)), 4)
    b_static = np.polyfit(x[:burn], y[:burn], 1)[0]
    roll["MCO_statique_formation"] = r3(np.sqrt(np.mean((b_static - beta[burn:]) ** 2)), 4)
    ch.record("E1_E2", {"kalman": res, "comparaison": roll})

    best = max(res, key=lambda k: res[k]["log_vraisemblance"])
    dbest = float(best.split("=")[1])
    st_ll, *_ = kalman_regression(y, x, delta=dbest, r=0.5, m0=[0, 1])
    st_lo, *_ = kalman_regression(y, x, delta=1e-7, r=0.5, m0=[0, 1])
    b60 = rolling_beta(y, x, 60)
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ax.plot(b60, color=MUTED, lw=0.6, label="MCO glissants 60 j")
    ax.plot(st_ll[:, 1], color=GOLD, lw=0.8, label=f"Kalman, $\\delta$ = {dbest:g} (max. vraisemblance)")
    ax.plot(st_lo[:, 1], color=ACCENT, lw=1.1, label="Kalman, $\\delta$ = 1e-07")
    ax.plot(beta, color="k", lw=1.3, label="vrai $\\beta_t$")
    ax.set_ylim(beta.min() - 0.25, beta.max() + 0.25)
    ax.set_xlabel("jour")
    ax.set_title("Suivi d'un ratio de couverture qui dérive")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ch.save_fig(fig, "kalman_beta")


# ---------------------------------------------------------------- E3
def trade_signal(spread_signal, scale, legs_notional, dspread, cost_bps, entry=1.0):
    """Ouvre quand |signal| > entry*scale, ferme au changement de signe. P&L = pos(t-1) * d spread(t)."""
    pos, pnl, ntr = 0.0, 0.0, 0
    for t in range(1, len(spread_signal)):
        pnl += pos * dspread[t]
        z = spread_signal[t] / scale[t]
        new = pos
        if pos == 0:
            if z > entry: new = -1.0
            elif z < -entry: new = 1.0
        elif (pos == 1 and z >= 0) or (pos == -1 and z <= 0):
            new = 0.0
        if new != pos:
            pnl -= abs(new - pos) * legs_notional[t] * cost_bps / 1e4
            ntr += abs(new) > 0
            pos = new
    return pnl, ntr


DELTA_E3 = 1e-6


def e3():
    rng = np.random.default_rng(83)
    out = {"kalman": [], "statique": []}
    for _ in range(100):
        y, x, beta = simulate_dynamic_pair(rng, n=1000)
        form = 250
        st, e, F, _ = kalman_regression(y, x, delta=DELTA_E3, r=0.5, m0=[0, 1])
        # écart Kalman : on garde le beta de la veille pour le P&L (pas d'anticipation)
        b_prev = np.r_[st[0, 1], st[:-1, 1]]
        a_prev = np.r_[st[0, 0], st[:-1, 0]]
        dspread_k = np.diff(y, prepend=y[0]) - b_prev * np.diff(x, prepend=x[0])
        notional = y + np.abs(b_prev) * x
        pk, _ = trade_signal(e[form:], np.sqrt(F[form:]), notional[form:], dspread_k[form:], 5.0, entry=1.0)
        # statique : MCO sur la formation, z-score à 2
        b0, a0 = np.polyfit(x[:form], y[:form], 1)
        s = y - a0 - b0 * x
        m, d = s[:form].mean(), s[:form].std()
        ps, _ = trade_signal(s[form:] - m, np.full(len(s) - form, d), (y + abs(b0) * x)[form:],
                             np.diff(s, prepend=s[0])[form:], 5.0, entry=2.0)
        gross0 = y[form] + abs(b0) * x[form]
        out["kalman"].append(pk / gross0 / 3)          # 750 jours = 3 ans -> rendement annuel sur le notionnel brut
        out["statique"].append(ps / gross0 / 3)
    ch.record("E3", {k: {"moyenne_annuelle_pct": r3(100 * np.mean(v), 2), "ecart_type_pct": r3(100 * np.std(v), 2),
                         "part_positive": r3(np.mean(np.array(v) > 0), 3)} for k, v in out.items()})


# ---------------------------------------------------------------- E4
def e4():
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for ax, (a, b) in zip(axes, [("KO", "PEP"), ("GLD", "GLDM")]):
        y, x = np.log(load_close(a)), np.log(load_close(b))
        st, e, F, ll = kalman_regression(y, x, delta=1e-5, r=1e-4, m0=[0, 1])
        out[f"{a}_{b}"] = {"beta_debut": r3(st[30, 1], 3), "beta_milieu": r3(st[250, 1], 3), "beta_fin": r3(st[-1, 1], 3),
                           "beta_min": r3(st[30:, 1].min(), 3), "beta_max": r3(st[30:, 1].max(), 3)}
        ax.plot(st[:, 1], color=INK, lw=1)
        ax.set_title(f"$\\beta_t$ de Kalman : log {a} sur log {b}")
        ax.set_xlabel("séance")
    fig.tight_layout()
    ch.save_fig(fig, "kalman_reel")
    ch.record("E4", out)


# ---------------------------------------------------------------- E5
def partial_coint_negll(params, s):
    rho = np.tanh(params[0]); sm, sr = np.exp(params[1]), np.exp(params[2])
    Fm = np.array([[rho, 0], [0, 1.0]]); Q = np.diag([sm**2, sr**2]); H = np.array([1.0, 1.0])
    m = np.array([0.0, s[0]]); P = np.diag([sm**2 / (1 - rho**2), 1e-8])
    ll = 0.0
    for t in range(1, len(s)):
        m = Fm @ m; P = Fm @ P @ Fm.T + Q
        f = H @ P @ H + 1e-10
        v = s[t] - H @ m
        K = P @ H / f
        m = m + K * v; P = P - np.outer(K, H @ P)
        ll += -0.5 * (np.log(2 * np.pi * f) + v**2 / f)
    return -ll


def e5():
    rng = np.random.default_rng(85)
    truth = (0.8, 1.0, 0.4)
    est = []
    for _ in range(25):
        n = 1000
        mm = ar1(rng, n - 1, truth[0], truth[1])[0]
        rr = np.cumsum(rng.normal(0, truth[2], n))
        s = mm + rr
        best = None
        for start in ([0.5, 0.0, -1.0], [1.5, -0.5, -0.5]):
            r = minimize(partial_coint_negll, start, args=(s,), method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-4})
            if best is None or r.fun < best.fun:
                best = r
        est.append((np.tanh(best.x[0]), np.exp(best.x[1]), np.exp(best.x[2])))
    est = np.array(est)
    r2 = lambda rho, sm, sr: 2 * sm**2 / (2 * sm**2 + (1 + rho) * sr**2)
    ch.record("E5", {"vrai": {"rho": truth[0], "sigma_M": truth[1], "sigma_R": truth[2], "R2_MR": r3(r2(*truth), 3)},
                     "mediane": {"rho": r3(np.median(est[:, 0]), 3), "sigma_M": r3(np.median(est[:, 1]), 3),
                                 "sigma_R": r3(np.median(est[:, 2]), 3),
                                 "R2_MR": r3(np.median([r2(*e) for e in est]), 3)},
                     "q05_q95_rho": [r3(np.quantile(est[:, 0], 0.05), 3), r3(np.quantile(est[:, 0], 0.95), 3)]})


if __name__ == "__main__":
    e1_e2(); e3(); e4(); e5()
    ch.save_results()
