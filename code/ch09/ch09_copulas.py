"""Chapitre 9 — Copules.

E1  Même tau de Kendall (0,5), trois copules : probabilités de co-extrêmes très différentes.
E2  Estimation par inversion du tau de Kendall et choix de copule par AIC (Gaussienne, Student, Clayton).
E3  Signal de « mispricing » cumulé (dans l'esprit de Xie et al., 2016) contre z-score,
    sur des paires cointégrées à chocs de Student.
E4  Données réelles KO/PEP : tau, copule retenue, co-extrêmes observés contre prédits.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, load_close, r3

ch = Chapter("ch09")
NU = 4.0


# ------------------------------------------------------------ simulation de copules
def sim_gauss(rng, n, rho):
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    return stats.norm.cdf(z)


def sim_t(rng, n, rho, nu=NU):
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    w = rng.chisquare(nu, n) / nu
    return stats.t.cdf(z / np.sqrt(w)[:, None], nu)


def sim_clayton(rng, n, theta):
    v = rng.gamma(1 / theta, 1, n)
    e = rng.exponential(size=(n, 2))
    return (1 + e / v[:, None]) ** (-1 / theta)


# ------------------------------------------------------------ densités (log)
def ll_gauss(u, v, rho):
    a, b = stats.norm.ppf(u), stats.norm.ppf(v)
    return np.sum(-0.5 * np.log(1 - rho**2) - (rho**2 * (a**2 + b**2) - 2 * rho * a * b) / (2 * (1 - rho**2)))


def ll_t(u, v, rho, nu=NU):
    a, b = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
    joint = stats.multivariate_t(loc=[0, 0], shape=[[1, rho], [rho, 1]], df=nu).logpdf(np.column_stack([a, b]))
    return np.sum(joint - stats.t.logpdf(a, nu) - stats.t.logpdf(b, nu))


def ll_clayton(u, v, th):
    return np.sum(np.log(1 + th) + (-th - 1) * (np.log(u) + np.log(v)) + (-2 - 1 / th) * np.log(u**-th + v**-th - 1))


def fit_all(u, v):
    tau = stats.kendalltau(u, v)[0]
    rho = np.sin(np.pi * tau / 2)
    th = max(2 * tau / (1 - tau), 1e-3)
    out = {"gaussienne": (rho, ll_gauss(u, v, rho)), "student": (rho, ll_t(u, v, rho)), "clayton": (th, ll_clayton(u, v, th))}
    aic = {k: 2 * 1 - 2 * vv[1] for k, vv in out.items()}
    return tau, out, min(aic, key=aic.get)


# ------------------------------------------------------------ h-fonctions : P(U <= u | V = v)
def h_gauss(u, v, rho):
    return stats.norm.cdf((stats.norm.ppf(u) - rho * stats.norm.ppf(v)) / np.sqrt(1 - rho**2))


def h_t(u, v, rho, nu=NU):
    a, b = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
    return stats.t.cdf((a - rho * b) / np.sqrt((nu + b**2) * (1 - rho**2) / (nu + 1)), nu + 1)


def h_clayton(u, v, th):
    return v ** (-th - 1) * (u**-th + v**-th - 1) ** (-1 / th - 1)


def pseudo_obs(x):
    return stats.rankdata(x) / (len(x) + 1)


# ---------------------------------------------------------------- E1
def e1():
    rng = np.random.default_rng(91)
    tau, n = 0.5, 400_000
    rho = np.sin(np.pi * tau / 2); th = 2 * tau / (1 - tau)
    sims = {"gaussienne": sim_gauss(rng, n, rho), "student_nu4": sim_t(rng, n, rho), "clayton": sim_clayton(rng, n, th)}
    out = {}
    for k, uv in sims.items():
        u, v = uv[:, 0], uv[:, 1]
        out[k] = {"tau_mesure": r3(stats.kendalltau(u[:20000], v[:20000])[0], 3),
                  "P(U<1%,V<1%)_en_pct": r3(100 * np.mean((u < 0.01) & (v < 0.01)), 3),
                  "P(U>99%,V>99%)_en_pct": r3(100 * np.mean((u > 0.99) & (v > 0.99)), 3),
                  "P(V<1%|U<1%)": r3(np.mean(v[u < 0.01] < 0.01), 3)}
    out["independance_P(U<1%,V<1%)_pct"] = 0.01
    out["dependance_de_queue_theorique"] = {"gaussienne": 0.0, "clayton_basse": r3(2 ** (-1 / th), 3),
                                            "student_nu4": r3(2 * stats.t.cdf(-np.sqrt((NU + 1) * (1 - rho) / (1 + rho)), NU + 1), 3)}
    ch.record("E1", out)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, (k, uv) in zip(axes, sims.items()):
        z = stats.norm.ppf(uv[:3000])
        ax.scatter(z[:, 0], z[:, 1], s=1.5, color=INK, alpha=0.5)
        ax.set_title({"gaussienne": "Gaussienne", "student_nu4": "Student (ν = 4)", "clayton": "Clayton"}[k])
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
        ax.set_xlabel("marge 1 (normale)")
    axes[0].set_ylabel("marge 2 (normale)")
    fig.suptitle("Même tau de Kendall (0,5), mêmes marges normales, trois copules", fontsize=9)
    fig.tight_layout()
    ch.save_fig(fig, "trois_copules")


# ---------------------------------------------------------------- E2
def e2():
    rng = np.random.default_rng(92)
    res = {}
    for name, gen, par in [("gaussienne", sim_gauss, 0.7071), ("student", sim_t, 0.7071), ("clayton", sim_clayton, 2.0)]:
        picks, ests = {"gaussienne": 0, "student": 0, "clayton": 0}, []
        for _ in range(100):
            uv = gen(rng, 500, par)
            u, v = pseudo_obs(uv[:, 0]), pseudo_obs(uv[:, 1])
            tau, fit, best = fit_all(u, v)
            picks[best] += 1
            ests.append(fit["clayton" if name == "clayton" else "gaussienne"][0])
        res[name] = {"choix_AIC_sur_100": picks, "parametre_moyen": r3(np.mean(ests), 3), "vrai_parametre": par}
    ch.record("E2", res)


# ---------------------------------------------------------------- E3
def make_pair(rng, n, phi=0.9):
    """Paire cointégrée en log-prix ; chocs de Student (nu=3) pour des queues épaisses."""
    common = np.cumsum(0.01 * rng.standard_t(3, n) / np.sqrt(3))
    gap = np.zeros(n)
    eps = 0.006 * rng.standard_t(3, n) / np.sqrt(3)
    for t in range(1, n):
        gap[t] = phi * gap[t - 1] + eps[t]
    x = np.log(50) + common - gap / 2
    y = np.log(50) + common + gap / 2
    return y, x


def zscore_strategy(y, x, form, cost):
    b, a = np.polyfit(x[:form], y[:form], 1)
    s = y - a - b * x
    m, d = s[:form].mean(), s[:form].std()
    z = (s - m) / d
    pos, pnl = 0.0, 0.0
    for t in range(form + 1, len(s)):
        pnl += pos * (s[t] - s[t - 1])
        new = pos
        if pos == 0:
            new = -1.0 if z[t] > 2 else (1.0 if z[t] < -2 else 0.0)
        elif pos * z[t] >= 0:
            new = 0.0
        if new != pos:
            pnl -= abs(new - pos) * (1 + abs(b)) * cost / 1e4
            pos = new
    return pnl / (1 + abs(b))


def copula_strategy(y, x, form, cost, thr=0.6):
    ry, rx = np.diff(y), np.diff(x)
    fy, fx = ry[:form - 1], rx[:form - 1]
    u0, v0 = pseudo_obs(fy), pseudo_obs(fx)
    tau, fit, best = fit_all(u0, v0)
    hfun = {"gaussienne": h_gauss, "student": h_t, "clayton": h_clayton}[best]
    par = fit[best][0]
    sy, sx = np.sort(fy), np.sort(fx)
    ecdf = lambda srt, z: np.clip(np.searchsorted(srt, z) / (len(srt) + 1), 1e-4, 1 - 1e-4)
    cmi_y = cmi_x = 0.0
    pos, pnl = 0.0, 0.0                 # pos = +1 : long y, court x
    for t in range(form, len(ry)):
        pnl += pos * (ry[t] - rx[t])
        u, v = ecdf(sy, ry[t]), ecdf(sx, rx[t])
        cmi_y += hfun(u, v, par) - 0.5
        cmi_x += hfun(v, u, par) - 0.5
        new = pos
        if pos == 0:
            if cmi_y > thr and cmi_x < -thr: new = -1.0        # y trop haut relativement à x
            elif cmi_y < -thr and cmi_x > thr: new = 1.0
        elif (pos == -1 and cmi_y <= 0) or (pos == 1 and cmi_y >= 0):
            new = 0.0
        if new != pos:
            pnl -= abs(new - pos) * 2 * cost / 1e4
            pos = new
            if new == 0:
                cmi_y = cmi_x = 0.0
    return pnl / 2, best


def e3():
    rng = np.random.default_rng(93)
    zs, cs, picks = [], [], {}
    for _ in range(60):
        y, x = make_pair(rng, 1000)
        zs.append(zscore_strategy(y, x, 500, 5))
        c, best = copula_strategy(y, x, 500, 5)
        cs.append(c); picks[best] = picks.get(best, 0) + 1
    ch.record("E3", {"z_score_moyen_pct_sur_500j": r3(100 * np.mean(zs), 2), "z_score_ecart_type_pct": r3(100 * np.std(zs), 2),
                     "copule_moyen_pct_sur_500j": r3(100 * np.mean(cs), 2), "copule_ecart_type_pct": r3(100 * np.std(cs), 2),
                     "copules_choisies": picks, "paires": 60})


# ---------------------------------------------------------------- E4
def e4():
    ko, pep = np.diff(np.log(load_close("KO"))), np.diff(np.log(load_close("PEP")))
    u, v = pseudo_obs(ko), pseudo_obs(pep)
    tau, fit, best = fit_all(u, v)
    q = 0.05
    obs = int(np.sum((u < q) & (v < q)))
    rho = fit["gaussienne"][0]
    sims = {"gaussienne": sim_gauss(np.random.default_rng(1), 200000, rho),
            "student": sim_t(np.random.default_rng(2), 200000, rho),
            "clayton": sim_clayton(np.random.default_rng(3), 200000, fit["clayton"][0])}
    pred = {k: r3(len(u) * np.mean((s[:, 0] < q) & (s[:, 1] < q)), 1) for k, s in sims.items()}
    ch.record("E4", {"tau_kendall": r3(tau, 3), "copule_AIC": best,
                     "loglik": {k: r3(v_[1], 1) for k, v_ in fit.items()},
                     "co_baisses_sous_5pct_observees": obs, "co_baisses_attendues": pred,
                     "attendu_si_independance": r3(len(u) * q * q, 1), "jours": len(u)})


if __name__ == "__main__":
    e1(); e2(); e3(); e4()
    ch.save_results()
