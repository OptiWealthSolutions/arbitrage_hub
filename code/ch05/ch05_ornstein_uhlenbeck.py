"""Chapitre 5 — Le processus d'Ornstein-Uhlenbeck.

E1  Discrétisation exacte vs schéma d'Euler : moyenne, variance stationnaire, autocorrélation.
E2  Estimation (theta, mu, sigma) par la correspondance AR(1) <-> OU, sur 500 trajectoires.
E3  Temps de premier passage : formule intégrale (équation de Kolmogorov) vs simulation.
E4  Seuils de trading optimaux : rendement par unité de temps (a - c) / E[cycle], formule et simulation.
E5  Données réelles : paramètres OU de l'écart GLD/GLDM sans tendance.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize, special

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, load_close, r3

ch = Chapter("ch05")


def ou_exact(rng, n, dt, theta, mu, sigma, x0, n_paths=1):
    phi = np.exp(-theta * dt)
    sd = sigma * np.sqrt((1 - phi**2) / (2 * theta))
    x = np.empty((n_paths, n + 1)); x[:, 0] = x0
    z = rng.normal(size=(n_paths, n))
    for t in range(n):
        x[:, t + 1] = mu + phi * (x[:, t] - mu) + sd * z[:, t]
    return x


def ou_euler(rng, n, dt, theta, mu, sigma, x0, n_paths=1):
    x = np.empty((n_paths, n + 1)); x[:, 0] = x0
    z = rng.normal(size=(n_paths, n))
    for t in range(n):
        x[:, t + 1] = x[:, t] + theta * (mu - x[:, t]) * dt + sigma * np.sqrt(dt) * z[:, t]
    return x


# ---------------------------------------------------------------- E1
def e1():
    rng = np.random.default_rng(51)
    theta, mu, sigma = 2.0, 0.0, 1.0          # temps en années, demi-vie = ln2/2 = 0,35 an
    var_th = sigma**2 / (2 * theta)
    out = {}
    for dt in (1 / 252, 1 / 12, 0.25):
        n = int(2000 / dt) if dt < 0.1 else 20000
        xe = ou_exact(rng, n, dt, theta, mu, sigma, 0.0)[0][1000:]
        xu = ou_euler(rng, n, dt, theta, mu, sigma, 0.0)[0][1000:]
        out[f"dt={dt:.4f}"] = {"var_exacte": r3(xe.var(), 4), "var_euler": r3(xu.var(), 4), "var_theorique": var_th,
                               "acf1_exacte": r3(np.corrcoef(xe[1:], xe[:-1])[0, 1], 4),
                               "acf1_euler": r3(np.corrcoef(xu[1:], xu[:-1])[0, 1], 4),
                               "acf1_theorique": r3(np.exp(-theta * dt), 4), "euler_phi": r3(1 - theta * dt, 4)}
    ch.record("E1", {"theta": theta, "sigma": sigma, "resultats": out})

    paths = ou_exact(rng, 252 * 2, 1 / 252, theta, 0.0, sigma, 1.5, n_paths=2000)
    t = np.arange(paths.shape[1]) / 252
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    for i in range(25):
        ax.plot(t, paths[i], color=MUTED, lw=0.4, alpha=0.6)
    m = 1.5 * np.exp(-theta * t)
    sd = np.sqrt(sigma**2 * (1 - np.exp(-2 * theta * t)) / (2 * theta))
    ax.plot(t, m, color=INK, lw=1.5, label=r"$\mathbb{E}[X_t]$")
    ax.fill_between(t, m - 2 * sd, m + 2 * sd, color=ACCENT, alpha=0.15, label=r"$\pm 2$ écarts-types")
    ax.set_xlabel("temps (années)")
    ax.set_title(r"OU : $\theta=2$, $\sigma=1$, départ $x_0=1{,}5$")
    ax.legend(frameon=False, fontsize=7)
    ch.save_fig(fig, "ou_trajectoires")


# ---------------------------------------------------------------- E2
def estimate_ou(x, dt):
    b, a = np.polyfit(x[:-1], x[1:], 1)
    e = x[1:] - a - b * x[:-1]
    theta = -np.log(b) / dt
    mu = a / (1 - b)
    sigma = e.std(ddof=2) * np.sqrt(2 * theta / (1 - b**2))
    return theta, mu, sigma


def e2():
    rng = np.random.default_rng(52)
    theta, mu, sigma, dt = 5.0, 1.0, 0.5, 1 / 252
    out = {}
    for years in (1, 4, 16):
        n = int(252 * years)
        est = []
        for _ in range(500):
            x = ou_exact(rng, n, dt, theta, mu, sigma, mu + rng.normal(0, sigma / np.sqrt(2 * theta)))[0]
            b = np.polyfit(x[:-1], x[1:], 1)[0]
            if 0 < b < 1:
                est.append(estimate_ou(x, dt))
        est = np.array(est)
        out[f"{years}_an(s)"] = {"theta_mediane": r3(np.median(est[:, 0]), 2), "theta_q05": r3(np.quantile(est[:, 0], 0.05), 2),
                                 "theta_q95": r3(np.quantile(est[:, 0], 0.95), 2), "mu_ecart_type": r3(est[:, 1].std(), 4),
                                 "sigma_mediane": r3(np.median(est[:, 2]), 4)}
    ch.record("E2", {"vrai": {"theta": theta, "mu": mu, "sigma": sigma, "demi_vie_jours": r3(252 * np.log(2) / theta, 1)},
                     "par_duree": out})


# ---------------------------------------------------------------- E3/E4 : temps de passage
def t_to_mean(a, theta, sigma):
    """E[temps pour aller de a (>0) à la moyenne 0]."""
    k = theta / sigma**2
    inner = lambda y: np.sqrt(np.pi / (4 * k)) * special.erfc(np.sqrt(k) * y)   # int_y^inf exp(-k z^2) dz
    f = lambda y: (2 / sigma**2) * np.exp(k * y**2) * inner(y)
    return integrate.quad(f, 0, a, limit=200)[0]


def t_from_mean(a, theta, sigma):
    """E[temps pour aller de la moyenne 0 à |x| = a]."""
    k = theta / sigma**2
    inner = lambda y: np.sqrt(np.pi / (4 * k)) * special.erf(np.sqrt(k) * y)    # int_0^y exp(-k z^2) dz
    f = lambda y: (2 / sigma**2) * np.exp(k * y**2) * inner(y)
    return integrate.quad(f, 0, a, limit=200)[0]


def e3():
    rng = np.random.default_rng(53)
    theta, sigma, dt = 1.0, 1.0, 1e-3
    sd_eq = sigma / np.sqrt(2 * theta)
    rows = []
    for a_eq in (0.5, 1.0, 1.5, 2.0):
        a = a_eq * sd_eq
        # simulation : de a vers 0
        n_paths, x = 4000, np.full(4000, a)
        tt = np.zeros(n_paths); alive = np.ones(n_paths, bool); step = 0
        phi = np.exp(-theta * dt); sd = sigma * np.sqrt((1 - phi**2) / (2 * theta))
        while alive.any() and step < 200000:
            x[alive] = phi * x[alive] + sd * rng.normal(size=alive.sum())
            tt[alive] += dt
            alive &= x > 0
            step += 1
        # simulation : de 0 vers |x| = a
        y = np.zeros(n_paths); t2 = np.zeros(n_paths); alive2 = np.ones(n_paths, bool); step = 0
        while alive2.any() and step < 400000:
            y[alive2] = phi * y[alive2] + sd * rng.normal(size=alive2.sum())
            t2[alive2] += dt
            alive2 &= np.abs(y) < a
            step += 1
        rows.append({"a_en_ecarts_types": a_eq, "T_vers_moyenne_formule": r3(t_to_mean(a, theta, sigma), 3),
                     "T_vers_moyenne_simulation": r3(tt.mean(), 3),
                     "T_depuis_moyenne_formule": r3(t_from_mean(a, theta, sigma), 3),
                     "T_depuis_moyenne_simulation": r3(t2.mean(), 3)})
    ch.record("E3", {"theta": theta, "sigma": sigma, "pas_simulation": dt, "tableau": rows})


def e4():
    theta, sigma = 1.0, 1.0
    sd_eq = sigma / np.sqrt(2 * theta)
    grid = np.linspace(0.05, 3.0, 120) * sd_eq
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    out = {}
    for c_eq, col in zip((0.0, 0.1, 0.25, 0.5), (INK, GREEN, GOLD, ACCENT)):
        c = c_eq * sd_eq
        rate = [(a - c) / (t_from_mean(a, theta, sigma) + t_to_mean(a, theta, sigma)) for a in grid]
        rate = np.array(rate) / sd_eq
        i = int(np.argmax(rate))
        out[f"cout={c_eq}"] = {"a_optimal_en_ecarts_types": r3(grid[i] / sd_eq, 2),
                               "rendement_par_unite_de_temps": r3(rate[i], 3)}
        ax.plot(grid / sd_eq, rate, color=col, label=f"coût = {c_eq} écart-type")
        ax.plot(grid[i] / sd_eq, rate[i], "o", color=col, ms=3)
    ax.set_xlabel(r"seuil d'entrée $a$ (en écarts-types stationnaires)")
    ax.set_ylabel("gain par unité de temps")
    ax.set_title("Seuil optimal : (a − c) / E[durée du cycle]")
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(bottom=0)
    ch.save_fig(fig, "seuils_optimaux")

    # vérification par simulation en temps discret quotidien
    rng = np.random.default_rng(54)
    dt, years = 1 / 252, 400
    theta_d, sigma_d = 12.6, 1.0            # demi-vie ~ 14 jours
    sdq = sigma_d / np.sqrt(2 * theta_d)
    x = ou_exact(rng, int(252 * years), dt, theta_d, 0.0, sigma_d, 0.0)[0]
    sim = {}
    for a_eq in (0.25, 0.5, 1.0, 1.5, 2.0):
        for c_eq in (0.0, 0.25):
            a, c = a_eq * sdq, c_eq * sdq
            pos, pnl, ntr = 0, 0.0, 0
            for t in range(1, len(x)):
                pnl += pos * (x[t] - x[t - 1])
                if pos == 0 and abs(x[t]) >= a:
                    pos = -np.sign(x[t]); pnl -= c / 2; ntr += 1
                elif pos != 0 and np.sign(x[t]) == pos:   # l'écart a repassé la moyenne
                    pos = 0; pnl -= c / 2
            sim[f"a={a_eq},c={c_eq}"] = {"gain_par_an_en_ecarts_types": r3(pnl / years / sdq, 2), "trades_par_an": r3(ntr / years, 1),
                                         "formule": r3((a - c) / (t_from_mean(a, theta_d, sigma_d) + t_to_mean(a, theta_d, sigma_d)) / sdq, 2)}
    ch.record("E4", {"optimum_theta1_sigma1": out, "verification_quotidienne_theta12.6": sim})


# ---------------------------------------------------------------- E5
def e5():
    g, m = np.log(load_close("GLD")), np.log(load_close("GLDM"))
    s = g - m
    t = np.arange(len(s))
    d = s - np.polyval(np.polyfit(t, s, 1), t)
    b = np.polyfit(d[:-1], d[1:], 1)[0]
    ch.record("E5", {"phi_quotidien": r3(b, 4),
                     "note": "phi proche de 0 : theta = -ln(phi)*252 n'a pas de sens ; l'écart est un bruit blanc à l'échelle quotidienne",
                     "ecart_type_pb": r3(1e4 * d.std(), 2)})


if __name__ == "__main__":
    e1(); e2(); e3(); e4(); e5()
    ch.save_results()
