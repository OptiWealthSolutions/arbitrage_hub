"""Chapitre 2 — Régression et estimation : toutes les vérifications du chapitre.

E1  MCO écrits à la main = statsmodels (données réelles KO/PEP).
E2  Asymétrie Y sur X / X sur Y et régression orthogonale (simulation + KO/PEP).
E3  Maximum de vraisemblance de l'AR(1) : conditionnel = MCO ; exact ~ conditionnel.
E4  Biais de petit échantillon et correction de Kendall.
E5  Erreurs-types naïves contre Newey-West quand les résidus sont autocorrélés.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import minimize

from style import Chapter, INK, ACCENT, MUTED, GREEN, load_close, ar1, r3

ch = Chapter("ch02")


# ---------------------------------------------------------------- E1
def ols(y, X):
    """MCO matriciels : b = (X'X)^-1 X'y, erreurs-types classiques, R^2."""
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    n, k = X.shape
    s2 = e @ e / (n - k)
    se = np.sqrt(np.diag(s2 * XtX_inv))
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    return b, se, r2, e


def e1():
    ko, pep = load_close("KO"), load_close("PEP")
    X = np.column_stack([np.ones_like(pep), pep])
    b, se, r2, e = ols(ko, X)
    m = sm.OLS(ko, X).fit()
    ch.record("E1", {"b_main": [r3(b[0], 4), r3(b[1], 5)], "se_main": [r3(se[0], 4), r3(se[1], 5)], "r2_main": r3(r2, 4),
                     "b_sm": [r3(m.params[0], 4), r3(m.params[1], 5)], "se_sm": [r3(m.bse[0], 4), r3(m.bse[1], 5)],
                     "r2_sm": r3(m.rsquared, 4), "ecart_max": float(np.max(np.abs(b - m.params))),
                     "orthogonalite_Xe": float(np.max(np.abs(X.T @ e)))})


# ---------------------------------------------------------------- E2
def tls_slope(x, y):
    """Pente de la régression orthogonale (moindres carrés totaux)."""
    sx2, sy2, sxy = np.var(x, ddof=1), np.var(y, ddof=1), np.cov(x, y)[0, 1]
    return (sy2 - sx2 + np.sqrt((sy2 - sx2) ** 2 + 4 * sxy**2)) / (2 * sxy)


def e2():
    rng = np.random.default_rng(22)
    beta, n = 1.5, 500
    res = {"OLS_Y_sur_X": [], "inverse_OLS_X_sur_Y": [], "TLS": []}
    for _ in range(1000):
        x = 100 + np.cumsum(rng.normal(0, 1, n))
        s = ar1(rng, n - 1, 0.9, 1.0)[0]
        y = beta * x + s
        res["OLS_Y_sur_X"].append(np.polyfit(x, y, 1)[0])
        res["inverse_OLS_X_sur_Y"].append(1 / np.polyfit(y, x, 1)[0])
        res["TLS"].append(tls_slope(x, y))
    sim = {k: {"moyenne": r3(np.mean(v), 4), "ecart_type": r3(np.std(v), 4)} for k, v in res.items()}

    # Variables stationnaires : l'asymétrie est forte quand la corrélation est faible
    rho_tab = []
    for rho in (0.3, 0.6, 0.9, 0.99):
        x = rng.normal(size=20000)
        y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=20000)
        b_yx = np.polyfit(x, y, 1)[0]
        b_xy = np.polyfit(y, x, 1)[0]
        rho_tab.append({"rho": rho, "b_YX": r3(b_yx), "un_sur_b_XY": r3(1 / b_xy), "produit": r3(b_yx * b_xy)})

    ko, pep = load_close("KO"), load_close("PEP")
    lk, lp = np.log(ko), np.log(pep)
    real = {"prix": {"b_KO_sur_PEP": r3(np.polyfit(pep, ko, 1)[0], 4),
                     "un_sur_b_PEP_sur_KO": r3(1 / np.polyfit(ko, pep, 1)[0], 4),
                     "TLS": r3(tls_slope(pep, ko), 4),
                     "corr_niveaux": r3(np.corrcoef(ko, pep)[0, 1], 4)},
            "log": {"b_logKO_sur_logPEP": r3(np.polyfit(lp, lk, 1)[0], 4),
                    "un_sur_b": r3(1 / np.polyfit(lk, lp, 1)[0], 4),
                    "TLS": r3(tls_slope(lp, lk), 4)}}
    ch.record("E2", {"simulation_beta_1.5": sim, "stationnaire": rho_tab, "KO_PEP": real})

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ax = axes[0]
    ax.scatter(pep, ko, s=4, color=MUTED, alpha=0.6, label="séances")
    xs = np.linspace(pep.min(), pep.max(), 10)
    for slope_fn, lab, col, ls in [(lambda: np.polyfit(pep, ko, 1), "KO sur PEP", INK, "-"),
                                   (None, "PEP sur KO (inversée)", ACCENT, "--")]:
        if slope_fn:
            b1, a1 = slope_fn()
        else:
            c1, d1 = np.polyfit(ko, pep, 1)
            b1, a1 = 1 / c1, -d1 / c1
        ax.plot(xs, a1 + b1 * xs, color=col, ls=ls, lw=1.3, label=lab)
    bt = tls_slope(pep, ko)
    ax.plot(xs, ko.mean() + bt * (xs - pep.mean()), color=GREEN, ls=":", lw=1.5, label="orthogonale")
    ax.set_xlabel("PEP ($)")
    ax.set_ylabel("KO ($)")
    ax.set_title("Trois droites pour un même nuage")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    rhos = [d["rho"] for d in rho_tab]
    ax.plot(rhos, [d["b_YX"] for d in rho_tab], "o-", color=INK, label=r"$\hat\beta_{Y|X}$")
    ax.plot(rhos, [d["un_sur_b_XY"] for d in rho_tab], "s--", color=ACCENT, label=r"$1/\hat\beta_{X|Y}$")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_yscale("log")
    ax.set_xlabel(r"corrélation $\rho$ ($\sigma_X=\sigma_Y=1$)")
    ax.set_title("Asymétrie selon la corrélation")
    ax.legend(frameon=False)
    ch.save_fig(fig, "asymetrie_regression")


# ---------------------------------------------------------------- E3
def e3():
    rng = np.random.default_rng(23)
    phi, mu, sigma, n = 0.9, 2.0, 0.5, 300
    x = ar1(rng, n - 1, phi, sigma, mu)[0]

    def negll_cond(p):
        c, f, ls = p
        s = np.exp(ls)
        e = x[1:] - c - f * x[:-1]
        return 0.5 * (n - 1) * np.log(2 * np.pi * s**2) + 0.5 * np.sum(e**2) / s**2

    def negll_exact(p):
        c, f, ls = p
        if abs(f) >= 0.9999:
            return 1e12
        s = np.exp(ls)
        m0, v0 = c / (1 - f), s**2 / (1 - f**2)
        return negll_cond(p) + 0.5 * np.log(2 * np.pi * v0) + 0.5 * (x[0] - m0) ** 2 / v0

    p0 = np.array([0.0, 0.5, 0.0])
    pc = minimize(negll_cond, p0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000}).x
    pe = minimize(negll_exact, p0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000}).x
    b, a = np.polyfit(x[:-1], x[1:], 1)
    resid = x[1:] - a - b * x[:-1]
    ch.record("E3", {"vrai": {"c": mu * (1 - phi), "phi": phi, "sigma": sigma},
                     "MCO": {"c": r3(a, 4), "phi": r3(b, 4), "sigma_MV": r3(np.sqrt(np.mean(resid**2)), 4)},
                     "MV_conditionnel": {"c": r3(pc[0], 4), "phi": r3(pc[1], 4), "sigma": r3(np.exp(pc[2]), 4)},
                     "MV_exact": {"c": r3(pe[0], 4), "phi": r3(pe[1], 4), "sigma": r3(np.exp(pe[2]), 4)}})

    # surface de log-vraisemblance en phi (c et sigma profilés)
    grid = np.linspace(0.75, 0.99, 200)
    prof = []
    for f in grid:
        cc = np.mean(x[1:] - f * x[:-1])
        e = x[1:] - cc - f * x[:-1]
        s2 = np.mean(e**2)
        prof.append(-0.5 * (n - 1) * (np.log(2 * np.pi * s2) + 1))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(grid, prof, color=INK)
    ax.axvline(b, color=ACCENT, ls="--", lw=1, label=rf"maximum $\hat\varphi={b:.3f}$")
    ax.axvline(phi, color=GREEN, ls=":", lw=1.2, label=rf"vrai $\varphi={phi}$")
    ax.set_xlabel(r"$\varphi$")
    ax.set_title("Log-vraisemblance profilée")
    ax.legend(frameon=False, fontsize=7)
    ch.save_fig(fig, "vraisemblance_ar1")


# ---------------------------------------------------------------- E4
def e4():
    rng = np.random.default_rng(24)
    phi = 0.95
    out = {}
    for n in (100, 250, 1000):
        raw, cor = [], []
        for _ in range(3000):
            x = ar1(rng, n - 1, phi, 1.0)[0]
            b = np.polyfit(x[:-1], x[1:], 1)[0]
            raw.append(b)
            cor.append(min(b + (1 + 3 * b) / n, 0.9999))
        raw, cor = np.array(raw), np.array(cor)
        hl = lambda f: np.median(-np.log(2) / np.log(np.clip(f, 1e-6, 0.9999)))
        out[n] = {"phi_moyen_brut": r3(raw.mean(), 4), "phi_moyen_corrige": r3(cor.mean(), 4),
                  "demi_vie_mediane_brute": r3(hl(raw), 2), "demi_vie_mediane_corrigee": r3(hl(cor), 2),
                  "ecart_type_phi": r3(raw.std(), 4)}
    ch.record("E4", {"phi": phi, "demi_vie_vraie": r3(-np.log(2) / np.log(phi), 2), "par_n": out})


# ---------------------------------------------------------------- E5
def e5():
    rng = np.random.default_rng(25)
    n, reps = 250, 2000
    rej_naive, rej_hac, rej_hac20 = 0, 0, 0
    for _ in range(reps):
        x = ar1(rng, n - 1, 0.9, 1.0)[0]
        u = ar1(rng, n - 1, 0.9, 1.0)[0]   # résidus autocorrélés, indépendants de x
        y = 1.0 + 0.0 * x + u               # vraie pente nulle
        X = sm.add_constant(x)
        m1 = sm.OLS(y, X).fit()
        m2 = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(4 * (n / 100) ** (2 / 9))})
        m3 = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
        rej_naive += abs(m1.tvalues[1]) > 1.96
        rej_hac += abs(m2.tvalues[1]) > 1.96
        rej_hac20 += abs(m3.tvalues[1]) > 1.96
    ch.record("E5", {"n": n, "repetitions": reps, "lags_NW": int(4 * (n / 100) ** (2 / 9)),
                     "rejet_naif": r3(rej_naive / reps), "rejet_NeweyWest": r3(rej_hac / reps), "rejet_NeweyWest_20_lags": r3(rej_hac20 / reps)})


if __name__ == "__main__":
    e1(); e2(); e3(); e4(); e5()
    ch.save_results()
    import json
    print(json.dumps(ch.results, indent=1, ensure_ascii=False))
