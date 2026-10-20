"""Chapitre 4 — Cointégration.

E1  Valeurs critiques d'Engle-Granger simulées (résidus estimés) vs Dickey-Fuller.
E2  Superconvergence de beta_hat dans une paire cointégrée.
E3  Modèle à correction d'erreur : estimation des vitesses d'ajustement et demi-vie de l'écart.
E4  Johansen sur un système de 3 actifs de rang 1.
E5  Cointégration partielle : quand la composante marche aléatoire grandit, ADF échoue.
E6  Données réelles : Engle-Granger et Johansen sur GLD/GLDM et KO/PEP.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, load_close, ar1, r3

ch = Chapter("ch04")


def adf_tau(x, lags=0):
    return adfuller(x, maxlag=lags, autolag=None, regression="c")[0]


def eg_tau(y, x):
    b, a = np.polyfit(x, y, 1)
    return adf_tau(y - a - b * x)


# ---------------------------------------------------------------- E1
def e1():
    rng = np.random.default_rng(41)
    n, reps = 500, 10000
    taus = []
    for _ in range(reps):
        x = np.cumsum(rng.normal(size=n))
        y = np.cumsum(rng.normal(size=n))
        taus.append(eg_tau(y, x))
    taus = np.array(taus)
    q = np.quantile(taus, [0.01, 0.05, 0.10])
    ch.record("E1", {"n": n, "repetitions": reps, "quantiles_EG": [r3(v, 2) for v in q],
                     "MacKinnon_2_variables": [-3.90, -3.34, -3.04],
                     "faux_positifs_si_seuil_DF_-2.86": r3((taus < -2.86).mean(), 3),
                     "faux_positifs_si_seuil_EG_-3.34": r3((taus < -3.34).mean(), 3)})
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    ax.hist(taus, bins=80, density=True, color=INK, alpha=0.8, label="résidus estimés (EG)")
    df_taus = np.array([adf_tau(np.cumsum(rng.normal(size=n))) for _ in range(5000)])
    ax.hist(df_taus, bins=80, density=True, color=GOLD, alpha=0.5, label="série observée (DF)")
    ax.axvline(q[1], color=INK, ls="--", lw=1)
    ax.axvline(-2.86, color=GOLD, ls="--", lw=1)
    ax.set_xlabel(r"statistique $\tau$ sous $H_0$ (pas de cointégration)")
    ax.set_title("Engle-Granger est plus exigeant que Dickey-Fuller")
    ax.legend(frameon=False, fontsize=7)
    ch.save_fig(fig, "eg_distribution")


# ---------------------------------------------------------------- E2
def e2():
    rng = np.random.default_rng(42)
    beta, out = 1.5, {}
    for n in (250, 1000, 4000):
        bs = []
        for _ in range(500):
            x = 100 + np.cumsum(rng.normal(size=n))
            y = beta * x + ar1(rng, n - 1, 0.9)[0]
            bs.append(np.polyfit(x, y, 1)[0])
        out[n] = {"moyenne": r3(np.mean(bs), 4), "ecart_type": r3(np.std(bs), 5)}
    # comparaison : régression de deux variables STATIONNAIRES (vitesse racine de n)
    st = {}
    for n in (250, 1000, 4000):
        bs = []
        for _ in range(500):
            x = rng.normal(size=n)
            y = beta * x + rng.normal(size=n)
            bs.append(np.polyfit(x, y, 1)[0])
        st[n] = r3(np.std(bs), 5)
    ch.record("E2", {"paire_cointegree": out, "variables_stationnaires_ecart_type": st})


# ---------------------------------------------------------------- E3
def e3():
    rng = np.random.default_rng(43)
    n, beta, ay, ax_ = 2000, 1.0, -0.10, 0.05
    x = np.zeros(n); y = np.zeros(n)
    x[0], y[0] = 100, 100
    for t in range(1, n):
        s = y[t - 1] - beta * x[t - 1]
        y[t] = y[t - 1] + ay * s + rng.normal(0, 1)
        x[t] = x[t - 1] + ax_ * s + rng.normal(0, 1)
    b_hat, a_hat = np.polyfit(x, y, 1)
    s = y - a_hat - b_hat * x
    dy, dx = np.diff(y), np.diff(x)
    Z = sm.add_constant(s[:-1])
    my, mx = sm.OLS(dy, Z).fit(), sm.OLS(dx, Z).fit()
    phi_s = np.polyfit(s[:-1], s[1:], 1)[0]
    ch.record("E3", {"vrai": {"alpha_y": ay, "alpha_x": ax_, "beta": beta,
                              "phi_ecart_theorique": 1 + ay - beta * ax_,
                              "demi_vie_theorique": r3(-np.log(2) / np.log(1 + ay - beta * ax_), 2)},
                     "estime": {"beta": r3(b_hat, 4), "alpha_y": r3(my.params[1], 4), "t_alpha_y": r3(my.tvalues[1], 1),
                                "alpha_x": r3(mx.params[1], 4), "t_alpha_x": r3(mx.tvalues[1], 1),
                                "phi_ecart": r3(phi_s, 4), "demi_vie": r3(-np.log(2) / np.log(phi_s), 2)}})


# ---------------------------------------------------------------- E4
def e4():
    rng = np.random.default_rng(44)
    n = 1000
    common = np.cumsum(rng.normal(size=n))
    other = np.cumsum(rng.normal(size=n))
    s = ar1(rng, n - 1, 0.9)[0]
    # x1 et x2 partagent la tendance commune ; x3 a sa propre tendance -> une seule relation
    x1 = common + rng.normal(0, 0.5, n)
    x2 = 0.5 * common + s
    x3 = other + rng.normal(0, 0.5, n)
    data = np.column_stack([x1, x2, x3])
    jo = coint_johansen(data, det_order=0, k_ar_diff=1)
    vec = jo.evec[:, 0] / jo.evec[1, 0]
    ch.record("E4", {"trace": [r3(v, 2) for v in jo.lr1], "trace_crit_95": [r3(v, 2) for v in jo.cvt[:, 1]],
                     "maxeig": [r3(v, 2) for v in jo.lr2], "maxeig_crit_95": [r3(v, 2) for v in jo.cvm[:, 1]],
                     "vecteur_normalise_sur_x2": [r3(v, 3) for v in vec],
                     "vrai_vecteur": "x2 - 0.5 x1 (soit -0.5, 1, 0)"})


# ---------------------------------------------------------------- E5
def e5():
    rng = np.random.default_rng(45)
    n, reps, rho, sm_ = 1000, 300, 0.9, 1.0
    rows = []
    for sr in (0.0, 0.1, 0.2, 0.4, 0.8):
        rej = 0
        for _ in range(reps):
            m = ar1(rng, n - 1, rho, sm_)[0]
            r = np.cumsum(rng.normal(0, sr, n)) if sr > 0 else np.zeros(n)
            rej += adfuller(m + r, autolag="AIC", regression="c")[1] < 0.05
        r2mr = 2 * sm_**2 / (2 * sm_**2 + (1 + rho) * sr**2)
        rows.append({"sigma_R": sr, "R2_MR": r3(r2mr, 3), "ADF_rejette": r3(rej / reps, 3)})
    ch.record("E5", {"rho": rho, "sigma_M": sm_, "n": n, "tableau": rows})


# ---------------------------------------------------------------- E6
def e6():
    gld, gldm = np.log(load_close("GLD")), np.log(load_close("GLDM"))
    ko, pep = np.log(load_close("KO")), np.log(load_close("PEP"))
    res = {}
    for name, y, x, trend in [("GLD_GLDM_constante", gld, gldm, "c"), ("GLD_GLDM_tendance", gld, gldm, "ct"),
                              ("KO_PEP", ko, pep, "c"), ("PEP_KO", pep, ko, "c")]:
        t, p, cv = coint(y, x, trend=trend, autolag="AIC")
        res[name] = {"tau": r3(t, 2), "p": r3(p, 3), "crit_5": r3(cv[1], 2)}
    for name, data in [("GLD_GLDM", np.column_stack([gld, gldm])), ("KO_PEP", np.column_stack([ko, pep]))]:
        jo = coint_johansen(data, det_order=1 if name == "GLD_GLDM" else 0, k_ar_diff=1)
        res["Johansen_" + name] = {"trace": [r3(v, 2) for v in jo.lr1], "crit_95": [r3(v, 2) for v in jo.cvt[:, 1]]}
    ch.record("E6", res)


if __name__ == "__main__":
    e1(); e2(); e3(); e4(); e5(); e6()
    ch.save_results()
