"""Chapitre 3 — Tester la stationnarité et le retour à la moyenne.

E1  Distribution de Dickey-Fuller simulée sous H0 (marche aléatoire) et valeurs critiques.
E2  Puissance du test ADF selon phi et la taille d'échantillon.
E3  KPSS : taille et puissance ; tableau de décision ADF x KPSS.
E4  Ratio de variance (théorie vs simulation) et test de Lo-MacKinlay.
E5  Exposant de Hurst par la méthode des variances des écarts.
E6  Étude de cas réelle : écart GLD/GLDM et écart log KO/PEP.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, load_close, ar1, r3

ch = Chapter("ch03")


def df_stat(x):
    """Statistique tau de Dickey-Fuller avec constante, sans retards : dx_t = a + g x_{t-1} + e."""
    dx, xl = np.diff(x), x[:-1]
    X = np.column_stack([np.ones_like(xl), xl])
    b, *_ = np.linalg.lstsq(X, dx, rcond=None)
    e = dx - X @ b
    s2 = e @ e / (len(dx) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return b[1] / se


# ---------------------------------------------------------------- E1
def e1():
    rng = np.random.default_rng(31)
    n, reps = 500, 20000
    taus = np.array([df_stat(np.cumsum(rng.normal(size=n))) for _ in range(reps)])
    q = np.quantile(taus, [0.01, 0.05, 0.10])
    size_if_normal = float((taus < -1.645).mean())
    ch.record("E1", {"n": n, "repetitions": reps, "quantiles_1_5_10": [r3(v, 3) for v in q],
                     "valeurs_tabulees_MacKinnon": [-3.43, -2.86, -2.57],
                     "rejet_si_on_utilise_-1.645": r3(size_if_normal, 3),
                     "moyenne_tau": r3(taus.mean(), 3)})
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    ax.hist(taus, bins=80, density=True, color=INK, alpha=0.8, label="DF simulée (H0 vraie)")
    z = np.linspace(-5, 3, 300)
    ax.plot(z, np.exp(-z**2 / 2) / np.sqrt(2 * np.pi), color=ACCENT, lw=1.3, label="loi normale")
    ax.axvline(q[1], color=GOLD, ls="--", lw=1, label=f"5 % DF = {q[1]:.2f}")
    ax.axvline(-1.645, color=ACCENT, ls=":", lw=1, label="5 % normale = −1,645")
    ax.set_xlabel(r"statistique $\tau$")
    ax.set_title("La statistique de Dickey-Fuller n'est pas normale")
    ax.legend(frameon=False, fontsize=7)
    ch.save_fig(fig, "dickey_fuller_distribution")


# ---------------------------------------------------------------- E2
def e2():
    rng = np.random.default_rng(32)
    phis = [0.90, 0.95, 0.98, 0.99]
    ns = [100, 250, 500, 1000, 2500]
    table = {}
    for phi in phis:
        row = []
        for n in ns:
            reps = 400 if n <= 1000 else 200
            rej = 0
            for _ in range(reps):
                x = ar1(rng, n - 1, phi, 1.0)[0]
                rej += adfuller(x, maxlag=1, autolag=None, regression="c")[1] < 0.05
            row.append(r3(rej / reps, 3))
        table[str(phi)] = row
    ch.record("E2", {"n": ns, "puissance_par_phi": table})
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for phi, c in zip(phis, [INK, GREEN, GOLD, ACCENT]):
        hl = -np.log(2) / np.log(phi)
        ax.plot(ns, table[str(phi)], "o-", color=c, label=rf"$\varphi={phi}$ (demi-vie {hl:.0f})")
    ax.axhline(0.05, color="k", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("nombre d'observations n")
    ax.set_ylabel("probabilité de rejeter H0")
    ax.set_title("Puissance du test ADF à 5 %")
    ax.legend(frameon=False, fontsize=7)
    ch.save_fig(fig, "puissance_adf")


# ---------------------------------------------------------------- E3
def e3():
    rng = np.random.default_rng(33)
    n, reps = 500, 400
    out = {}
    for name, gen in [("marche_aleatoire", lambda: np.cumsum(rng.normal(size=n))),
                      ("AR_0.95", lambda: ar1(rng, n - 1, 0.95)[0]),
                      ("AR_0.5", lambda: ar1(rng, n - 1, 0.5)[0])]:
        counts = {"ADF_rejette_et_KPSS_accepte": 0, "ADF_accepte_et_KPSS_rejette": 0,
                  "les_deux_rejettent": 0, "aucun_ne_rejette": 0}
        for _ in range(reps):
            x = gen()
            a = adfuller(x, autolag="AIC", regression="c")[1] < 0.05
            k = kpss(x, regression="c", nlags="auto")[1] < 0.05
            key = ("les_deux_rejettent" if a and k else "ADF_rejette_et_KPSS_accepte" if a else
                   "ADF_accepte_et_KPSS_rejette" if k else "aucun_ne_rejette")
            counts[key] += 1
        out[name] = {k: r3(v / reps, 3) for k, v in counts.items()}
    ch.record("E3", {"n": n, "repetitions": reps, "frequences": out})


# ---------------------------------------------------------------- E4
def vr(x, q):
    d1 = np.diff(x)
    dq = x[q:] - x[:-q]
    return dq.var(ddof=1) / (q * d1.var(ddof=1))


def lo_mackinlay_z(x, q):
    n = len(x) - 1
    return (vr(x, q) - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * n))


def e4():
    rng = np.random.default_rng(34)
    qs = np.array([2, 5, 10, 20, 40, 60])
    n = 5000
    rw = np.cumsum(rng.normal(size=n))
    arx = ar1(rng, n - 1, 0.9)[0]
    # « tendance » à court terme : rendements AR(1) positivement autocorrélés
    rets = ar1(rng, n - 1, 0.3)[0]
    mom = np.cumsum(rets)
    th_ar = (1 - 0.9**qs) / (qs * (1 - 0.9))
    res = {"q": qs.tolist(),
           "VR_marche_aleatoire": [r3(vr(rw, q)) for q in qs],
           "VR_AR0.9_simule": [r3(vr(arx, q)) for q in qs],
           "VR_AR0.9_theorie": [r3(v) for v in th_ar],
           "VR_rendements_autocorreles": [r3(vr(mom, q)) for q in qs],
           "z_LoMacKinlay_q10": {"RW": r3(lo_mackinlay_z(rw, 10), 2), "AR0.9": r3(lo_mackinlay_z(arx, 10), 2),
                                 "momentum": r3(lo_mackinlay_z(mom, 10), 2)}}
    ch.record("E4", res)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(qs, res["VR_marche_aleatoire"], "o-", color=INK, label="marche aléatoire")
    ax.plot(qs, res["VR_AR0.9_simule"], "s-", color=GREEN, label="AR(1), φ = 0,9 (simulé)")
    ax.plot(qs, th_ar, "--", color=GREEN, lw=0.8, label="AR(1) théorie")
    ax.plot(qs, res["VR_rendements_autocorreles"], "^-", color=ACCENT, label="rendements autocorrélés")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xlabel("horizon q")
    ax.set_ylabel("VR(q)")
    ax.set_title("Ratio de variance")
    ax.legend(frameon=False, fontsize=7)
    ch.save_fig(fig, "ratio_variance")


# ---------------------------------------------------------------- E5
def hurst(x, lags=range(2, 50)):
    lags = np.array(list(lags))
    tau = [np.std(x[l:] - x[:-l]) for l in lags]
    slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return float(slope)


def e5():
    rng = np.random.default_rng(35)
    n, reps = 2000, 200
    hs = {"marche_aleatoire": [], "AR_0.9": [], "AR_0.99": [], "rendements_autocorreles_0.3": []}
    for _ in range(reps):
        hs["marche_aleatoire"].append(hurst(np.cumsum(rng.normal(size=n))))
        hs["AR_0.9"].append(hurst(ar1(rng, n - 1, 0.9)[0]))
        hs["AR_0.99"].append(hurst(ar1(rng, n - 1, 0.99)[0]))
        hs["rendements_autocorreles_0.3"].append(hurst(np.cumsum(ar1(rng, n - 1, 0.3)[0])))
    ch.record("E5", {k: {"moyenne": r3(np.mean(v)), "q05": r3(np.quantile(v, 0.05)), "q95": r3(np.quantile(v, 0.95))}
                     for k, v in hs.items()})


# ---------------------------------------------------------------- E6
def describe(spread, label, reg="c"):
    adf = adfuller(spread, autolag="AIC", regression=reg)
    kp = kpss(spread, regression=reg, nlags="auto")
    b = np.polyfit(spread[:-1], spread[1:], 1)[0]
    hl = -np.log(2) / np.log(b) if 0 < b < 1 else float("inf")
    return {"serie": label, "ADF_stat": r3(adf[0], 2), "ADF_p": r3(adf[1], 3), "KPSS_stat": r3(kp[0], 3),
            "KPSS_p": r3(kp[1], 3), "phi": r3(b, 4), "demi_vie_jours": r3(hl, 1),
            "VR_10": r3(vr(spread, 10), 3), "Hurst": r3(hurst(spread, range(2, 30)), 3)}


def e6():
    gld, gldm = load_close("GLD"), load_close("GLDM")
    ko, pep = load_close("KO"), load_close("PEP")
    s_gold = np.log(gld) - np.log(gldm)
    s_gold = s_gold - s_gold.mean()
    b = np.polyfit(np.log(pep), np.log(ko), 1)[0]
    s_kp = np.log(ko) - b * np.log(pep)
    s_kp = s_kp - s_kp.mean()
    s_kp1 = np.log(ko) - np.log(pep)
    res = [describe(np.log(gld), "log GLD (un prix seul)"),
           describe(s_gold, "log GLD - log GLDM"),
           describe(s_gold, "log GLD - log GLDM, tests avec tendance", reg="ct"),
           describe(s_gold - np.polyval(np.polyfit(np.arange(len(s_gold)), s_gold, 1), np.arange(len(s_gold))),
                    "log GLD - log GLDM sans tendance"),
           describe(s_kp1 - s_kp1.mean(), "log KO - log PEP (beta = 1)"),
           describe(s_kp, f"log KO - {b:.3f} log PEP (beta MCO)")]
    # tendance déterministe du ratio GLD/GLDM : frais de gestion différents
    t = np.arange(len(gld)) / 252
    slope = np.polyfit(t, np.log(gld / gldm), 1)[0]
    tt = np.arange(len(s_gold))
    det = s_gold - np.polyval(np.polyfit(tt, s_gold, 1), tt)
    ch.record("E6", {"tableau": res, "pente_annuelle_log_ratio_GLD_GLDM": r3(slope, 5),
                     "ecart_GLD_GLDM_sans_tendance_pb": {"ecart_type": r3(1e4 * det.std(), 2),
                                                         "max_abs": r3(1e4 * np.abs(det).max(), 1),
                                                         "q99_abs": r3(1e4 * np.quantile(np.abs(det), 0.99), 2),
                                                         "part_sup_5pb": r3((np.abs(det) > 5e-4).mean(), 3)}})

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    axes[0].plot(1e4 * s_gold, color=INK, lw=0.7)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title("log GLD − log GLDM (pb, centré)")
    axes[0].set_xlabel("séance")
    axes[1].plot(s_kp1 - s_kp1.mean(), color=ACCENT, lw=0.8)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("log KO − log PEP (centré)")
    axes[1].set_xlabel("séance")
    fig.tight_layout()
    ch.save_fig(fig, "cas_reels_ecarts")


if __name__ == "__main__":
    e1(); e2(); e3(); e4(); e5(); e6()
    ch.save_results()
    import json
    print(json.dumps(ch.results, indent=1, ensure_ascii=False))
