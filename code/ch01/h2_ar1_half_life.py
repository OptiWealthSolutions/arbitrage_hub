"""H2 — Propriétés de l'AR(1) et demi-vie du retour à la moyenne.

Hypothèses testées (chapitre 1, section AR(1)) pour X_t = phi X_{t-1} + eps_t, |phi| < 1 :
  (a) Var(X_t) = sigma^2 / (1 - phi^2)            (variance stationnaire)
  (b) Corr(X_t, X_{t-h}) = phi^h                    (autocorrélation géométrique)
  (c) E[X_t | X_0 = x0] = phi^t x0, donc l'écart moyen est divisé par 2 après
      h_{1/2} = -ln 2 / ln phi périodes             (demi-vie)
  (d) Sur un échantillon court, l'estimateur MCO de phi est biaisé vers le bas,
      donc la demi-vie estimée est biaisée vers le bas et très dispersée.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import INK, ACCENT, MUTED, ar1, save_fig, save_results, ols_slope_intercept

SEED = 2
PHI, SIGMA = 0.95, 1.0
HALF_LIFE_TH = -np.log(2) / np.log(PHI)


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    denom = np.dot(x, x)
    return np.array([np.dot(x[h:], x[: len(x) - h]) / denom for h in range(max_lag + 1)])


def estimate_phi(x: np.ndarray) -> float:
    b, _ = ols_slope_intercept(x[1:], x[:-1])
    return b


def main() -> dict:
    rng = np.random.default_rng(SEED)

    # (a)(b) une longue trajectoire
    long = ar1(rng, 200_000, PHI, SIGMA)[0]
    var_emp = float(long.var())
    var_th = SIGMA**2 / (1 - PHI**2)
    lags = 60
    acf_emp = acf(long, lags)
    acf_th = PHI ** np.arange(lags + 1)

    # (c) décroissance moyenne depuis un écart initial x0 = 10
    x0 = 10.0
    paths = ar1(rng, 60, PHI, SIGMA, x0=x0, n_paths=20_000)
    mean_path = paths.mean(axis=0)
    t = np.arange(61)
    crossing = float(np.interp(x0 / 2, mean_path[::-1], t[::-1]))  # premier t où E[X_t] = x0/2

    # (d) biais de petit échantillon
    out_small = {}
    for n in (250, 1000, 5000):
        phis = np.array([estimate_phi(ar1(rng, n, PHI, SIGMA)[0]) for _ in range(2000)])
        phis_ok = phis[(phis > 0) & (phis < 1)]
        hl = -np.log(2) / np.log(phis_ok)
        out_small[n] = {
            "phi_moyen": round(float(phis.mean()), 4),
            "biais_phi": round(float(phis.mean() - PHI), 4),
            "demi_vie_mediane": round(float(np.median(hl)), 2),
            "demi_vie_q05": round(float(np.quantile(hl, 0.05)), 2),
            "demi_vie_q95": round(float(np.quantile(hl, 0.95)), 2),
        }
        if n == 250:
            hl_250 = hl

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    ax = axes[0]
    ax.bar(np.arange(lags + 1), acf_emp, color=MUTED, width=0.8, label="empirique")
    ax.plot(np.arange(lags + 1), acf_th, color=ACCENT, lw=1.3, label=r"$\varphi^h$")
    ax.set_title("Autocorrélation")
    ax.set_xlabel("retard h")
    ax.legend(frameon=False)

    ax = axes[1]
    for i in range(30):
        ax.plot(t, paths[i], color=MUTED, lw=0.4, alpha=0.5)
    ax.plot(t, mean_path, color=INK, lw=1.6, label="moyenne simulée")
    ax.plot(t, x0 * PHI**t, color=ACCENT, lw=1.0, ls="--", label=r"$\varphi^t x_0$")
    ax.axhline(x0 / 2, color="k", lw=0.6, ls=":")
    ax.axvline(HALF_LIFE_TH, color="k", lw=0.6, ls=":")
    ax.set_title(f"Retour depuis $x_0={x0:g}$")
    ax.set_xlabel("t")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[2]
    ax.hist(np.clip(hl_250, 0, 60), bins=40, color=INK, alpha=0.85)
    ax.axvline(HALF_LIFE_TH, color=ACCENT, lw=1.3, label=f"vraie demi-vie {HALF_LIFE_TH:.1f}")
    ax.set_title("Demi-vie estimée, n = 250")
    ax.set_xlabel("demi-vie (périodes)")
    ax.legend(frameon=False)
    save_fig(fig, "h2_ar1_half_life")

    res = {
        "phi": PHI, "sigma": SIGMA, "seed": SEED,
        "variance_empirique": round(var_emp, 3), "variance_theorique": round(var_th, 3),
        "acf_lag1_emp": round(float(acf_emp[1]), 4), "acf_lag10_emp": round(float(acf_emp[10]), 4),
        "acf_lag10_th": round(float(acf_th[10]), 4),
        "demi_vie_theorique": round(float(HALF_LIFE_TH), 2),
        "demi_vie_observee_sur_moyenne": round(crossing, 2),
        "petits_echantillons": out_small,
    }
    save_results("h2_ar1_half_life", res)
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, ensure_ascii=False))
