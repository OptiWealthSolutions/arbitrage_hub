"""H1 — La variance d'une marche aléatoire croît linéairement avec le temps.

Hypothèse testée (chapitre 1, proposition « marche aléatoire ») :
    P_t = P_0 + eps_1 + ... + eps_t, eps_i i.i.d. N(0, sigma^2)
    => E[P_t] = P_0   et   Var(P_t) = t * sigma^2.

Protocole : 20 000 trajectoires de 1 000 pas, sigma = 1. On mesure la variance
transversale (entre trajectoires) à chaque date t, puis on régresse log Var(P_t)
sur log t : la pente doit valoir 1 (croissance linéaire) et l'ordonnée 0 (sigma^2 = 1).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import INK, ACCENT, MUTED, random_walk, save_fig, save_results, ols_slope_intercept

SEED = 1
N_PATHS, N_STEPS, SIGMA = 20_000, 1_000, 1.0


def main() -> dict:
    rng = np.random.default_rng(SEED)
    paths = random_walk(rng, N_STEPS, N_PATHS, SIGMA)
    t = np.arange(N_STEPS + 1)
    var_t = paths.var(axis=0, ddof=1)
    mean_t = paths.mean(axis=0)

    checkpoints = [10, 100, 500, 1000]
    table = [{"t": int(k), "var_empirique": round(float(var_t[k]), 2),
              "var_theorique": float(k * SIGMA**2),
              "ecart_relatif_pct": round(100 * (var_t[k] / (k * SIGMA**2) - 1), 2)}
             for k in checkpoints]

    slope, intercept = ols_slope_intercept(np.log(var_t[1:]), np.log(t[1:]))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax = axes[0]
    for i in range(40):
        ax.plot(t, paths[i], color=MUTED, lw=0.5, alpha=0.6)
    band = 2 * np.sqrt(t) * SIGMA
    ax.plot(t, band, color=ACCENT, lw=1.2, label=r"$\pm 2\sigma\sqrt{t}$")
    ax.plot(t, -band, color=ACCENT, lw=1.2)
    ax.set_title("40 marches aléatoires")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$P_t$")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    ax.plot(t[1:], var_t[1:], color=INK, lw=1.4, label="variance empirique")
    ax.plot(t[1:], t[1:] * SIGMA**2, color=ACCENT, lw=1.0, ls="--", label=r"théorie $t\sigma^2$")
    ax.set_title("Variance transversale en fonction de t")
    ax.set_xlabel("t")
    ax.set_ylabel(r"Var$(P_t)$")
    ax.legend(frameon=False)
    save_fig(fig, "h1_random_walk_variance")

    res = {
        "hypothese": "Var(P_t) = t sigma^2 et E[P_t] = P_0",
        "seed": SEED, "n_paths": N_PATHS, "n_steps": N_STEPS,
        "points_de_controle": table,
        "pente_loglog": round(slope, 4),
        "ordonnee_loglog": round(intercept, 4),
        "moyenne_max_abs": round(float(np.abs(mean_t).max()), 3),
        "conclusion": "pente ~ 1 : croissance linéaire de la variance confirmée",
    }
    save_results("h1_random_walk_variance", res)
    return res


if __name__ == "__main__":
    out = main()
    for row in out["points_de_controle"]:
        print(row)
    print("pente log-log :", out["pente_loglog"], "| ordonnée :", out["ordonnee_loglog"])
    print("|moyenne| max sur t :", out["moyenne_max_abs"])
