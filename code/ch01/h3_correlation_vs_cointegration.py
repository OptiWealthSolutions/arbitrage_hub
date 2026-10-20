"""H3 — Corrélation des rendements et cointégration sont deux choses différentes.

Deux paires simulées sur 1 000 jours :

  Paire A, « très corrélée mais pas cointégrée » :
      X_t = X_{t-1} + u_t,   Y_t = Y_{t-1} + v_t,   Corr(u_t, v_t) = 0.9
      Les deux prix bougent ensemble chaque jour, mais aucun mécanisme ne les ramène
      l'un vers l'autre : l'écart Y - X est lui-même une marche aléatoire.

  Paire B, « peu corrélée mais cointégrée » :
      X_t = X_{t-1} + u_t,   Y_t = X_t + s_t,   s_t AR(1) avec phi = 0.5, écart-type élevé
      Les rendements quotidiens sont dominés par le bruit de s_t (corrélation faible),
      mais l'écart Y - X = s_t est stationnaire : il revient toujours vers 0.

On mesure pour chacune la corrélation des rendements et la p-valeur du test ADF sur
l'écart (test vu au chapitre 3 ; ici on l'utilise comme boîte noire). Puis on répète
l'expérience 500 fois pour ne pas conclure sur un seul tirage.
"""
from __future__ import annotations

import warnings

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

from common import INK, ACCENT, ar1, save_fig, save_results

warnings.filterwarnings("ignore")
SEED = 3
N = 1_000


def pair_a(rng):
    cov = np.array([[1.0, 0.9], [0.9, 1.0]])
    shocks = rng.multivariate_normal([0, 0], cov, size=N)
    x = 100 + np.cumsum(shocks[:, 0])
    y = 100 + np.cumsum(shocks[:, 1])
    return x, y


def pair_b(rng):
    x = 100 + np.cumsum(rng.normal(0, 1.0, N))
    s = ar1(rng, N - 1, phi=0.5, sigma=2.0)[0]
    y = x + s
    return x, y


def stats(x, y):
    rx, ry = np.diff(x), np.diff(y)
    corr = float(np.corrcoef(rx, ry)[0, 1])
    spread = y - x
    pval = float(adfuller(spread, autolag="AIC")[1])
    return corr, pval, spread


def main() -> dict:
    rng = np.random.default_rng(SEED)
    xa, ya = pair_a(rng)
    xb, yb = pair_b(rng)
    ca, pa, sa = stats(xa, ya)
    cb, pb, sb = stats(xb, yb)

    reps = {"A": [], "B": []}
    for _ in range(500):
        reps["A"].append(stats(*pair_a(rng))[:2])
        reps["B"].append(stats(*pair_b(rng))[:2])
    summary = {}
    for k, v in reps.items():
        arr = np.array(v)
        summary[k] = {
            "corr_rendements_moyenne": round(float(arr[:, 0].mean()), 3),
            "part_ADF_rejette_5pct": round(float((arr[:, 1] < 0.05).mean()), 3),
        }

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.0), sharex=True)
    axes[0, 0].plot(xa, color=INK, lw=0.9, label="X")
    axes[0, 0].plot(ya, color=ACCENT, lw=0.9, label="Y")
    axes[0, 0].set_title(f"Paire A : corr. rendements = {ca:.2f}")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].plot(sa, color=INK, lw=0.9)
    axes[1, 0].axhline(0, color="k", lw=0.5)
    axes[1, 0].set_title(f"Écart A (Y−X) : p-valeur ADF = {pa:.2f}")
    axes[1, 0].set_xlabel("jour")
    axes[0, 1].plot(xb, color=INK, lw=0.9, label="X")
    axes[0, 1].plot(yb, color=ACCENT, lw=0.6, alpha=0.8, label="Y")
    axes[0, 1].set_title(f"Paire B : corr. rendements = {cb:.2f}")
    axes[0, 1].legend(frameon=False)
    axes[1, 1].plot(sb, color=INK, lw=0.6)
    axes[1, 1].axhline(0, color="k", lw=0.5)
    axes[1, 1].set_title(f"Écart B (Y−X) : p-valeur ADF = {pb:.3f}")
    axes[1, 1].set_xlabel("jour")
    fig.tight_layout()
    save_fig(fig, "h3_correlation_vs_cointegration")

    res = {
        "seed": SEED, "n": N,
        "tirage_illustre": {"A": {"corr": round(ca, 3), "adf_p": round(pa, 4)},
                            "B": {"corr": round(cb, 3), "adf_p": round(pb, 4)}},
        "sur_500_tirages": summary,
    }
    save_results("h3_correlation_vs_cointegration", res)
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, ensure_ascii=False))
