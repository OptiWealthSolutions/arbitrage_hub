"""H4 — Régression fallacieuse : deux marches aléatoires INDÉPENDANTES semblent liées.

Hypothèse testée (Granger et Newbold, 1974 ; chapitre 1, section « piège ») :
si l'on régresse un prix sur un autre prix alors que les deux sont des marches aléatoires
indépendantes, le test de Student usuel rejette « pente = 0 » bien plus souvent que 5 %,
et le R^2 est souvent élevé. En revanche, la même régression sur les RENDEMENTS
(différences premières) se comporte normalement (environ 5 % de rejets).

Protocole : 2 000 répétitions, n = 500 observations.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import INK, ACCENT, save_fig, save_results

SEED = 4
N, REPS = 500, 2_000


def ols_t_and_r2(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """t-stat de la pente (erreurs-types MCO classiques) et R^2 de y = a + b x + e."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    t = beta[1] / np.sqrt(cov[1, 1])
    r2 = 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2)
    return float(t), float(r2)


def main() -> dict:
    rng = np.random.default_rng(SEED)
    t_lvl, r2_lvl, t_ret, r2_ret = [], [], [], []
    for _ in range(REPS):
        x = np.cumsum(rng.normal(size=N))
        y = np.cumsum(rng.normal(size=N))  # indépendante de x par construction
        a, b = ols_t_and_r2(y, x)
        t_lvl.append(a); r2_lvl.append(b)
        a, b = ols_t_and_r2(np.diff(y), np.diff(x))
        t_ret.append(a); r2_ret.append(b)
    t_lvl, r2_lvl, t_ret, r2_ret = map(np.array, (t_lvl, r2_lvl, t_ret, r2_ret))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    bins = np.linspace(-60, 60, 61)
    axes[0].hist(np.clip(t_lvl, -60, 60), bins=bins, color=INK, alpha=0.85)
    axes[0].axvspan(-1.96, 1.96, color=ACCENT, alpha=0.25, label=r"$|t|<1{,}96$")
    axes[0].set_title("t de la pente : prix sur prix")
    axes[0].legend(frameon=False)
    axes[1].hist(t_ret, bins=40, color=INK, alpha=0.85)
    axes[1].axvspan(-1.96, 1.96, color=ACCENT, alpha=0.25)
    axes[1].set_title("t de la pente : rendements sur rendements")
    save_fig(fig, "h4_spurious_regression")

    res = {
        "seed": SEED, "n": N, "repetitions": REPS,
        "niveaux": {"part_rejet_5pct": round(float((np.abs(t_lvl) > 1.96).mean()), 3),
                    "R2_median": round(float(np.median(r2_lvl)), 3),
                    "part_R2_sup_0.3": round(float((r2_lvl > 0.3).mean()), 3)},
        "rendements": {"part_rejet_5pct": round(float((np.abs(t_ret) > 1.96).mean()), 3),
                       "R2_median": round(float(np.median(r2_ret)), 4)},
    }
    save_results("h4_spurious_regression", res)
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, ensure_ascii=False))
