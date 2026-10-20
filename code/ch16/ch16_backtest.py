"""Chapitre 16 — Backtest rigoureux : anticipation, erreur d'estimation, tests multiples.

Toutes les expériences utilisent le moteur et les mesures du paquet (src/arbhub/backtest).

E1  Trois façons de se tromper sans le savoir : un écart SANS avantage (marche aléatoire)
    paraît rentable si l'on applique la position le jour même ou si l'on normalise le z-score
    avec la moyenne et l'écart-type de toute la période.
E2  Erreur-type du ratio de Sharpe : dispersion de l'estimateur selon la durée du backtest,
    avec rendements normaux puis à queues épaisses (Student à 4 degrés de liberté).
E3  Maximum de N ratios de Sharpe de stratégies sans avantage, comparé à la formule de
    Bailey et López de Prado (2014).
E4  Optimisation d'une grille de 50 variantes sur la première moitié de l'historique :
    Sharpe dans l'échantillon, hors échantillon et ratio de Sharpe déflaté (DSR),
    dans un monde sans avantage et dans un monde avec avantage.
E5  Probabilité de surapprentissage (PBO) par validation croisée combinatoire (CSCV)
    sur les mêmes grilles.
E6  Biais de survie : backtest sur toutes les paires contre backtest sur les seules paires
    encore cotées à la fin.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "common"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, r3
from arbhub.backtest.engine import pair_backtest, rolling_zscore, zscore_positions
from arbhub.backtest.metrics import (annualized_sharpe, deflated_sharpe, expected_max_sharpe,
                                     pbo_cscv, sharpe, sharpe_se)

ch = Chapter("ch16")
A = np.sqrt(252)


def world(rng, n, phi, beta=1.5, sig_s=1.0):
    """x marche aléatoire autour de 100 ; y = beta x + s, s AR(1) (phi=1 : pas d'avantage)."""
    x = 100 + np.cumsum(rng.normal(0, 1.0, n))
    s = np.zeros(n)
    e = rng.normal(0, sig_s, n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + e[t]
    return beta * x + s + 50, x


# ---------------------------------------------------------------- E1
def e1(rng, n_worlds=300, n=1000, window=60):
    out = {"correct": [], "meme_jour": [], "normalisation_globale": [], "beta_global": []}
    for _ in range(n_worlds):
        y, x = world(rng, n, phi=1.0)
        s = y - 1.5 * x
        q = zscore_positions(rolling_zscore(s, window), 2.0, 0.0)
        r = pair_backtest(y, x, 1.5, q, cost_bps=0)
        out["correct"].append(annualized_sharpe(r.net[window:]))
        # bogue 1 : la position décidée en t est appliquée à la variation de t-1 à t
        same_day = np.concatenate([[0.0], q[1:] * np.diff(s)])
        out["meme_jour"].append(annualized_sharpe(same_day[window:]))
        # bogue 2 : z-score avec la moyenne et l'écart-type de TOUTE la période
        zg = (s - s.mean()) / s.std()
        qg = zscore_positions(zg, 2.0, 0.0)
        rg = pair_backtest(y, x, 1.5, qg, cost_bps=0)
        out["normalisation_globale"].append(annualized_sharpe(rg.net))
        # bogue 3 : ratio de couverture estimé par MCO sur TOUTE la période (z glissant correct)
        b, a = np.polyfit(x, y, 1)
        qb = zscore_positions(rolling_zscore(y - b * x, window), 2.0, 0.0)
        rb = pair_backtest(y, x, b, qb, cost_bps=0)
        out["beta_global"].append(annualized_sharpe(rb.net[window:]))
    # le bogue « jour même » dans un monde AVEC avantage (phi = 0,97)
    edge = {"correct": [], "meme_jour": []}
    for _ in range(n_worlds):
        y, x = world(rng, n, phi=0.97)
        s = y - 1.5 * x
        q = zscore_positions(rolling_zscore(s, window), 2.0, 0.0)
        edge["correct"].append(annualized_sharpe(pair_backtest(y, x, 1.5, q).net[window:]))
        edge["meme_jour"].append(annualized_sharpe(np.concatenate([[0.0], q[1:] * np.diff(s)])[window:]))
    res = {k: {"sharpe_median": r3(np.median(v), 2), "part_positive": r3(np.mean(np.array(v) > 0), 3)}
           for k, v in out.items()}
    res["monde_avec_avantage"] = {k: {"sharpe_median": r3(np.median(v), 2)} for k, v in edge.items()}
    return res, out


# ---------------------------------------------------------------- E2
def e2(rng, n_sims=4000):
    sr_d = 1.0 / A                               # Sharpe annuel vrai = 1
    rows = []
    for years in (1, 3, 5, 10):
        T = 252 * years
        normal = rng.normal(sr_d, 1.0, (n_sims, T))
        t4 = rng.standard_t(4, (n_sims, T)) / np.sqrt(2.0) + sr_d    # variance 1
        sd_n = float((normal.mean(1) / normal.std(1, ddof=1) * A).std())
        sd_t = float((t4.mean(1) / t4.std(1, ddof=1) * A).std())
        rows.append({"annees": years,
                     "ecart_type_normal": r3(sd_n), "formule_lo": r3(sharpe_se(sr_d, T) * A),
                     "ecart_type_student4": r3(sd_t),
                     "intervalle_95_normal": [r3(1 - 1.96 * sd_n, 2), r3(1 + 1.96 * sd_n, 2)]})
    return rows


# ---------------------------------------------------------------- E3
def e3(rng, T=756, reps=200):
    rows = []
    for N in (1, 5, 10, 50, 100, 500, 1000):
        mx = []
        for _ in range(reps):
            R = rng.normal(0, 1, (T, N))
            srs = R.mean(0) / R.std(0, ddof=1)
            mx.append(srs.max() * A)
        rows.append({"N": N, "max_moyen_simule": r3(np.mean(mx), 3),
                     "formule": r3(expected_max_sharpe(N, 1.0 / (T - 1)) * A, 3) if N > 1 else 0.0})
    return rows


# ---------------------------------------------------------------- E4 / E5
GRID = [(e, w, xt) for e in (1.0, 1.5, 2.0, 2.5, 3.0) for w in (20, 40, 60, 120, 250) for xt in (0.0, 0.5)]


def grid_returns(y, x):
    s = y - 1.5 * x
    cols = []
    for entry, w, xt in GRID:
        q = zscore_positions(rolling_zscore(s, w), entry, xt)
        r = pair_backtest(y, x, 1.5, q, cost_bps=5)
        cols.append(r.net)
    return np.column_stack(cols)[250:]          # on écarte le démarrage des fenêtres


def e4_e5(rng, n_worlds=60, n=1750):
    res = {}
    keep = {}
    for label, phi in (("sans_avantage", 1.0), ("avec_avantage", 0.97)):
        is_best, oos_of_best, dsr, pbo, oos_median = [], [], [], [], []
        for k in range(n_worlds):
            y, x = world(rng, n, phi=phi, sig_s=1.0)
            R = grid_returns(y, x)
            half = R.shape[0] // 2
            Ris, Roos = R[:half], R[half:]
            srs = np.array([sharpe(Ris[:, j]) for j in range(R.shape[1])])
            j = int(np.argmax(srs))
            is_best.append(srs[j] * A)
            oos_of_best.append(sharpe(Roos[:, j]) * A)
            oos_median.append(np.median([sharpe(Roos[:, i]) for i in range(R.shape[1])]) * A)
            dsr.append(deflated_sharpe(Ris[:, j], len(GRID), float(np.var(srs, ddof=1))))
            if k < 60:
                p, logits = pbo_cscv(R, n_blocks=12)
                pbo.append(p)
                keep.setdefault(label, []).extend(list(logits))
        res[label] = {"sharpe_IS_meilleure": r3(np.mean(is_best), 2),
                      "sharpe_OOS_meilleure": r3(np.mean(oos_of_best), 2),
                      "sharpe_OOS_mediane_grille": r3(np.mean(oos_median), 2),
                      "dsr_moyen": r3(np.mean(dsr), 3),
                      "part_dsr_sup_0_95": r3(np.mean(np.array(dsr) > 0.95), 3),
                      "pbo_moyen": r3(np.mean(pbo), 3),
                      "is_best": [r3(v, 3) for v in is_best],
                      "oos_best": [r3(v, 3) for v in oos_of_best]}
    return res, keep


# ---------------------------------------------------------------- E6
def e6(rng, n_pairs=400, n=1500, form=500, p_break=0.3):
    all_sr, surv_sr, all_pnl, surv_pnl = [], [], [], []
    n_surv = 0
    for _ in range(n_pairs):
        x = 100 + np.cumsum(rng.normal(0, 1.0, n))
        s = np.zeros(n)
        broken = rng.random() < p_break
        tb = rng.integers(form + 100, n - 200) if broken else n
        drift = rng.choice([-1, 1]) * 0.15
        for t in range(1, n):
            if t < tb:
                s[t] = 0.9 * s[t - 1] + rng.normal()
            else:
                s[t] = s[t - 1] + drift + rng.normal()   # divergence durable (faillite, rachat...)
        y = 1.5 * x + s + 50
        m, d = s[:form].mean(), s[:form].std(ddof=1)
        z = (s - m) / d
        q = zscore_positions(z, 2.0, 0.0)
        q[:form] = 0.0
        r = pair_backtest(y, x, 1.5, q, cost_bps=5)
        if broken:
            # l'action défaillante est radiée 100 jours après la rupture : on solde à cette date
            cut = min(tb + 100, n)
            net = r.net[form:cut]
        else:
            net = r.net[form:]
        all_sr.append(annualized_sharpe(net))
        all_pnl.append(net.sum())
        if not broken:
            n_surv += 1
            surv_sr.append(annualized_sharpe(net))
            surv_pnl.append(net.sum())
    return {"paires": n_pairs, "survivantes": n_surv,
            "toutes": {"pnl_moyen": r3(np.mean(all_pnl), 2), "sharpe_median": r3(np.median(all_sr), 2)},
            "survivantes_seules": {"pnl_moyen": r3(np.mean(surv_pnl), 2),
                                   "sharpe_median": r3(np.median(surv_sr), 2)}}


def main():
    rng = np.random.default_rng(16)
    r1, raw1 = e1(rng)
    ch.record("E1", r1)
    r2 = e2(rng)
    ch.record("E2", r2)
    r3_ = e3(rng)
    ch.record("E3", r3_)
    r45, logits = e4_e5(rng)
    ch.record("E4_E5", {k: {kk: vv for kk, vv in v.items() if kk not in ("is_best", "oos_best")}
                        for k, v in r45.items()})
    ch.record("E6", e6(rng))
    ch.record("grille", {"variantes": len(GRID), "seuils": [1.0, 1.5, 2.0, 2.5, 3.0],
                         "fenetres": [20, 40, 60, 120, 250], "sorties": [0.0, 0.5], "cout_pb": 5})
    ch.save_results()

    # Figure 1 : E1
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    labels = {"correct": "correct", "beta_global": "$\\beta$ estimé sur\ntoute la période",
              "normalisation_globale": "normalisation\nglobale", "meme_jour": "position\nle jour même"}
    data = [raw1[k] for k in labels]
    bp = ax.boxplot(data, orientation="horizontal", tick_labels=list(labels.values()), widths=0.55, patch_artist=True,
                    medianprops=dict(color=ACCENT))
    for p, c in zip(bp["boxes"], (INK, GREEN, GOLD, ACCENT)):
        p.set_facecolor(c); p.set_alpha(0.35)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Sharpe annualisé (écart sans avantage, 300 mondes)")
    ch.save_fig(fig, "biais_anticipation")

    # Figure 2 : E3
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    Ns = [row["N"] for row in r3_]
    ax.plot(Ns, [row["max_moyen_simule"] for row in r3_], "o", color=INK, label="simulation")
    ax.plot(Ns, [row["formule"] for row in r3_], "-", color=ACCENT, label="formule")
    ax.set_xscale("log")
    ax.set_xlabel("nombre de variantes essayées N")
    ax.set_ylabel("Sharpe annualisé maximal")
    ax.legend(frameon=False)
    ch.save_fig(fig, "max_sharpe")

    # Figure 3 : E4 + E5
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for lab, c in (("sans_avantage", ACCENT), ("avec_avantage", GREEN)):
        axes[0].scatter(r45[lab]["is_best"], r45[lab]["oos_best"], s=9, color=c, alpha=0.7,
                        label=lab.replace("_", " "))
        axes[1].hist(logits[lab], bins=40, color=c, alpha=0.55, label=lab.replace("_", " "), density=True)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("Sharpe de la meilleure variante (dans l'échantillon)")
    axes[0].set_ylabel("Sharpe hors échantillon")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].axvline(0, color="k", lw=0.7)
    axes[1].set_xlabel("logit du rang hors échantillon")
    axes[1].set_title("CSCV : logit $\\leq 0$ = surapprentissage")
    fig.tight_layout()
    ch.save_fig(fig, "is_oos_pbo")
    return ch.results


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, ensure_ascii=False))
