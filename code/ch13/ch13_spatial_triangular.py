"""Chapitre 13 — Arbitrage spatial et triangulaire.

E1  Course à la latence : part des opportunités capturées et gain selon notre latence,
    face à des concurrents colocalisés.
E2  Arbitrage triangulaire : fréquence des opportunités nettes selon la fourchette, les frais
    et le bruit des cotations.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, r3

ch = Chapter("ch13")


def e1():
    rng = np.random.default_rng(131)
    n = 200_000                                      # opportunités simulées
    edge = rng.exponential(3.0, n)                   # gain net de frais, en pb
    dur = rng.exponential(0.050, n)                  # durée de vie de l'écart : 50 ms en moyenne
    comp = rng.lognormal(np.log(0.002), 0.5, (n, 5)).min(1)   # 5 concurrents, ~2 ms
    rows = []
    for lat in (0.001, 0.002, 0.005, 0.02, 0.1, 0.5):
        win = (lat < dur) & (lat < comp)
        rows.append({"latence_ms": lat * 1000, "part_capturee": r3(win.mean(), 4),
                     "gain_moyen_par_opportunite_pb": r3((edge * win).mean(), 3)})
    # opportunités lentes (minutes) mais avec coût de transfert / capital immobilisé
    slow = {"duree_moyenne_s": 300, "part_capturee_latence_500ms": r3(np.mean(rng.exponential(300, 50000) > 0.5), 3)}
    ch.record("E1", {"concurrents": 5, "duree_moyenne_ms": 50, "tableau": rows, "opportunites_lentes": slow})
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ax.plot([r["latence_ms"] for r in rows], [100 * r["part_capturee"] for r in rows], "o-", color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("notre latence (ms, échelle log)")
    ax.set_ylabel("opportunités capturées (%)")
    ax.set_title("Course à la latence : le plus rapide prend tout")
    ch.save_fig(fig, "course_latence")


def e2():
    rng = np.random.default_rng(132)
    n = 1_000_000
    out = []
    for half_spread in (0.5, 1.0):
        for fee in (0.0, 0.5, 2.0):
            for noise in (0.2, 1.0):
                # bruit indépendant sur le prix milieu de chacune des 3 cotations (en pb)
                eps = rng.normal(0, noise, (n, 3))
                # cycle USD -> EUR -> JPY -> USD : on paie la demi-fourchette et les frais à chaque étape
                gross = eps.sum(1) - 3 * half_spread - 3 * fee
                gross2 = -eps.sum(1) - 3 * half_spread - 3 * fee         # cycle inverse
                best = np.maximum(gross, gross2)
                theo = 2 * stats.norm.sf((3 * half_spread + 3 * fee) / (noise * np.sqrt(3)))
                out.append({"demi_fourchette_pb": half_spread, "frais_pb": fee, "bruit_pb": noise,
                            "part_opportunites": r3((best > 0).mean(), 5), "theorie": r3(theo, 5),
                            "gain_moyen_si_positif_pb": r3(best[best > 0].mean(), 2) if (best > 0).any() else None})
    ch.record("E2", out)


if __name__ == "__main__":
    e1(); e2()
    ch.save_results()
