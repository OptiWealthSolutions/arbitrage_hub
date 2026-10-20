"""Chapitre 12 — Microstructure et coûts.

E1  Estimateur de Roll : fourchette retrouvée sur transactions simulées ; échec sur clôtures quotidiennes réelles.
E2  Modèle de coût complet d'un aller-retour de paire et avantage minimal (hypothèses de frais explicites).
E3  Hypothèses d'exécution : même clôture, clôture suivante, demi-fourchette payée -> effet sur un écart OU.
E4  Loi d'impact en racine carrée : ordre de grandeur pour différentes tailles d'ordre.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib.pyplot as plt

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, load_close, r3

ch = Chapter("ch12")


# ---------------------------------------------------------------- E1
def roll(p):
    dp = np.diff(p)
    c = np.cov(dp[1:], dp[:-1])[0, 1]
    return (2 * np.sqrt(-c), c) if c < 0 else (np.nan, c)


def e1():
    rng = np.random.default_rng(121)
    out = {}
    for s_bp in (2, 10, 50):
        mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.0002, 200000)))       # bruit fondamental par transaction
        q = rng.choice([-1, 1], size=len(mid))
        s = mid * s_bp / 1e4
        p = mid + q * s / 2
        est, _ = roll(p)
        out[f"vraie_fourchette_{s_bp}pb"] = r3(1e4 * est / 100, 2)
    real = {}
    for sym in ("GLD", "GLDM", "KO", "PEP"):
        p = load_close(sym)
        est, c = roll(p)
        real[sym] = {"autocovariance": r3(c, 4), "fourchette_estimee_pb": None if np.isnan(est) else r3(1e4 * est / p.mean(), 1)}
    ch.record("E1", {"simulation_estimation_pb": out, "clotures_quotidiennes_reelles": real})


# ---------------------------------------------------------------- E2
# Hypothèses de coûts ILLUSTRATIVES (à remplacer par vos propres conditions tarifaires)
SCENARIOS = {
    "actions_US_courtier_bas_cout": {"commission_pb": 0.5, "demi_fourchette_pb": 1.0, "impact_pb": 0.5, "emprunt_annuel_pct": 0.3},
    "actions_moyennes_capitalisations": {"commission_pb": 1.0, "demi_fourchette_pb": 5.0, "impact_pb": 3.0, "emprunt_annuel_pct": 1.0},
    "crypto_taker_taker": {"commission_pb": 5.0, "demi_fourchette_pb": 1.0, "impact_pb": 1.0, "emprunt_annuel_pct": 0.0},
    "crypto_maker_CFD_taker": {"commission_pb": 1.5, "demi_fourchette_pb": 1.5, "impact_pb": 0.5, "emprunt_annuel_pct": 0.0},
}


def round_trip_cost_bp(sc, holding_days):
    per_leg_per_side = sc["commission_pb"] + sc["demi_fourchette_pb"] + sc["impact_pb"]
    trading = 2 * 2 * per_leg_per_side                       # 2 jambes x (entrée + sortie)
    borrow = sc["emprunt_annuel_pct"] * 100 * holding_days / 360   # sur la jambe courte (pb du notionnel d'une jambe)
    return trading, borrow


def e2():
    out = {}
    for name, sc in SCENARIOS.items():
        tr, bo = round_trip_cost_bp(sc, 10)
        tot = tr + bo
        # en pb du notionnel d'UNE jambe ; un aller-retour capture ~2 sigma_eq de l'écart
        out[name] = {"cout_trading_aller_retour_pb": r3(tr, 1), "cout_emprunt_10j_pb": r3(bo, 2), "total_pb": r3(tot, 1),
                     "sigma_eq_minimal_pour_couvrir_2x_les_couts_pb": r3(tot, 1)}
    ch.record("E2", {"hypotheses": SCENARIOS, "resultats": out,
                     "note": "un aller-retour à seuils +-2 sigma -> 0 capture ~2 sigma_eq ; exiger gain >= 2 x coûts => sigma_eq >= coûts"})


# ---------------------------------------------------------------- E3
def e3():
    rng = np.random.default_rng(123)
    phi, n = np.exp(-np.log(2) / 5), 252 * 50                 # demi-vie 5 jours
    sd_eq_bp = 30.0
    s = np.zeros(n)
    e = rng.normal(0, sd_eq_bp * np.sqrt(1 - phi**2), n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + e[t]
    out = {}
    for name, delay, half_spread in [("meme_cloture_sans_frais", 0, 0.0), ("meme_cloture_demi_fourchette_2pb", 0, 2.0),
                                     ("cloture_suivante_sans_frais", 1, 0.0), ("cloture_suivante_demi_fourchette_2pb", 1, 2.0)]:
        sig = np.zeros(n)
        pos = 0
        for t in range(n):
            if pos == 0 and abs(s[t]) > 2 * sd_eq_bp:
                pos = -np.sign(s[t])
            elif pos != 0 and np.sign(s[t]) == pos:
                pos = 0
            sig[t] = pos
        held = np.r_[np.zeros(delay), sig[:-delay]] if delay else sig
        pnl = held[:-1] * np.diff(s)
        trades = np.abs(np.diff(np.r_[0, held]))
        pnl = pnl - trades[1:] * 2 * half_spread              # 2 jambes
        out[name] = {"gain_annuel_pb": r3(pnl.sum() / 50, 1), "sharpe": r3(np.sqrt(252) * pnl.mean() / pnl.std(), 2)}
    ch.record("E3", {"demi_vie_jours": 5, "sigma_eq_pb": sd_eq_bp, "resultats": out})


# ---------------------------------------------------------------- E4
def e4():
    sigma_d, Y = 0.015, 1.0
    rows = []
    for frac in (0.0001, 0.001, 0.01, 0.05):
        rows.append({"part_du_volume_quotidien": frac, "impact_pb": r3(1e4 * Y * sigma_d * np.sqrt(frac), 2)})
    ch.record("E4", {"volatilite_quotidienne": sigma_d, "Y": Y, "tableau": rows})


if __name__ == "__main__":
    e1(); e2(); e3(); e4()
    ch.save_results()
