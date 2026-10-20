"""Chapitre 15 — Exécution et couverture : maker sur une plateforme, couverture taker sur l'autre.

Modèle (pas de 50 ms, 4 heures) :
  - prix fondamental m_t (marche aléatoire, volatilité de l'or ~15 %/an) ;
  - plateforme de référence B (CFD) : mid = m_t, demi-fourchette h_B ;
  - plateforme de cotation A : mid = m_{t - lag} + écart AR(1) (bruit de base) ; lag > 0 : A suit B ;
    lag < 0 : A précède B ;
  - on affiche un ordre d'achat maker sur A au prix  bid_B(t - d_q) + niveau  (d_q = retard de recotation) ;
  - exécution quand le mid de A passe sous notre prix ; couverture : vente au bid de B après une latence tau.
Mesure : écart réalisé = prix de vente B - prix d'achat A, comparé au niveau visé (sélection adverse + glissement).

E1  Glissement moyen selon le retard de recotation et la latence de couverture (A suit B de 200 ms).
E2  Même chose quand A précède B (on cote sur la plateforme en retard).
E3  Risque de jambe d'une exécution taker-taker : variance de l'écart réalisé vs décalage entre les deux ordres.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib.pyplot as plt

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, r3

ch = Chapter("ch15")
DT = 0.05                       # secondes par pas
P0 = 4000.0
SIG_STEP = P0 * 0.15 / np.sqrt(252 * 6.5 * 3600 / DT)    # $ par pas
H_B = 0.15                      # demi-fourchette du CFD en $
LEVEL = -1.2                    # on achète 1,2 $ sous le bid de B (~3 pb)


def simulate(rng, steps, lag_steps):
    m = P0 + np.cumsum(rng.normal(0, SIG_STEP, steps))
    noise = np.zeros(steps)
    e = rng.normal(0, 0.35, steps)
    phi = np.exp(-DT / 20.0)                                  # bruit de base de demi-vie ~14 s
    for t in range(1, steps):
        noise[t] = phi * noise[t - 1] + np.sqrt(1 - phi**2) * e[t] * 3.0
    if lag_steps >= 0:
        a_mid = np.r_[np.full(lag_steps, m[0]), m[:steps - lag_steps]] + noise
    else:
        k = -lag_steps
        a_mid = np.r_[m[k:], np.full(k, m[-1])] + noise
    return m, a_mid


def fills_and_slippage(m, a_mid, dq_steps, tau_steps):
    b_bid = m - H_B
    steps = len(m)
    quote = np.r_[np.full(dq_steps, b_bid[0]), b_bid[:steps - dq_steps]] + LEVEL if dq_steps > 0 else b_bid + LEVEL
    below = a_mid <= quote
    entry = np.where(below[1:] & ~below[:-1])[0] + 1          # croisements vers le bas = exécutions
    entry = entry[entry + tau_steps < steps]
    realised = b_bid[entry + tau_steps] - quote[entry]          # on a acheté à quote, on vend au bid de B
    slip = realised - (-LEVEL)                                  # 0 = on a obtenu exactement le niveau visé
    return len(entry), slip


def grid(lag_ms):
    rng = np.random.default_rng(151)
    steps = int(4 * 3600 / DT)
    m, a = simulate(rng, steps, int(lag_ms / 1000 / DT))
    rows = []
    for dq in (0, 100, 500, 2000):
        for tau in (50, 500, 2000):
            n, s = fills_and_slippage(m, a, int(dq / 1000 / DT), int(tau / 1000 / DT))
            rows.append({"retard_recotation_ms": dq, "latence_couverture_ms": tau, "executions": n,
                         "glissement_moyen_dollars": r3(s.mean(), 3), "glissement_moyen_pb": r3(1e4 * s.mean() / P0, 2),
                         "ecart_type_pb": r3(1e4 * s.std() / P0, 2)})
    return rows


def e1_e2():
    r1 = grid(200)
    r2 = grid(-200)
    ch.record("E1_A_suit_B_200ms", r1)
    ch.record("E2_A_precede_B_200ms", r2)
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for rows, lab, col in ((r1, "A suit B (cote sur la référence)", GREEN), (r2, "A précède B (référence en retard)", ACCENT)):
        sel = [r for r in rows if r["latence_couverture_ms"] == 500]
        ax.plot([r["retard_recotation_ms"] for r in sel], [r["glissement_moyen_pb"] for r in sel], "o-", color=col, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("retard de recotation (ms)")
    ax.set_ylabel("glissement moyen (pb)")
    ax.set_title("Sélection adverse d'un ordre maker couvert (couverture 500 ms)")
    ax.legend(frameon=False, fontsize=7)
    ch.save_fig(fig, "selection_adverse")


def e3():
    rng = np.random.default_rng(153)
    rows = []
    for gap_ms in (10, 100, 1000, 5000):
        g = gap_ms / 1000
        # deux ordres au marché dont les exécutions sont séparées de g secondes : l'écart réalisé
        # subit le mouvement du prix pendant g (marche aléatoire)
        moves = rng.normal(0, SIG_STEP * np.sqrt(g / DT), 200000)
        rows.append({"decalage_ms": gap_ms, "ecart_type_pb": r3(1e4 * moves.std() / P0, 3),
                     "theorie_pb": r3(1e4 * SIG_STEP * np.sqrt(g / DT) / P0, 3),
                     "q01_pb": r3(1e4 * np.quantile(moves, 0.01) / P0, 3)})
    ch.record("E3", rows)


if __name__ == "__main__":
    e1_e2(); e3()
    ch.save_results()
