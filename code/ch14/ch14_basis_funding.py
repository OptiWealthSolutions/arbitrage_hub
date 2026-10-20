"""Chapitre 14 — Arbitrage de base, cash-and-carry et funding des perpétuels.

E1  Cash-and-carry sur un future daté : gain final quasi certain, mais pertes latentes et appels de marge.
E2  Portage du funding (long comptant / court perpétuel) : rendement contre probabilité de liquidation selon le levier.
E3  Écart entre deux plateformes avec une base qui « fond » (portage) : centre fixe, moyenne mobile
    (bandes de Bollinger) ou base théorique (approche grid_futures_bot d'atjte).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib.pyplot as plt

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, r3

ch = Chapter("ch14")


# ---------------------------------------------------------------- E1
def e1():
    rng = np.random.default_rng(141)
    T_days, r, sigma = 63, 0.04, 0.20                 # 3 mois
    dev0, sd_b = 0.010, 0.004                          # base 1 % au-dessus du juste prix, bruit de base
    n_paths = 20000
    dt = 1 / 252
    t = np.arange(T_days + 1)
    S = 100 * np.exp(np.cumsum(np.c_[np.zeros(n_paths), rng.normal(-0.5 * sigma**2 * dt, sigma * np.sqrt(dt), (n_paths, T_days))], axis=1))
    tau = (T_days - t) / 252
    # pont brownien pour l'écart à la base théorique : part de dev0, finit à 0
    W = np.cumsum(np.c_[np.zeros(n_paths), rng.normal(0, 1, (n_paths, T_days))], axis=1)
    bridge = W - (t / T_days) * W[:, -1:]
    dev = dev0 * (1 - t / T_days) + sd_b * bridge / np.sqrt(T_days) * 2
    F = S * np.exp(r * tau) * (1 + dev)
    # position : long 1 comptant (financé au taux r), court 1 future
    fin = 100 * (np.exp(r * t / 252) - 1)
    pnl = (S - 100) - (F - F[:, :1]) - fin
    final = pnl[:, -1]
    worst = pnl.min(1)
    margin_calls = np.maximum(0, (F - F[:, :1])).max(1)       # appel de marge cumulé maximal sur la jambe courte
    ch.record("E1", {"base_initiale_pct": 100 * dev0, "gain_final_moyen": r3(final.mean(), 3), "gain_final_ecart_type": r3(final.std(), 4),
                     "gain_final_theorique": r3(100 * np.exp(r * T_days / 252) * dev0, 3),
                     "pire_perte_latente_mediane": r3(np.median(worst), 3), "pire_perte_latente_q01": r3(np.quantile(worst, 0.01), 3),
                     "appel_de_marge_max_median": r3(np.median(margin_calls), 2), "appel_de_marge_max_q99": r3(np.quantile(margin_calls, 0.99), 2)})
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    for i in range(40):
        ax.plot(t, pnl[i], color=MUTED, lw=0.5, alpha=0.7)
    ax.plot(t, np.median(pnl, 0), color=INK, lw=1.4, label="médiane")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("jour")
    ax.set_ylabel("P&L (pour 100 de notionnel)")
    ax.set_title("Cash-and-carry : convergence certaine, chemin incertain")
    ax.legend(frameon=False, fontsize=7)
    ch.save_fig(fig, "cash_and_carry")


# ---------------------------------------------------------------- E2
def e2():
    rng = np.random.default_rng(142)
    n_paths, days = 5000, 365
    sigma_y, mm = 0.60, 0.005                         # volatilité annuelle crypto, marge de maintien 0,5 %
    dt = 1 / 365
    # rendements quotidiens : diffusion + sauts
    diff = rng.normal(0, sigma_y * np.sqrt(dt), (n_paths, days))
    jumps = rng.binomial(1, 4 / 365, (n_paths, days)) * rng.normal(0, 0.08, (n_paths, days))
    ret = diff + jumps
    # funding par jour (3 paiements de 8 h) : AR(1) autour de 0,03 %/jour (~11 %/an), peut devenir négatif
    f = np.zeros((n_paths, days))
    for d in range(1, days):
        f[:, d] = 0.0003 + 0.97 * (f[:, d - 1] - 0.0003) + rng.normal(0, 0.00015, n_paths)
    rows = []
    for L in (1, 2, 3, 5, 10):
        # on remet la marge à niveau chaque jour ; liquidation si la hausse d'un jour dépasse 1/L - mm
        thr = 1 / L - mm
        liq = (np.exp(ret) - 1) > thr
        liquidated = liq.any(1)
        first = np.where(liquidated, liq.argmax(1), days)
        carry = np.array([f[i, :first[i]].sum() for i in range(n_paths)])
        # en cas de liquidation : perte de la marge restante (approx. mm) + coût de ré-entrée 10 pb ; jambe comptant conservée
        loss = np.where(liquidated, mm + 0.001, 0.0)
        # capital engagé : comptant (1) + marge du perpétuel (1/L)
        ret_cap = (carry - loss) / (1 + 1 / L)
        rows.append({"levier_perp": L, "seuil_hausse_quotidienne_pct": r3(100 * thr, 1),
                     "proba_liquidation_annuelle": r3(liquidated.mean(), 3),
                     "rendement_annuel_moyen_sur_capital_pct": r3(100 * ret_cap.mean(), 2),
                     "rendement_q05_pct": r3(100 * np.quantile(ret_cap, 0.05), 2)})
    neg = (f < 0).mean()
    ch.record("E2", {"funding_moyen_annuel_pct": r3(100 * 365 * f.mean(), 2), "part_jours_funding_negatif": r3(neg, 3), "tableau": rows})


# ---------------------------------------------------------------- E3
def e3():
    rng = np.random.default_rng(143)
    days = 252 * 4
    out = {"centre_fixe": [], "bollinger_20j": [], "base_theorique": []}
    for _ in range(300):
        # base d'un future daté sur l'or : portage de 4 %/an qui fond jusqu'à l'échéance (1 an), puis roulement
        t = np.arange(days)
        dte = 252 - (t % 252)
        fair_bp = 0.04 * dte / 252 * 1e4
        noise = np.zeros(days)
        e = rng.normal(0, 8, days)
        for d in range(1, days):
            noise[d] = 0.8 * noise[d - 1] + e[d]                # écart à la base théorique, en pb
        basis = fair_bp + noise
        form = 252
        mu_fixed = basis[:form].mean()
        sd = basis[:form].std()
        roll_m = np.array([basis[max(0, d - 19):d + 1].mean() for d in range(days)])
        roll_s = np.array([basis[max(0, d - 19):d + 1].std() + 1e-9 for d in range(days)])
        sd_noise = noise[:form].std()
        for name, center, scale in (("centre_fixe", np.full(days, mu_fixed), np.full(days, sd)),
                                    ("bollinger_20j", roll_m, roll_s),
                                    ("base_theorique", fair_bp, np.full(days, sd_noise))):
            z = (basis - center) / scale
            pos, pnl = 0.0, 0.0
            for d in range(form, days):
                if (d % 252) == 0:                                  # roulement : on solde
                    if pos != 0:
                        pnl -= 4.0; pos = 0.0
                # long base = long future / court comptant : gagne la variation de base et reçoit les intérêts
                # sur la vente à découvert du comptant (portage) ; court base : l'inverse. Le portage compense
                # exactement la fonte mécanique de la base théorique.
                if (d % 252) != 0:
                    pnl += pos * ((basis[d] - basis[d - 1]) + 0.04 / 252 * 1e4)
                new = pos
                if pos == 0:
                    new = -1.0 if z[d] > 1.5 else (1.0 if z[d] < -1.5 else 0.0)
                elif pos * z[d] >= 0:
                    new = 0.0
                if new != pos:
                    pnl -= abs(new - pos) * 4.0                     # 4 pb par changement (2 jambes)
                    pos = new
            out[name].append(pnl / 3)                               # pb par an
    ch.record("E3", {k: {"gain_moyen_pb_par_an": r3(np.mean(v), 1), "ecart_type": r3(np.std(v), 1)} for k, v in out.items()})


if __name__ == "__main__":
    e1(); e2(); e3()
    ch.save_results()
