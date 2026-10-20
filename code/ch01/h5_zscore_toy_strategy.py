"""H5 — Une première règle de pairs trading en z-score : avantage, coûts, rupture.

Monde simulé (donc la vérité est connue) :
    X_t marche aléatoire autour de 100 ;  Y_t = beta X_t + s_t,  beta = 1,5
    s_t AR(1) de paramètre phi = 0,9 (demi-vie ~ 6,6 jours), écart-type innovation 1.

Protocole, SANS information future :
  1. Formation (jours 0 à 499) : on estime beta_hat et alpha_hat par MCO de Y sur X,
     puis la moyenne m et l'écart-type d des résidus.
  2. Trading (jours 500 à 1499) : z_t = (Y_t - beta_hat X_t - alpha_hat - m) / d.
     z_t > +2 : on VEND le spread (vendre 1 Y, acheter beta_hat X) ;
     z_t < -2 : on ACHÈTE le spread ; sortie quand z_t repasse par 0.
     La position décidée à la clôture t s'applique à la variation de t à t+1.
  3. Coût : c points de base du notionnel échangé sur CHAQUE jambe, à chaque transaction.

Hypothèses testées :
  (a) avec un spread réellement stationnaire, la règle gagne en moyenne ;
  (b) les coûts réduisent l'avantage et peuvent l'annuler ;
  (c) si la relation se rompt (le spread devient une marche aléatoire au jour 200 de la
      période de trading), l'espérance de gain APRÈS la rupture tombe à zéro moins les
      coûts, alors que la plupart des trades restent gagnants : beaucoup de petits gains,
      quelques très grosses pertes (profil « ramasser des pièces devant un rouleau
      compresseur »).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import INK, ACCENT, MUTED, save_fig, save_results, ols_slope_intercept

SEED = 5
BETA, PHI, SIG_S = 1.5, 0.9, 1.0
N_FORM, N_TRADE = 500, 1000
ENTRY, EXIT = 2.0, 0.0
COSTS_BPS = [0, 5, 10, 25, 50]
REPS = 400


def simulate(rng: np.random.Generator, break_at: int | None = None):
    n = N_FORM + N_TRADE
    x = 100 + np.cumsum(rng.normal(0, 1.0, n))
    s = np.zeros(n)
    for t in range(1, n):
        phi = 1.0 if (break_at is not None and t >= N_FORM + break_at) else PHI
        s[t] = phi * s[t - 1] + rng.normal(0, SIG_S)
    y = BETA * x + s
    return x, y


def backtest(x: np.ndarray, y: np.ndarray, cost_bps: float):
    xf, yf = x[:N_FORM], y[:N_FORM]
    b, a = ols_slope_intercept(yf, xf)
    resid_f = yf - b * xf - a
    m, d = resid_f.mean(), resid_f.std(ddof=1)

    xt, yt = x[N_FORM:], y[N_FORM:]
    spread = yt - b * xt - a
    z = (spread - m) / d

    pos = np.zeros(len(z))  # +1 = long spread (long Y, short b X), -1 = short spread
    for t in range(len(z)):
        prev = pos[t - 1] if t > 0 else 0.0
        if prev == 0:
            pos[t] = -1.0 if z[t] > ENTRY else (1.0 if z[t] < -ENTRY else 0.0)
        elif prev == 1 and z[t] >= EXIT:
            pos[t] = 0.0
        elif prev == -1 and z[t] <= EXIT:
            pos[t] = 0.0
        else:
            pos[t] = prev
    pos[-1] = 0.0  # on solde en fin de période

    gross = np.zeros(len(z))
    gross[1:] = pos[:-1] * np.diff(spread)              # pas d'anticipation : pos(t-1) * d spread(t)
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    notional_traded = dpos * (yt + abs(b) * xt)          # les deux jambes
    costs = notional_traded * cost_bps / 1e4
    net = gross - costs
    n_trades = int((np.diff(np.concatenate([[0.0], pos])) != 0).sum() // 2)
    # découpage en trades : un trade = une séquence de jours où pos != 0
    trades, start = [], None
    for t in range(len(pos)):
        if pos[t] != 0 and start is None:
            start = t
        if start is not None and pos[t] == 0:
            trades.append((start, t, float(net[start:t + 1].sum())))
            start = None
    return {"net": net, "pnl": float(net.sum()), "n_round_trips": n_trades,
            "beta_hat": b, "z": z, "pos": pos, "trades": trades}


def sharpe(daily: np.ndarray) -> float:
    sd = daily.std(ddof=1)
    return float(np.sqrt(252) * daily.mean() / sd) if sd > 0 else 0.0


def main() -> dict:
    rng = np.random.default_rng(SEED)
    worlds_ok = [simulate(rng) for _ in range(REPS)]
    BREAK = 200
    worlds_break = [simulate(rng, break_at=BREAK) for _ in range(REPS)]

    by_cost = {}
    for c in COSTS_BPS:
        runs = [backtest(x, y, c) for x, y in worlds_ok]
        pnl = np.array([r["pnl"] for r in runs])
        shp = np.array([sharpe(r["net"]) for r in runs])
        by_cost[c] = {"pnl_moyen": round(float(pnl.mean()), 2),
                      "part_pnl_positif": round(float((pnl > 0).mean()), 3),
                      "sharpe_median": round(float(np.median(shp)), 2),
                      "allers_retours_moyens": round(float(np.mean([r["n_round_trips"] for r in runs])), 1)}
    runs_b = [backtest(x, y, 5) for x, y in worlds_break]
    runs_ok5 = [backtest(x, y, 5) for x, y in worlds_ok]
    # P&L réalisé APRÈS la date de rupture (jours >= BREAK de la période de trading)
    pnl_b = np.array([r["net"][BREAK:].sum() for r in runs_b])
    pnl_ok5 = np.array([r["net"][BREAK:].sum() for r in runs_ok5])
    tr_b = np.array([p for r in runs_b for (s0, _, p) in r["trades"] if s0 >= BREAK])
    tr_ok = np.array([p for r in runs_ok5 for (s0, _, p) in r["trades"] if s0 >= BREAK])

    def trade_stats(tr):
        return {"nb_trades": int(len(tr)), "taux_gagnants": round(float((tr > 0).mean()), 3),
                "gain_moyen": round(float(tr[tr > 0].mean()), 2),
                "perte_moyenne": round(float(tr[tr <= 0].mean()), 2) if (tr <= 0).any() else 0.0,
                "pire_trade": round(float(tr.min()), 2)}
    beta_hats = np.array([backtest(x, y, 0)["beta_hat"] for x, y in worlds_ok[:200]])

    # --- figures
    ex = backtest(*worlds_ok[0], 5)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    ax = axes[0]
    ax.plot(ex["z"], color=INK, lw=0.6)
    for lvl in (ENTRY, -ENTRY):
        ax.axhline(lvl, color=ACCENT, lw=0.8, ls="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.fill_between(np.arange(len(ex["pos"])), -4, 4, where=ex["pos"] != 0, color=MUTED, alpha=0.2, lw=0)
    ax.set_ylim(-4, 4)
    ax.set_title("z-score et périodes en position")
    ax.set_xlabel("jour de trading")

    ax = axes[1]
    xs = list(by_cost.keys())
    ax.plot(xs, [by_cost[c]["sharpe_median"] for c in xs], marker="o", color=INK)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Sharpe médian selon le coût")
    ax.set_xlabel("coût (pb par jambe)")

    ax = axes[2]
    bins = np.linspace(min(pnl_b.min(), pnl_ok5.min()), max(pnl_b.max(), pnl_ok5.max()), 50)
    ax.hist(pnl_ok5, bins=bins, color=INK, alpha=0.75, label="relation stable")
    ax.hist(pnl_b, bins=bins, color=ACCENT, alpha=0.6, label="après rupture")
    ax.set_title("P&L après le jour 200, coût 5 pb")
    ax.set_xlabel("P&L ($ par unité de spread)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_fig(fig, "h5_zscore_toy_strategy")

    res = {
        "seed": SEED, "repetitions": REPS, "beta": BETA, "phi": PHI,
        "beta_hat_moyen": round(float(beta_hats.mean()), 4),
        "beta_hat_ecart_type": round(float(beta_hats.std()), 4),
        "par_cout_bps": by_cost,
        "apres_jour_200_cout_5bps": {
            "stable": {"pnl_moyen": round(float(pnl_ok5.mean()), 2),
                       "pnl_q05": round(float(np.quantile(pnl_ok5, 0.05)), 2),
                       "trades": trade_stats(tr_ok)},
            "rupture": {"pnl_moyen": round(float(pnl_b.mean()), 2),
                        "erreur_type_moyenne": round(float(pnl_b.std(ddof=1) / np.sqrt(len(pnl_b))), 2),
                        "pnl_q05": round(float(np.quantile(pnl_b, 0.05)), 2),
                        "trades": trade_stats(tr_b)}},
    }
    save_results("h5_zscore_toy_strategy", res)
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, ensure_ascii=False))
