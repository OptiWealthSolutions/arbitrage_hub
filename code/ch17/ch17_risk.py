"""Chapitre 17 — Risque et limites de l'arbitrage.

E1  Critère de Kelly : croissance, drawdowns et ruine selon la fraction de Kelly utilisée,
    avec le vrai avantage connu puis avec un avantage ESTIMÉ sur 3 ans (erreur d'estimation).
E2  Protections contre la rupture d'une paire : stop-loss en z, stop-loss en durée,
    alarme CUSUM, phi glissant. Coût dans un monde stable, bénéfice dans un monde qui rompt.
E3  Spirale de marge (Shleifer et Vishny, 1997) : un écart qui converge à coup sûr à
    l'échéance, mais s'élargit d'abord. Probabilité de liquidation forcée et gain espéré
    selon le levier.
E4  Diversification et foule : volatilité d'un portefeuille de N paires selon la corrélation,
    et perte d'un portefeuille « diversifié » quand les corrélations montent en crise.
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
from arbhub.backtest.engine import pair_backtest, zscore_positions
from arbhub.backtest.metrics import annualized_sharpe, max_drawdown
from arbhub.risk.sizing import cusum_alarm, kelly_fraction, rolling_ar1_phi

ch = Chapter("ch17")
CUSUM_K, CUSUM_H = 0.5, 10.0


# ------------------------------------------------------------------ E1
def e1(rng, years=10, n_paths=4000):
    T = 252 * years
    mu_a, vol_a = 0.10, 0.10                 # stratégie non levée : 10 % / an, vol 10 % -> Sharpe 1
    mu, sig = mu_a / 252, vol_a / np.sqrt(252)
    f_star = kelly_fraction(mu, sig)         # = 10
    fractions = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    r = rng.normal(mu, sig, (n_paths, T))
    rows = []
    curves = {}
    for frac in fractions:
        f = frac * f_star
        logw = np.cumsum(np.log1p(np.clip(f * r, -0.999, None)), axis=1)
        w_end = np.exp(logw[:, -1])
        peak = np.maximum.accumulate(logw, axis=1)
        dd = 1 - np.exp((logw - peak).min(axis=1))
        rows.append({"fraction_kelly": frac, "levier": r3(f, 2),
                     "croissance_annuelle_mediane_pct": r3(100 * (np.median(w_end) ** (1 / years) - 1), 1),
                     "richesse_mediane": r3(np.median(w_end), 2),
                     "proba_perte_sur_10_ans": r3(np.mean(w_end < 1), 3),
                     "proba_drawdown_50pct": r3(np.mean(dd > 0.5), 3),
                     "drawdown_median_pct": r3(100 * np.median(dd), 1)})
        curves[frac] = np.median(logw, axis=0)
    # erreur d'estimation : f* estimé sur 3 ans d'historique
    hist = rng.normal(mu, sig, (n_paths, 3 * 252))
    f_hat = hist.mean(1) / hist.var(1, ddof=1)
    est = {"f_vrai": r3(f_star, 2), "f_estime_median": r3(np.median(f_hat), 2),
           "f_estime_q10": r3(np.quantile(f_hat, 0.10), 2), "f_estime_q90": r3(np.quantile(f_hat, 0.90), 2),
           "part_f_estime_sup_2f": r3(np.mean(f_hat > 2 * f_star), 3),
           "part_f_estime_negatif": r3(np.mean(f_hat < 0), 3)}
    return rows, est, curves


# ------------------------------------------------------------------ E2
def make_pair(rng, n, phi, break_at=None, drift=0.0):
    x = 100 + np.cumsum(rng.normal(0, 1.0, n))
    s = np.zeros(n)
    for t in range(1, n):
        if break_at is not None and t >= break_at:
            s[t] = s[t - 1] + drift + rng.normal()
        else:
            s[t] = phi * s[t - 1] + rng.normal()
    return 1.5 * x + s + 50, x, s


def run_rules(y, x, s, form, hl):
    m, d = s[:form].mean(), s[:form].std(ddof=1)
    z = (s - m) / d
    out = {}
    base = zscore_positions(z, 2.0, 0.0)
    base[:form] = 0
    out["aucune"] = base
    for st in (3.0, 4.0):
        q = zscore_positions(z, 2.0, 0.0, stop=st)
        q[:form] = 0
        out[f"stop_z_{st:g}"] = q
    # stop en durée : on sort après 3 demi-vies, puis on attend un retour à |z| < 1 pour réarmer
    q = base.copy()
    held, cool = 0, False
    for t in range(len(q)):
        if cool:
            if abs(z[t]) < 1.0:
                cool = False
            q[t] = 0.0
            continue
        if base[t] != 0 and (t == 0 or base[t - 1] == base[t]):
            held += 1
        else:
            held = 1 if base[t] != 0 else 0
        if held > 3 * hl:
            q[t] = 0.0
            cool, held = True, 0
    out["stop_duree_3hl"] = q
    # CUSUM sur le z-score (moyenne 0, écart-type 1 sous l'hypothèse de stabilité) :
    # une alarme coupe définitivement la paire ; l'AR(1) est rendu ~indépendant en filtrant
    # e_t = s_t - phi s_{t-1} estimé sur la formation.
    phi_hat = np.polyfit(s[:form - 1], s[1:form], 1)[0]
    e = s[1:] - phi_hat * s[:-1]
    me, sde = e[:form - 1].mean(), e[:form - 1].std(ddof=1)
    a = cusum_alarm(e[form - 1:], me, sde, k=CUSUM_K, h=CUSUM_H)
    q = base.copy()
    if a is not None:
        q[form + a:] = 0.0
    out["cusum"] = q
    # phi glissant : pas de nouvelle position et sortie tant que phi estimé sur 120 j dépasse 0,98
    ph = rolling_ar1_phi(s, 120)
    off = (ph > 0.98) & ~np.isnan(ph)
    q = zscore_positions(np.where(off, 0.0, z), 2.0, 0.0)
    q[:form] = 0
    q[off] = 0
    out["phi_glissant"] = q
    alarm_day = None if a is None else int(a)
    return out, alarm_day


def e2(rng, n_worlds=300, n=1500, form=500, brk=900):
    hl = -np.log(2) / np.log(0.9)
    rules = None
    agg = {"stable": {}, "rupture": {}}
    alarms = {"stable": [], "rupture": []}
    for scen in ("stable", "rupture"):
        for _ in range(n_worlds):
            drift = rng.choice([-1, 1]) * 0.1
            y, x, s = make_pair(rng, n, 0.9, break_at=brk if scen == "rupture" else None, drift=drift)
            qs, al = run_rules(y, x, s, form, hl)
            alarms[scen].append(al)
            for k, q in qs.items():
                r = pair_backtest(y, x, 1.5, q, cost_bps=5)
                net = r.net[form:]
                agg[scen].setdefault(k, {"pnl": [], "dd": []})
                agg[scen][k]["pnl"].append(net.sum())
                agg[scen][k]["dd"].append(max_drawdown(net))
            rules = list(qs)
    res = {}
    for scen in agg:
        res[scen] = {k: {"pnl_moyen": r3(np.mean(v["pnl"]), 1), "pnl_q05": r3(np.quantile(v["pnl"], 0.05), 1),
                         "drawdown_moyen": r3(np.mean(v["dd"]), 1)} for k, v in agg[scen].items()}
    st = alarms["stable"]
    rp = alarms["rupture"]
    delays = [a - (brk - form) for a in rp if a is not None and a >= brk - form]
    res["cusum"] = {"fausses_alarmes_monde_stable": r3(np.mean([a is not None for a in st]), 3),
                    "detection_apres_rupture": r3(len(delays) / len(rp), 3),
                    "alarme_avant_rupture": r3(np.mean([a is not None and a < brk - form for a in rp]), 3),
                    "delai_median_jours": r3(np.median(delays), 0) if delays else None}
    return res, rules


# ------------------------------------------------------------------ E3
def e3(rng, n_paths=20000, T=120, p_crisis=0.5):
    """Spirale de marge. Écart initial s0 = 2 qui converge à coup sûr vers 0 à l'échéance T
    (pont brownien, comme la base d'un future qui expire). Dans la moitié des trajectoires,
    une crise de liquidité l'élargit en moyenne de +3 pendant les 20 premiers jours, puis se
    résorbe. L'arbitragiste vend l'écart avec une taille L (unités d'écart par unité de capital
    propre) ; il est liquidé de force si sa perte latente dépasse 75 % de son capital."""
    s0, vol = 2.0, 0.25
    crisis = rng.random(n_paths) < p_crisis
    push = np.where(np.arange(T) < 20, 3.0 / 20, 0.0)
    rows = []
    for L in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0):
        rs = np.random.default_rng(170)          # mêmes trajectoires pour tous les leviers
        s = np.full(n_paths, s0)
        alive = np.ones(n_paths, bool)
        pnl = np.zeros(n_paths)
        for t in range(T):
            remaining = T - t
            drift = -s / remaining + crisis * push[t]
            s_new = s + drift + vol * rs.normal(size=n_paths)
            if t == T - 1:
                s_new = np.zeros(n_paths)
            pnl = np.where(alive, pnl + L * (s - s_new), pnl)
            alive &= ~(alive & (pnl < -0.75))
            s = s_new
        rows.append({"levier": L, "gain_si_on_tient": r3(L * s0, 2),
                     "proba_liquidation": r3(1 - alive.mean(), 3),
                     "proba_liquidation_si_crise": r3(1 - alive[crisis].mean(), 3),
                     "gain_moyen": r3(pnl.mean(), 3)})
    return rows


# ------------------------------------------------------------------ E4
def e4(rng, n_pairs=20, days=2520):
    sig = 1.0
    analytic = []
    for rho in (0.0, 0.05, 0.1, 0.3, 0.6):
        v = sig * np.sqrt(1 / n_pairs + rho * (1 - 1 / n_pairs))
        analytic.append({"rho": rho, "vol_portefeuille_relative": r3(v, 3),
                         "N_effectif": r3(1 / v**2, 1)})
    # crise : pendant 10 jours, corrélation 0,6 et dérive défavorable (liquidations forcées)
    mu = 0.1 * sig                        # Sharpe quotidien 0,1 par paire -> ~1,6 annuel
    n_sims = 2000
    worst_normal, worst_crisis = [], []
    for _ in range(n_sims):
        common = rng.normal(size=days)
        idio = rng.normal(size=(days, n_pairs))
        rho = np.full(days, 0.05)
        drift = np.full(days, mu)
        c0 = rng.integers(100, days - 20)
        rho[c0:c0 + 10] = 0.6
        drift[c0:c0 + 10] = -1.0 * sig
        ret = drift[:, None] + sig * (np.sqrt(rho)[:, None] * common[:, None] + np.sqrt(1 - rho)[:, None] * idio)
        port = ret.mean(axis=1)
        # même monde sans crise
        ret_n = mu + sig * (np.sqrt(0.05) * common[:, None] + np.sqrt(0.95) * idio)
        port_n = ret_n.mean(axis=1)
        worst_normal.append(max_drawdown(port_n) / port_n.std())
        worst_crisis.append(max_drawdown(port) / port_n.std())
    return analytic, {"drawdown_max_en_ecarts_types_sans_crise": r3(np.median(worst_normal), 1),
                      "drawdown_max_en_ecarts_types_avec_crise": r3(np.median(worst_crisis), 1),
                      "sharpe_annuel_par_paire": r3(0.1 * np.sqrt(252), 2)}


def main():
    rng = np.random.default_rng(17)
    rows1, est1, curves = e1(rng)
    ch.record("E1", {"kelly": rows1, "estimation": est1})
    res2, rules = e2(rng)
    ch.record("E2", res2)
    rows3 = e3(rng)
    ch.record("E3", rows3)
    an4, sim4 = e4(rng)
    ch.record("E4", {"analytique": an4, "crise": sim4})
    ch.save_results()

    # Figure E1
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    t = np.arange(len(next(iter(curves.values())))) / 252
    cols = {0.1: GREEN, 0.25: MUTED, 0.5: "#5b7db1", 1.0: INK, 1.5: GOLD, 2.0: ACCENT}
    for frac, c in curves.items():
        axes[0].plot(t, c, color=cols[frac], lw=1.2, label=f"{frac:g} × Kelly")
    axes[0].set_xlabel("années")
    axes[0].set_ylabel("log-richesse médiane")
    axes[0].legend(frameon=False, fontsize=7)
    fr = [r["fraction_kelly"] for r in rows1]
    axes[1].plot(fr, [r["croissance_annuelle_mediane_pct"] for r in rows1], "o-", color=INK, label="croissance médiane (%/an)")
    ax2 = axes[1].twinx()
    ax2.plot(fr, [100 * r["proba_drawdown_50pct"] for r in rows1], "s--", color=ACCENT, label="P(drawdown > 50 %) (%)")
    axes[1].set_xlabel("fraction de Kelly")
    axes[1].set_ylabel("croissance médiane (%/an)", color=INK)
    ax2.set_ylabel("P(drawdown > 50 %) (%)", color=ACCENT)
    ax2.grid(False)
    fig.tight_layout()
    ch.save_fig(fig, "kelly")

    # Figure E2
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    names = list(res2["stable"].keys())
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [res2["stable"][k]["pnl_moyen"] for k in names], 0.4, color=INK, label="relation stable")
    ax.bar(xs + 0.2, [res2["rupture"][k]["pnl_moyen"] for k in names], 0.4, color=ACCENT, label="rupture au jour 400")
    ax.set_xticks(xs, [n.replace("_", " ") for n in names], fontsize=7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("gain moyen (1 000 jours)")
    ax.legend(frameon=False)
    ch.save_fig(fig, "protections_rupture")

    # Figure E3
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    Ls = [r["levier"] for r in rows3]
    ax.plot(Ls, [r["gain_si_on_tient"] for r in rows3], "--", color=MUTED, label="gain si l'on tient jusqu'au bout")
    ax.plot(Ls, [r["gain_moyen"] for r in rows3], "o-", color=INK, label="gain moyen réalisé")
    ax2 = ax.twinx()
    ax2.plot(Ls, [100 * r["proba_liquidation"] for r in rows3], "s-", color=ACCENT)
    ax2.set_ylabel("P(liquidation) (%)", color=ACCENT)
    ax2.grid(False)
    ax.set_xlabel("taille de la position (levier)")
    ax.set_ylabel("gain (en capital initial)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ch.save_fig(fig, "spirale_marge")
    return ch.results


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, ensure_ascii=False))
