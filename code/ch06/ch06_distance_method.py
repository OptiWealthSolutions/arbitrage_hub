"""Chapitre 6 — La méthode des distances (Gatev, Goetzmann, Rouwenhorst, 2006).

E1  Réplication du protocole GGR sur un univers simulé (100 actions, 5 secteurs, 10 vraies paires) :
    formation 12 mois, trading 6 mois, 20 meilleures paires, ouverture à 2 écarts-types, fermeture au croisement.
E2  Qualité de la sélection : part de vraies paires dans le top 20, gain par type de paire.
E3  Sensibilité aux coûts et au nombre de paires.
E4  Données réelles : règle GGR sur KO/PEP et GLD/GLDM (formation 252 séances, trading 248).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib.pyplot as plt

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, load_close, r3
from universe import simulate_universe

ch = Chapter("ch06")
FORM, TRADE = 252, 126


def ssd_matrix(norm):
    """Somme des écarts quadratiques entre toutes les colonnes de norm (T x N)."""
    sq = (norm**2).sum(0)
    return sq[:, None] + sq[None, :] - 2 * norm.T @ norm


def trade_pair(pi, pj, sd_form, k_open=2.0, cost_bps=0.0):
    """Règle GGR sur la période de trading. pi, pj : prix bruts sur la période. Renvoie (payoff total, nb trades)."""
    ni, nj = pi / pi[0], pj / pj[0]
    d = ni - nj
    payoff, n_tr, t = 0.0, 0, 0
    T = len(d)
    while t < T - 1:
        if abs(d[t]) > k_open * sd_form:
            s = np.sign(d[t])                     # +1 : i trop cher -> court i, long j
            t0 = t
            t += 1
            while t < T - 1 and np.sign(d[t]) == s:
                t += 1
            long_, short_ = (pj, pi) if s > 0 else (pi, pj)
            payoff += (long_[t] / long_[t0] - 1) - (short_[t] / short_[t0] - 1) - 4 * cost_bps / 1e4
            n_tr += 1
        t += 1
    return payoff, n_tr


def run_ggr(prices, true_pairs, n_top=20, cost_bps=0.0, k_open=2.0):
    T, N = prices.shape
    truth = set(true_pairs)
    rows = []
    for start in range(0, T - FORM - TRADE + 1, TRADE):
        f = prices[start:start + FORM]
        tr = prices[start + FORM:start + FORM + TRADE]
        norm = f / f[0]
        ssd = ssd_matrix(norm)
        iu = np.triu_indices(N, 1)
        order = np.argsort(ssd[iu])[:n_top]
        pairs = [(int(iu[0][o]), int(iu[1][o])) for o in order]
        pay, opened, true_sel, pay_true, pay_false = [], 0, 0, [], []
        for i, j in pairs:
            sd_form = (norm[:, i] - norm[:, j]).std()
            p, n_tr = trade_pair(tr[:, i], tr[:, j], sd_form, k_open, cost_bps)
            pay.append(p)
            opened += n_tr > 0
            is_true = (i, j) in truth
            true_sel += is_true
            (pay_true if is_true else pay_false).append(p)
        rows.append({"committed": np.mean(pay), "employed": np.sum(pay) / max(opened, 1),
                     "share_true": true_sel / n_top, "pay_true": pay_true, "pay_false": pay_false,
                     "opened": opened})
    return rows


def e1_e2_e3():
    rng = np.random.default_rng(61)
    universes = [simulate_universe(rng, n_days=FORM + TRADE * 12) for _ in range(6)]
    res = {}
    for cost in (0, 5, 10, 20):
        allrows = []
        for prices, tp, _ in universes:
            allrows += run_ggr(prices, tp, 20, cost)
        com = np.array([r["committed"] for r in allrows])
        emp = np.array([r["employed"] for r in allrows])
        res[f"cout_{cost}pb"] = {"rendement_semestriel_capital_engage_moyen": r3(100 * com.mean(), 3),
                                 "annualise_pct": r3(200 * com.mean(), 2),
                                 "rendement_semestriel_capital_employe_moyen": r3(100 * emp.mean(), 3),
                                 "part_semestres_positifs": r3((com > 0).mean(), 3)}
        if cost == 0:
            share = np.mean([r["share_true"] for r in allrows])
            pt = np.concatenate([r["pay_true"] for r in allrows])
            pf = np.concatenate([r["pay_false"] for r in allrows])
            opened = np.mean([r["opened"] for r in allrows])
            res["selection"] = {"semestres": len(allrows), "part_vraies_paires_dans_top20": r3(share, 3),
                                "vraies_paires_existantes": 10,
                                "gain_moyen_vraie_paire_pct": r3(100 * pt.mean(), 3),
                                "gain_moyen_fausse_paire_pct": r3(100 * pf.mean(), 3),
                                "paires_ouvertes_sur_20": r3(opened, 1)}
    ch.record("E1_E2_E3", res)

    # Sensibilité au nombre de paires (coût 5 pb)
    sens = {}
    for k in (5, 10, 20, 50):
        com = []
        for prices, tp, _ in universes:
            com += [r["committed"] for r in run_ggr(prices, tp, k, 5)]
        sens[k] = r3(200 * np.mean(com), 2)
    ch.record("E3_nombre_de_paires_annualise_pct", sens)

    # Figure : une paire vraie et une fausse paire du top 20
    prices, tp, _ = universes[0]
    f = prices[:FORM]; tr = prices[FORM:FORM + TRADE]
    norm = f / f[0]
    ssd = ssd_matrix(norm); iu = np.triu_indices(prices.shape[1], 1)
    order = np.argsort(ssd[iu])[:20]
    pairs = [(int(iu[0][o]), int(iu[1][o])) for o in order]
    true_p = next(p for p in pairs if p in set(tp))
    false_p = next((p for p in pairs if p not in set(tp)), pairs[-1])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=False)
    for ax, (i, j), lab in [(axes[0], true_p, "vraie paire (cointégrée)"), (axes[1], false_p, "paire choisie par hasard de formation")]:
        full = prices[:FORM + TRADE, [i, j]]
        nn = full / full[0]
        ax.plot(nn[:, 0], color=INK, lw=0.8)
        ax.plot(nn[:, 1], color=ACCENT, lw=0.8)
        ax.axvline(FORM, color="k", lw=0.6, ls=":")
        ax.set_title(lab)
        ax.set_xlabel("jour (formation | trading)")
    fig.tight_layout()
    ch.save_fig(fig, "ggr_paires")


def e4():
    out = {}
    for name, a, b in [("KO_PEP", "KO", "PEP"), ("GLD_GLDM", "GLD", "GLDM")]:
        pa, pb = load_close(a), load_close(b)
        f_a, f_b = pa[:FORM] / pa[0], pb[:FORM] / pb[0]
        sd_form = (f_a - f_b).std()
        p0, n0 = trade_pair(pa[FORM:], pb[FORM:], sd_form, 2.0, 0)
        p5, _ = trade_pair(pa[FORM:], pb[FORM:], sd_form, 2.0, 5)
        out[name] = {"ecart_type_formation_pct": r3(100 * sd_form, 3), "trades": n0,
                     "gain_brut_pct": r3(100 * p0, 2), "gain_net_5pb_pct": r3(100 * p5, 2)}
    ch.record("E4", out)


if __name__ == "__main__":
    e1_e2_e3(); e4()
    ch.save_results()
