"""Chapitre 7 — Cointégration appliquée : sélection des paires et faux positifs.

E1  Tester les 4 950 paires d'un univers simulé : vrais positifs, faux positifs,
    corrections de Bonferroni et de Benjamini-Hochberg, restriction sectorielle.
E2  Performance hors échantillon (z-score, coûts 5 pb) selon la règle de sélection.
E3  Persistance : une paire sélectionnée sur une fenêtre l'est-elle sur la suivante ?
E4  Ratio de couverture statique contre glissant sur les vraies paires.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.adfvalues import mackinnonp

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, r3
from universe import simulate_universe

ch = Chapter("ch07")
FORM, TRADE = 500, 250
NU = 12   # nombre d'univers simulés par scénario


def eg_pvalue(y, x):
    """Engle-Granger (constante, 1 retard) ; p-valeur de MacKinnon pour 2 variables."""
    b, a = np.polyfit(x, y, 1)
    e = y - a - b * x
    de, el, dl = np.diff(e)[1:], e[1:-1], np.diff(e)[:-1]
    X = np.column_stack([np.ones_like(el), el, dl])
    coef, *_ = np.linalg.lstsq(X, de, rcond=None)
    r = de - X @ coef
    s2 = r @ r / (len(de) - 3)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    tau = coef[1] / se
    return float(mackinnonp(tau, regression="c", N=2)), b, a, e


def bh_select(p, q=0.05):
    m = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    ok = p[order] <= thresh
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    sel = np.zeros(m, bool)
    sel[order[:k]] = True
    return sel


def half_life(e):
    b = np.polyfit(e[:-1], e[1:], 1)[0]
    return -np.log(2) / np.log(b) if 0 < b < 1 else np.inf


def zscore_trade(y, x, b, a, m, d, cost_bps=5.0, entry=2.0, stop=None):
    """Règle z-score sur l'écart des LOG-prix s = y - a - b x.

    Une unité d'écart = 1 $ de Y contre b $ de X ; la variation de s approxime le gain en
    fraction du dollar investi sur Y. Rendement exprimé par dollar de notionnel brut (1 + |b|).
    """
    s = y - a - b * x
    z = (s - m) / d
    pos, pnl = 0.0, 0.0
    gross = 1.0 + abs(b)
    for t in range(1, len(z)):
        pnl += pos * (s[t] - s[t - 1])
        new = pos
        if pos == 0:
            if z[t] > entry: new = -1.0
            elif z[t] < -entry: new = 1.0
        elif (pos == 1 and z[t] >= 0) or (pos == -1 and z[t] <= 0):
            new = 0.0
        elif stop is not None and abs(z[t]) > stop:
            new = 0.0
        if new != pos:
            pnl -= abs(new - pos) * gross * cost_bps / 1e4
            pos = new
    if pos != 0:
        pnl -= gross * cost_bps / 1e4
    return pnl / gross


def analyse_universe(prices, true_pairs, sector_of):
    lp = np.log(prices)
    N = lp.shape[1]
    truth = set(true_pairs)
    members = set(k for pr in true_pairs for k in pr)
    rows = []
    for i in range(N):
        for j in range(i + 1, N):
            p, b, a, e = eg_pvalue(lp[:FORM, i], lp[:FORM, j])
            cat = "vraie" if (i, j) in truth else ("partielle" if (i in members or j in members) else "fausse")
            rows.append((i, j, p, b, a, e.mean(), e.std(), half_life(e), sector_of[i] == sector_of[j], (i, j) in truth, cat))
    return rows


def e1_e2_e3(twin_phi=0.95, tag="phi_0.95", make_fig=True):
    rng = np.random.default_rng(71)
    stats = {k: [] for k in ("p<0.05", "Bonferroni", "BH", "secteur+BH", "secteur+BH+demi-vie<30")}
    oos = {k: [] for k in stats}
    persist = []
    for u in range(NU):
        prices, tp, sector = simulate_universe(rng, n_days=FORM + TRADE * 2, twin_phi=twin_phi)
        rows = analyse_universe(prices, tp, sector)
        p = np.array([r[2] for r in rows])
        same = np.array([r[8] for r in rows])
        true_ = np.array([r[9] for r in rows])
        hl = np.array([r[7] for r in rows])
        m = len(p)
        sel = {"p<0.05": p < 0.05,
               "Bonferroni": p < 0.05 / m,
               "BH": bh_select(p, 0.05)}
        sec_idx = np.where(same)[0]
        s_bh = np.zeros(m, bool)
        s_bh[sec_idx[bh_select(p[sec_idx], 0.05)]] = True
        sel["secteur+BH"] = s_bh
        sel["secteur+BH+demi-vie<30"] = s_bh & (hl < 30)
        lp = np.log(prices)
        for k, s in sel.items():
            stats[k].append({"selectionnees": int(s.sum()), "vraies": int((s & true_).sum()),
                             "fausses": int((s & ~true_).sum()), "tests": int(m if not k.startswith("secteur") else len(sec_idx))})
            for idx in np.where(s)[0]:
                i, j, _, b, a, mu, sd = rows[idx][:7]
                r = zscore_trade(lp[FORM:FORM + TRADE, i], lp[FORM:FORM + TRADE, j], b, a, mu, sd)
                oos[k].append((r, rows[idx][9], rows[idx][10], u))
        # persistance : resélection BH sur la fenêtre suivante (décalée de TRADE)
        sel_now = set((rows[t][0], rows[t][1]) for t in np.where(sel["BH"])[0])
        lp2 = np.log(prices[TRADE:TRADE + FORM])
        again = 0
        for (i, j) in sel_now:
            p2 = eg_pvalue(lp2[:, i], lp2[:, j])[0]
            again += p2 < 0.05
        persist.append((len(sel_now), again, sum((i, j) in set(tp) for (i, j) in sel_now)))

    summ = {}
    for k in stats:
        s = stats[k]
        rr = np.array([x[0] for x in oos[k]]) if oos[k] else np.array([0.0])
        tt = np.array([x[1] for x in oos[k]]) if oos[k] else np.array([False])
        cats = np.array([x[2] for x in oos[k]]) if oos[k] else np.array(["fausse"])
        partial = cats == "partielle"
        pure = cats == "fausse"
        unis = np.array([x[3] for x in oos[k]]) if oos[k] else np.array([0])

        def clustered(mask):
            means = [rr[mask & (unis == uu)].mean() for uu in range(NU) if (mask & (unis == uu)).any()]
            if len(means) < 2:
                return None
            return {"moyenne_des_univers_pct": r3(100 * np.mean(means), 2),
                    "erreur_type_inter_univers_pct": r3(100 * np.std(means, ddof=1) / np.sqrt(len(means)), 2),
                    "univers": len(means), "min_pct": r3(100 * np.min(means), 1), "max_pct": r3(100 * np.max(means), 1)}
        summ[k] = {"tests_par_univers": s[0]["tests"],
                   "selectionnees_moy": r3(np.mean([x["selectionnees"] for x in s]), 2),
                   "vraies_moy": r3(np.mean([x["vraies"] for x in s]), 2),
                   "fausses_moy": r3(np.mean([x["fausses"] for x in s]), 2),
                   "rendement_moyen_par_paire_pct": r3(100 * rr.mean(), 3),
                   "erreur_type_pct": r3(100 * rr.std(ddof=1) / np.sqrt(len(rr)), 3) if len(rr) > 1 else None,
                   "rendement_vraies_pct": r3(100 * rr[tt].mean(), 3) if tt.any() else None,
                   "faux_positifs_partiellement_lies_moy": r3(partial.sum() / NU, 2),
                   "rendement_partiellement_lies_pct": r3(100 * rr[partial].mean(), 3) if partial.any() else None,
                   "faux_positifs_sans_lien_moy": r3(pure.sum() / NU, 2),
                   "rendement_sans_lien_pct": r3(100 * rr[pure].mean(), 3) if pure.any() else None,
                   "erreur_type_naive_sans_lien_pct": r3(100 * rr[pure].std(ddof=1) / np.sqrt(pure.sum()), 3) if pure.sum() > 1 else None,
                   "sans_lien_par_univers": clustered(pure),
                   "partiels_par_univers": clustered(partial),
                   "vraies_par_univers": clustered(tt)}
    ch.record("E1_E2_" + tag, {"vraies_paires_par_univers": 10, "univers": NU, "par_regle": summ})
    ch.record("E3_" + tag, {"BH_selectionnees": sum(x[0] for x in persist), "reselectionnees_fenetre_suivante": sum(x[1] for x in persist),
                     "dont_vraies": sum(x[2] for x in persist)})

    if not make_fig:
        return
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    keys = list(summ.keys())
    v = [summ[k]["vraies_moy"] for k in keys]
    f = [summ[k]["fausses_moy"] for k in keys]
    xs = np.arange(len(keys))
    ax.bar(xs, v, color=GREEN, label="vraies paires")
    ax.bar(xs, f, bottom=v, color=ACCENT, alpha=0.8, label="faux positifs")
    ax.set_xticks(xs)
    ax.set_xticklabels(["p<5 %", "Bonferroni", "BH", "secteur\n+BH", "secteur+BH\n+demi-vie"], fontsize=7)
    ax.set_ylabel("paires retenues (moyenne)")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title("Sélection parmi 4 950 paires (demi-vie des vraies paires : 4,3 j)")
    ax.legend(frameon=False, fontsize=7)
    ch.save_fig(fig, "selection_faux_positifs")


def e4():
    rng = np.random.default_rng(74)
    errs = {"statique_formation": [], "glissant_60j": [], "glissant_250j": []}
    for _ in range(3):
        prices, tp, _ = simulate_universe(rng, n_days=FORM + TRADE)
        lp = np.log(prices)
        for (i, j) in tp:
            y, x = lp[:, i], lp[:, j]
            b0, a0 = np.polyfit(x[:FORM], y[:FORM], 1)
            for name, w in (("glissant_60j", 60), ("glissant_250j", 250)):
                bs = []
                for t in range(FORM, FORM + TRADE, 10):
                    bs.append(np.polyfit(x[t - w:t], y[t - w:t], 1)[0])
                errs[name].append(np.mean(np.abs(np.array(bs) - 1.0)))
            errs["statique_formation"].append(abs(b0 - 1.0))
    ch.record("E4", {k: r3(np.mean(v), 4) for k, v in errs.items()} | {"vrai_beta": 1.0})


if __name__ == "__main__":
    e1_e2_e3(0.95, "phi_0.95", make_fig=False)
    e1_e2_e3(0.85, "phi_0.85", make_fig=True)
    e4()
    ch.save_results()
