"""Chapitre 11 — Apprentissage automatique pour l'arbitrage statistique.

E1  Fuite d'information : validation croisée K-fold mélangée vs walk-forward purgé,
    sur des données SANS aucune prévisibilité (étiquettes à horizon de 10 jours qui se chevauchent).
E2  Prévision du signe du rendement d'un écart OU : forêt aléatoire vs règle z-score linéaire.
E3  Sélection de paires par ACP + clustering (dans l'esprit de Sarmento et Horta, 2020)
    puis test de cointégration à l'intérieur des groupes.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch07"))
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, ar1, r3
from universe import simulate_universe
from ch07_cointegration_selection import eg_pvalue, bh_select

ch = Chapter("ch11")


def features(x):
    """Variables explicatives calculées uniquement avec le passé (jusqu'à t inclus)."""
    r = np.diff(x, prepend=x[0])
    f = []
    for w in (1, 5, 10, 20):
        f.append(np.r_[np.full(w, np.nan), x[w:] - x[:-w]])
    m20 = np.array([x[max(0, t - 19):t + 1].mean() for t in range(len(x))])
    s20 = np.array([x[max(0, t - 19):t + 1].std() + 1e-12 for t in range(len(x))])
    f.append((x - m20) / s20)
    f.append(np.array([r[max(0, t - 19):t + 1].std() for t in range(len(x))]))
    return np.column_stack(f)


# ---------------------------------------------------------------- E1
def e1():
    rng = np.random.default_rng(111)
    H = 10
    accs = {"kfold_melange": [], "walk_forward_purge": []}
    for _ in range(10):
        x = np.cumsum(rng.normal(size=3000))               # aucune prévisibilité
        X = features(x)
        y = (np.r_[x[H:] - x[:-H], np.full(H, np.nan)] > 0).astype(float)
        ok = ~np.isnan(X).any(1) & (np.arange(len(x)) < len(x) - H)
        X, y = X[ok], y[ok]
        kf = KFold(5, shuffle=True, random_state=0)
        a = []
        for tr, te in kf.split(X):
            m = RandomForestClassifier(200, min_samples_leaf=1, random_state=0, n_jobs=-1).fit(X[tr], y[tr])
            a.append((m.predict(X[te]) == y[te]).mean())
        accs["kfold_melange"].append(np.mean(a))
        # walk-forward purgé : on entraîne sur le passé, on retire H jours avant le test
        a = []
        n = len(X)
        for k in range(5, 10):
            te = np.arange(k * n // 10, (k + 1) * n // 10)
            tr = np.arange(0, max(0, te[0] - H))
            m = RandomForestClassifier(200, min_samples_leaf=1, random_state=0, n_jobs=-1).fit(X[tr], y[tr])
            a.append((m.predict(X[te]) == y[te]).mean())
        accs["walk_forward_purge"].append(np.mean(a))
    ch.record("E1", {k: {"precision_moyenne": r3(np.mean(v), 3), "min": r3(np.min(v), 3), "max": r3(np.max(v), 3)}
                     for k, v in accs.items()} | {"series": 10, "horizon_etiquette": H, "vraie_precision": 0.5})


# ---------------------------------------------------------------- E2
def e2():
    rng = np.random.default_rng(112)
    out = {"foret": [], "regle_zscore": []}
    hits = {"foret": [], "regle_zscore": []}
    for _ in range(10):
        s = ar1(rng, 3999, 0.95, 0.01)[0] + np.cumsum(rng.normal(0, 0.002, 4000))   # OU + petite dérive aléatoire
        X = features(s)
        ret = np.r_[s[1:] - s[:-1], np.nan]
        y = (ret > 0).astype(float)
        ok = ~np.isnan(X).any(1) & ~np.isnan(ret)
        Xo, yo, ro = X[ok], y[ok], ret[ok]
        n = len(Xo); cut = n // 2
        m = RandomForestClassifier(300, min_samples_leaf=50, random_state=0, n_jobs=-1).fit(Xo[:cut - 1], yo[:cut - 1])
        p = m.predict_proba(Xo[cut:])[:, 1]
        pos_rf = np.where(p > 0.55, 1, np.where(p < 0.45, -1, 0))
        z = Xo[cut:, 4]                                       # z-score 20 jours
        pos_z = np.where(z < -1, 1, np.where(z > 1, -1, 0))
        for name, pos in (("foret", pos_rf), ("regle_zscore", pos_z)):
            pnl = pos * ro[cut:] - 1e-4 * np.abs(np.diff(pos, prepend=0))
            out[name].append(pnl.sum())
            act = pos != 0
            hits[name].append((np.sign(ro[cut:][act]) == pos[act]).mean())
    ch.record("E2", {k: {"pnl_moyen": r3(np.mean(v), 4), "taux_de_reussite": r3(np.mean(hits[k]), 3)} for k, v in out.items()})


# ---------------------------------------------------------------- E3
def e3():
    rng = np.random.default_rng(113)
    rows = []
    for u in range(4):
        prices, tp, sector = simulate_universe(rng, n_days=500, twin_phi=0.85)
        lp = np.log(prices)
        rets = np.diff(lp, axis=0)
        Z = StandardScaler().fit_transform(rets)
        emb = PCA(n_components=8).fit(Z).components_.T                # coordonnées des actions sur 8 composantes
        emb = StandardScaler().fit_transform(emb)
        lab = OPTICS(min_samples=2, xi=0.05).fit(emb).labels_
        cand = [(i, j) for i in range(lp.shape[1]) for j in range(i + 1, lp.shape[1]) if lab[i] == lab[j] and lab[i] != -1]
        truth = set(tp)
        pv = np.array([eg_pvalue(lp[:, i], lp[:, j])[0] for i, j in cand]) if cand else np.array([])
        sel = bh_select(pv, 0.05) if len(pv) else np.array([], bool)
        chosen = [c for c, s in zip(cand, sel) if s]
        rows.append({"groupes": int(lab.max() + 1), "paires_candidates": len(cand),
                     "vraies_dans_candidates": sum(c in truth for c in cand),
                     "retenues_apres_BH": len(chosen), "vraies_retenues": sum(c in truth for c in chosen)})
    ch.record("E3", {"par_univers": rows, "paires_possibles": 4950, "vraies_paires": 10})


if __name__ == "__main__":
    e1(); e2(); e3()
    ch.save_results()
