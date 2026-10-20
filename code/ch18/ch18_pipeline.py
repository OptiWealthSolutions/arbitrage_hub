"""Chapitre 18 — Du papier au code : parité, critères de passage, suivi en paper trading.

E1  Parité recherche / production : la version événementielle (PairStrategy.on_bar) et la
    version vectorisée donnent-elles exactement les mêmes positions, sur données réelles ?
E2  Grille de passage (« go / no-go ») appliquée à trois candidats : une paire simulée
    réellement cointégrée, KO/PEP et GLD/GLDM (500 séances réelles). Formation sur les
    250 premières séances, test sur les 250 suivantes.
E3  Retard d'exécution : décider à la clôture de t et n'être exécuté qu'à la clôture de t+1.
E4  Suivi du paper trading : au bout de combien de jours détecte-t-on qu'une stratégie
    réelle est moins bonne que son backtest ? Et combien de fausses alarmes sinon ?
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "common"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.stattools import coint

from style import Chapter, INK, ACCENT, MUTED, GREEN, GOLD, r3
from arbhub.data.loaders import load_case_study
from arbhub.signals.pair_strategy import PairConfig, PairStrategy, log_spread_pnl, vectorized_positions
from arbhub.backtest.metrics import annualized_sharpe, deflated_sharpe, sharpe
from arbhub.stats.mean_reversion import half_life

warnings.filterwarnings("ignore")
ch = Chapter("ch18")
FORM = 250
COST = 5.0
GRID = [PairConfig(beta_window=bw, z_window=zw, entry=e, exit=0.0, stop=4.0)
        for bw in (60, 120) for zw in (20, 40, 60) for e in (1.5, 2.0) if zw <= bw]


def sim_pair(rng, n=500, phi=0.85):
    lx = np.log(60) + np.cumsum(rng.normal(0, 0.012, n))
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0, 0.006)
    return np.exp(0.1 + 1.0 * lx + s), np.exp(lx)


# ------------------------------------------------------------------ E1
def e1():
    out = {}
    for name, (a, b) in {"KO_PEP": ("KO", "PEP"), "GLD_GLDM": ("GLD", "GLDM")}.items():
        y, x = load_case_study(a), load_case_study(b)
        cfg = PairConfig(beta_window=120, z_window=60, entry=2.0, exit=0.0, stop=4.0)
        q, beta, z = vectorized_positions(y, x, cfg)
        st = PairStrategy(cfg)
        ev = np.array([st.on_bar(yy, xx) for yy, xx in zip(y, x)])
        out[name] = {"ecart_max_positions": float(np.max(np.abs(q - ev[:, 0]))),
                     "ecart_max_beta": float(np.nanmax(np.abs(beta - ev[:, 1]))),
                     "changements_de_position": int(np.count_nonzero(np.diff(q)))}
    return out


# ------------------------------------------------------------------ E2
def breakeven_cost(y, x, q, b):
    g0 = log_spread_pnl(y, x, q, b, 0.0)
    g1 = log_spread_pnl(y, x, q, b, 1.0)          # coût de 1 pb
    per_bp = g0.sum() - g1.sum()
    return float(g0.sum() / per_bp) if per_bp > 0 else float("inf")


def gates(y, x):
    ly, lx = np.log(y), np.log(x)
    yf, xf = ly[:FORM], lx[:FORM]
    p_eg = float(coint(yf, xf, trend="c")[1])
    b, a = np.polyfit(xf, yf, 1)
    hl = half_life(yf - a - b * xf)
    # sélection de la meilleure variante de la grille sur la formation (sauf démarrage)...
    is_srs, oos = [], []
    for cfg in GRID:
        q, beta, _ = vectorized_positions(y, x, cfg)
        g = log_spread_pnl(y, x, q, beta, COST)
        is_srs.append(sharpe(g[cfg.beta_window:FORM]))
        oos.append((cfg, q, beta, g))
    j = int(np.argmax(is_srs))
    cfg, q, beta, g = oos[j]
    g_test = g[FORM:]
    dsr = deflated_sharpe(g[cfg.beta_window:FORM], len(GRID), float(np.var(is_srs, ddof=1)))
    q_t = q.copy()
    q_t[:FORM] = 0
    be = breakeven_cost(y, x, q_t, beta)
    res = {"p_engle_granger_formation": r3(p_eg, 4), "demi_vie_formation_jours": r3(hl, 1),
           "variante_retenue": {"beta_window": cfg.beta_window, "z_window": cfg.z_window, "entry": cfg.entry},
           "sharpe_formation": r3(is_srs[j] * np.sqrt(252), 2),
           "dsr_formation": r3(dsr, 3),
           "sharpe_test_net": r3(annualized_sharpe(g_test), 2),
           "rendement_test_net_pct": r3(100 * g_test.sum(), 2),
           "trades_test": int(np.count_nonzero(np.diff(q[FORM:]))),
           "cout_rentabilite_nulle_pb": r3(be, 1)}
    res["G1_cointegration"] = p_eg < 0.05
    res["G2_demi_vie_1_30j"] = bool(1 <= hl <= 30)
    res["G3_dsr_sup_0_95"] = bool(dsr > 0.95)
    res["G4_test_positif"] = bool(g_test.sum() > 0)
    res["G5_marge_couts_x2"] = bool(be > 2 * COST)
    res["decision"] = "GO paper trading" if all(res[k] for k in res if k.startswith("G")) else "NO-GO"
    return res, g


def e2(rng):
    out, curves = {}, {}
    for k in range(3):
        y, x = sim_pair(rng)
        r, g = gates(y, x)
        out[f"simulee_cointegree_{k + 1}"] = r
        if k == 0:
            curves["paire simulée 1"] = g
    for name, (a, b) in {"KO_PEP": ("KO", "PEP"), "GLD_GLDM": ("GLD", "GLDM")}.items():
        r, g = gates(load_case_study(a), load_case_study(b))
        out[name] = r
        curves[name.replace("_", "/")] = g
    return out, curves


# ------------------------------------------------------------------ E3
def e3(rng, n_worlds=200):
    rows = []
    for phi in (0.5, 0.8, 0.9, 0.95):
        s0, s1 = [], []
        for _ in range(n_worlds):
            y, x = sim_pair(rng, n=1000, phi=phi)
            cfg = PairConfig(beta_window=120, z_window=60, entry=2.0, exit=0.0, stop=4.0)
            q, b, _ = vectorized_positions(y, x, cfg)
            g = log_spread_pnl(y, x, q, b, COST)
            qd = np.concatenate([[0.0], q[:-1]])             # exécution un jour plus tard
            bd = np.concatenate([[np.nan], b[:-1]])
            gd = log_spread_pnl(y, x, qd, bd, COST)
            s0.append(annualized_sharpe(g[120:]))
            s1.append(annualized_sharpe(gd[120:]))
        rows.append({"phi": phi, "demi_vie_jours": r3(-np.log(2) / np.log(phi), 1),
                     "sharpe_median_execution_cloture": r3(np.median(s0), 2),
                     "sharpe_median_execution_lendemain": r3(np.median(s1), 2)})
    return rows


# ------------------------------------------------------------------ E4
def e4(rng, n_sims=4000, horizon=500):
    """Test unilatéral (on ne cherche que la sous-performance) : le Sharpe réalisé depuis le début du paper trading est-il
    significativement inférieur au Sharpe annoncé par le backtest ? Alarme quand
    z = (SR_live - SR_bt) / se(SR) < -1,645, vérifié chaque jour à partir du 20e."""
    rows = []
    for true_a, claim_a in ((1.5, 1.5), (1.0, 2.0), (0.0, 1.5), (0.0, 3.0)):
        mu = true_a / np.sqrt(252)
        sr_bt = claim_a / np.sqrt(252)
        r = rng.normal(mu, 1.0, (n_sims, horizon))
        cs = np.cumsum(r, axis=1)
        cs2 = np.cumsum(r * r, axis=1)
        T = np.arange(1, horizon + 1)
        m = cs / T
        sd = np.sqrt(np.maximum(cs2 / T - m * m, 1e-12) * T / np.maximum(T - 1, 1))
        sr = m / sd
        zt = (sr - sr_bt) / np.sqrt(1.0 / np.maximum(T - 1, 1))
        zt[:, :19] = 0.0
        alarm = zt < norm.ppf(0.05)
        first = np.where(alarm.any(axis=1), alarm.argmax(axis=1) + 1, np.nan)
        # contrôle unique, décidé à l'avance, à 250 jours
        once = zt[:, 249] < norm.ppf(0.05)
        # quatre contrôles (60, 125, 250, 500 j) au seuil de Bonferroni 5 % / 4
        checks = zt[:, [59, 124, 249, 499]] < norm.ppf(0.05 / 4)
        rows.append({"sharpe_vrai": true_a, "sharpe_annonce": claim_a,
                     "part_alarmes_controle_quotidien": r3(np.mean(~np.isnan(first)), 3),
                     "part_alarmes_controle_unique_250j": r3(np.mean(once), 3),
                     "part_alarmes_4_controles_bonferroni": r3(np.mean(checks.any(axis=1)), 3),
                     "delai_median_jours": r3(np.nanmedian(first), 0) if np.any(~np.isnan(first)) else None})
    return rows


def main():
    rng = np.random.default_rng(18)
    ch.record("E1", e1())
    r2, curves = e2(rng)
    ch.record("E2", r2)
    ch.record("E2_grille", {"variantes": len(GRID), "cout_pb_par_jambe": COST,
                            "formation": FORM, "test": 250})
    ch.record("E3", e3(rng))
    ch.record("E4", e4(rng))
    ch.save_results()

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    cols = [INK, ACCENT, GOLD]
    for (name, g), c in zip(curves.items(), cols):
        ax.plot(100 * np.cumsum(g), color=c, lw=1.1, label=name)
    ax.axvline(FORM, color="k", lw=0.7, ls="--")
    ax.text(FORM + 4, ax.get_ylim()[1] * 0.85, "début du test", fontsize=7)
    ax.set_xlabel("séance")
    ax.set_ylabel("gain cumulé net (% du capital)")
    ax.legend(frameon=False, fontsize=7)
    ch.save_fig(fig, "go_nogo")
    return ch.results


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2, ensure_ascii=False, default=str))
