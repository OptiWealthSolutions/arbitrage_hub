"""Univers d'actions simulé, partagé par les chapitres 6, 7, 10 et 11.

Structure (log-prix quotidiens) :
    log P_i(t) = facteur_marché(t) * b_i + facteur_secteur_s(t) * c_i + idio_i(t)

- N_SECTORS secteurs de taille égale, facteurs = marches aléatoires ;
- dans chaque secteur, les premières actions forment des « jumelles » : deux actions d'une paire
  jumelle partagent la même composante idiosyncratique de long terme (marche aléatoire) et ne
  diffèrent que par un écart AR(1) stationnaire -> paire réellement cointégrée ;
- les autres actions ont une composante idiosyncratique en marche aléatoire -> pas de cointégration.

La fonction renvoie aussi la liste des vraies paires, pour mesurer la qualité d'une sélection.
"""
from __future__ import annotations

import numpy as np


def simulate_universe(rng: np.random.Generator, n_days: int = 1500, n_sectors: int = 5,
                      per_sector: int = 20, twins_per_sector: int = 2, vol_idio: float = 0.012,
                      vol_mkt: float = 0.009, vol_sec: float = 0.006, twin_phi: float = 0.95,
                      twin_sd: float = 0.004, drift_break: float | None = None):
    n = n_sectors * per_sector
    mkt = np.cumsum(rng.normal(0.0002, vol_mkt, n_days))
    logp = np.zeros((n_days, n))
    true_pairs = []
    sector_of = np.repeat(np.arange(n_sectors), per_sector)
    for s in range(n_sectors):
        sec = np.cumsum(rng.normal(0, vol_sec, n_days))
        idx = np.arange(s * per_sector, (s + 1) * per_sector)
        b = rng.uniform(0.8, 1.2, per_sector)
        c = rng.uniform(0.7, 1.3, per_sector)
        for k, i in enumerate(idx):
            logp[:, i] = b[k] * mkt + c[k] * sec
        # paires jumelles
        for t in range(twins_per_sector):
            i, j = idx[2 * t], idx[2 * t + 1]
            common = np.cumsum(rng.normal(0, vol_idio, n_days))
            gap = np.zeros(n_days)
            e = rng.normal(0, twin_sd, n_days)
            for d in range(1, n_days):
                gap[d] = twin_phi * gap[d - 1] + e[d]
            # même exposition aux facteurs pour les jumelles
            logp[:, j] = b[2 * t] * mkt + c[2 * t] * sec
            logp[:, i] += common + gap / 2
            logp[:, j] += common - gap / 2
            true_pairs.append((int(i), int(j)))
        for k in range(2 * twins_per_sector, per_sector):
            logp[:, idx[k]] += np.cumsum(rng.normal(0, vol_idio, n_days))
    logp += np.log(rng.uniform(20, 200, n))
    return np.exp(logp), true_pairs, sector_of
