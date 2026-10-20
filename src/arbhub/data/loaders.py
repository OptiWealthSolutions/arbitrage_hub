"""Chargement des données (seul module du paquet qui lit le disque).

Référence : chapitre 18, section « Architecture du paquet ».
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[3] / "data" / "case_studies"


def load_case_study(symbol: str, directory: Path | None = None) -> np.ndarray:
    """Clôtures quotidiennes d'une étude de cas (500 séances, oct. 2024 - oct. 2026).

    Entrée : symbole parmi GLD, GLDM, KO, PEP (voir data/case_studies/README.md).
    Sortie : tableau (500,) de prix, ordre chronologique. Lève FileNotFoundError sinon.
    """
    path = (directory or CASE_DIR) / f"{symbol}_close.txt"
    x = np.loadtxt(path, delimiter=",")
    if x.ndim != 1 or not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError(f"données invalides dans {path}")
    return x
