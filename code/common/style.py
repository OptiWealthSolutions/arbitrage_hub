"""Style et utilitaires partagés par les scripts des chapitres 2 à 18.

Usage dans un script de chapitre :

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
    from style import Chapter, INK, ACCENT, MUTED, GREEN
    ch = Chapter("ch05")
    ...
    ch.save_fig(fig, "ou_simulation")       # -> docs/cours/figures/ch05/ou_simulation.pdf
    ch.save_results({"theta": 0.5})          # -> code/ch05/results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "case_studies"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

INK = "#1f2a44"
ACCENT = "#b5452b"
MUTED = "#8a8f98"
GREEN = "#3a7d44"
GOLD = "#b8860b"


class Chapter:
    def __init__(self, tag: str):
        self.tag = tag
        self.fig_dir = ROOT / "docs" / "cours" / "figures" / tag
        self.res_path = ROOT / "code" / tag / "results.json"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.res_path.parent.mkdir(parents=True, exist_ok=True)
        self.results: dict = {}

    def save_fig(self, fig, name: str) -> Path:
        path = self.fig_dir / f"{name}.pdf"
        fig.savefig(path)
        plt.close(fig)
        return path

    def record(self, key: str, value) -> None:
        self.results[key] = value

    def save_results(self, extra: dict | None = None) -> Path:
        if extra:
            self.results.update(extra)
        self.res_path.write_text(json.dumps(self.results, indent=2, ensure_ascii=False, default=_to_py))
        return self.res_path


def _to_py(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_close(symbol: str) -> np.ndarray:
    """Clôtures quotidiennes réelles (500 séances, oct. 2024 - oct. 2026), voir data/case_studies/README.md."""
    return np.loadtxt(DATA_DIR / f"{symbol}_close.txt", delimiter=",")


def ar1(rng, n, phi, sigma=1.0, mu=0.0, x0=None, n_paths=1):
    x = np.empty((n_paths, n + 1))
    x[:, 0] = mu + rng.normal(0, sigma / np.sqrt(1 - phi**2), n_paths) if x0 is None else x0
    eps = rng.normal(0, sigma, (n_paths, n))
    for t in range(1, n + 1):
        x[:, t] = mu + phi * (x[:, t - 1] - mu) + eps[:, t - 1]
    return x


def r3(x, d=3):
    return round(float(x), d)
