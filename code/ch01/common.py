"""Outils partagés par les scripts du chapitre 1.

- style des figures (police serif proche de Times, comme les PDF du cours)
- chemins de sortie (figures dans docs/01_fondations/figures, chiffres dans results/)
- petites fonctions de simulation réutilisées par plusieurs scripts
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "01_fondations" / "figures"
RES_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

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

INK = "#1f2a44"      # série principale
ACCENT = "#b5452b"   # série de comparaison / théorie
MUTED = "#8a8f98"    # trajectoires secondaires


def save_fig(fig, name: str) -> Path:
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def save_results(name: str, payload: dict) -> Path:
    path = RES_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def random_walk(rng: np.random.Generator, n_steps: int, n_paths: int = 1,
                sigma: float = 1.0, p0: float = 0.0) -> np.ndarray:
    """Marches aléatoires P_t = P_{t-1} + eps_t, forme (n_paths, n_steps+1), P_0 = p0."""
    eps = rng.normal(0.0, sigma, size=(n_paths, n_steps))
    paths = np.concatenate([np.full((n_paths, 1), p0), p0 + np.cumsum(eps, axis=1)], axis=1)
    return paths


def ar1(rng: np.random.Generator, n_steps: int, phi: float, sigma: float = 1.0,
        mu: float = 0.0, x0: float | None = None, n_paths: int = 1) -> np.ndarray:
    """AR(1) centré sur mu : X_t - mu = phi (X_{t-1} - mu) + eps_t.

    Si x0 est None, X_0 est tiré dans la loi stationnaire N(mu, sigma^2 / (1 - phi^2)).
    """
    x = np.empty((n_paths, n_steps + 1))
    if x0 is None:
        x[:, 0] = mu + rng.normal(0.0, sigma / np.sqrt(1 - phi**2), size=n_paths)
    else:
        x[:, 0] = x0
    eps = rng.normal(0.0, sigma, size=(n_paths, n_steps))
    for t in range(1, n_steps + 1):
        x[:, t] = mu + phi * (x[:, t - 1] - mu) + eps[:, t - 1]
    return x


def ols_slope_intercept(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Régression y = a + b x par moindres carrés ; renvoie (b, a)."""
    xm, ym = x.mean(), y.mean()
    b = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
    return float(b), float(ym - b * xm)
