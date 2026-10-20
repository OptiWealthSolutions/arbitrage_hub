import numpy as np

from arbhub.stats.mean_reversion import ar1_phi, half_life, zscore_formation


def simulate_ar1(phi, n, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x


def test_half_life_recovers_truth_on_long_sample():
    x = simulate_ar1(0.95, 50_000)
    assert abs(half_life(x) - (-np.log(2) / np.log(0.95))) < 1.0


def test_random_walk_has_phi_close_to_one():
    x = np.cumsum(np.random.default_rng(1).normal(size=20_000))
    assert ar1_phi(x) > 0.995


def test_zscore_uses_only_formation_data():
    train = np.array([0.0, 1.0, -1.0, 0.5, -0.5])
    test = np.array([2.0, 3.0])
    z1 = zscore_formation(train, test)
    # modifier les données de test ne doit pas changer la normalisation
    z2 = zscore_formation(train, np.array([2.0, 300.0]))
    assert z1[0] == z2[0]
