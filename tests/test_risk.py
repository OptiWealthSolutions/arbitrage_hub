import numpy as np

from arbhub.risk.sizing import (cusum_alarm, kelly_fraction, log_growth, rolling_ar1_phi,
                                vol_target_weights)


def test_kelly_maximizes_log_growth():
    mu, sigma = 0.0004, 0.01
    f = kelly_fraction(mu, sigma)
    assert np.isclose(f, 4.0)
    assert log_growth(f, mu, sigma) > log_growth(0.8 * f, mu, sigma)
    assert log_growth(f, mu, sigma) > log_growth(1.2 * f, mu, sigma)
    assert np.isclose(log_growth(2 * f, mu, sigma), 0.0)      # à 2 f*, la croissance est nulle


def test_vol_target_uses_past_only_and_hits_target():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2.0, 3000)
    w = vol_target_weights(x, 1.0, window=60)
    x2 = x.copy()
    x2[2000:] *= 10
    w2 = vol_target_weights(x2, 1.0, window=60)
    assert np.array_equal(w[:2000], w2[:2000])
    realized = (w[:-1] * x[1:])[100:].std()
    assert abs(realized - 1.0) < 0.1


def test_cusum_detects_shift_not_noise():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, 300)
    assert cusum_alarm(noise, 0.0, 1.0, k=0.5, h=8.0) is None
    shifted = np.concatenate([noise, rng.normal(1.5, 1, 100)])
    t = cusum_alarm(shifted, 0.0, 1.0, k=0.5, h=8.0)
    assert t is not None and 300 <= t < 330


def test_rolling_phi_tracks_regime():
    rng = np.random.default_rng(2)
    s = np.zeros(1200)
    for t in range(1, 1200):
        phi = 0.8 if t < 600 else 1.0
        s[t] = phi * s[t - 1] + rng.normal()
    p = rolling_ar1_phi(s, 200)
    assert np.isnan(p[150])
    assert abs(p[590] - 0.8) < 0.1
    assert p[1190] > 0.9
