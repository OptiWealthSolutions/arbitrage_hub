"""Tests du moteur de backtest et des mesures du chapitre 16."""
import numpy as np

from arbhub.backtest.engine import pair_backtest, rolling_zscore, zscore_positions
from arbhub.backtest.metrics import (deflated_sharpe, expected_max_sharpe, max_drawdown,
                                     pbo_cscv, probabilistic_sharpe, sharpe, sharpe_se)


def _pair(seed=0, n=800):
    rng = np.random.default_rng(seed)
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.9 * s[t - 1] + rng.normal()
    return 1.5 * x + s, x


def test_no_lookahead_signal_and_pnl():
    """Modifier les données après t ne change ni les positions ni le P&L jusqu'à t."""
    y, x = _pair()
    t0 = 500
    z = rolling_zscore(y - 1.5 * x, 60)
    q = zscore_positions(z, 2.0, 0.0)
    r = pair_backtest(y, x, 1.5, q, cost_bps=5)
    y2, x2 = y.copy(), x.copy()
    y2[t0 + 1:] += 50.0
    x2[t0 + 1:] -= 20.0
    z2 = rolling_zscore(y2 - 1.5 * x2, 60)
    q2 = zscore_positions(z2, 2.0, 0.0)
    r2 = pair_backtest(y2, x2, 1.5, q2, cost_bps=5)
    assert np.array_equal(q[: t0 + 1], q2[: t0 + 1])
    assert np.allclose(r.net[: t0 + 1], r2.net[: t0 + 1])


def test_pnl_matches_manual_formula():
    y, x = _pair(1, 300)
    q = zscore_positions(rolling_zscore(y - 1.5 * x, 40), 1.5, 0.0)
    r = pair_backtest(y, x, 1.5, q)
    s = y - 1.5 * x
    manual = np.concatenate([[0.0], q[:-1] * np.diff(s)])
    assert np.allclose(r.gross, manual)
    assert r.costs.sum() == 0.0


def test_costs_only_when_trading():
    y, x = _pair(2, 200)
    q = np.zeros(200)
    q[50:80] = 1.0
    r = pair_backtest(y, x, 1.5, q, cost_bps=10)
    assert np.count_nonzero(r.costs) == 2           # entrée en 50, sortie en 80
    expected_entry = (y[50] + 1.5 * x[50]) * 10 / 1e4
    assert np.isclose(r.costs[50], expected_entry)


def test_stop_loss_closes_position():
    z = np.array([0.0, 2.5, 3.0, 4.5, 3.0, 0.5, -0.1])
    q = zscore_positions(z, 2.0, 0.0, stop=4.0)
    assert list(q) == [0, -1, -1, 0, -1, -1, 0]


def test_rolling_zscore_matches_naive():
    s = np.random.default_rng(3).normal(size=100).cumsum()
    z = rolling_zscore(s, 20)
    t = 57
    w = s[t - 19: t + 1]
    assert np.isclose(z[t], (s[t] - w.mean()) / w.std(ddof=1))
    assert np.isnan(z[18]) and not np.isnan(z[19])


def test_psr_is_half_at_threshold_and_se_formula():
    assert np.isclose(probabilistic_sharpe(0.1, 0.1, 500), 0.5)
    assert np.isclose(sharpe_se(0.0, 101), 0.1)


def test_expected_max_sharpe_increases_with_trials():
    v = 0.01
    vals = [expected_max_sharpe(n, v) for n in (2, 10, 100, 1000)]
    assert all(a < b for a, b in zip(vals, vals[1:]))
    assert expected_max_sharpe(1, v) == 0.0


def test_dsr_penalizes_many_trials():
    r = np.random.default_rng(4).normal(0.05, 1.0, 750)
    assert deflated_sharpe(r, 1000, 1 / 750) < deflated_sharpe(r, 2, 1 / 750)


def test_max_drawdown():
    assert max_drawdown(np.array([1.0, -2.0, -1.0, 3.0])) == 3.0


def test_pbo_noise_vs_signal():
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 1, (1600, 30))
    pbo_noise, _ = pbo_cscv(noise, n_blocks=10)
    signal = noise.copy()
    signal[:, 0] += 0.15
    pbo_signal, _ = pbo_cscv(signal, n_blocks=10)
    assert 0.25 < pbo_noise < 0.75
    assert pbo_signal < 0.05
    assert sharpe(signal[:, 0]) > sharpe(noise[:, 0])


def test_annualized_sharpe_and_cscv_count():
    from arbhub.backtest.metrics import annualized_sharpe, n_cscv_combinations
    r = np.array([0.01, -0.005, 0.02, 0.0])
    assert np.isclose(annualized_sharpe(r, 252), sharpe(r) * np.sqrt(252))
    assert n_cscv_combinations(16) == 12870


def test_dsr_defined_for_flat_returns():
    assert 0.0 <= deflated_sharpe(np.zeros(300), 20, 0.004) < 0.5
