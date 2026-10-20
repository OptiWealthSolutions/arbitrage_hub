import numpy as np
import pytest

from arbhub.data.loaders import load_case_study
from arbhub.signals.pair_strategy import (PairConfig, PairStrategy, log_spread_pnl,
                                          vectorized_positions)


def _sim(seed=0, n=700):
    rng = np.random.default_rng(seed)
    lx = np.log(50) + np.cumsum(rng.normal(0, 0.01, n))
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.92 * s[t - 1] + rng.normal(0, 0.004)
    return np.exp(0.2 + 1.1 * lx + s), np.exp(lx)


def test_event_driven_matches_vectorized():
    y, x = _sim()
    cfg = PairConfig(beta_window=120, z_window=60, entry=2.0, exit=0.0, stop=4.0)
    q, b, z = vectorized_positions(y, x, cfg)
    st = PairStrategy(cfg)
    out = np.array([st.on_bar(yy, xx) for yy, xx in zip(y, x)])
    assert np.array_equal(q, out[:, 0])
    assert np.allclose(b[119:], out[119:, 1])
    assert np.allclose(z[119:], out[119:, 2])
    assert np.isnan(out[:119, 1]).all()


def test_no_lookahead():
    y, x = _sim(1)
    cfg = PairConfig()
    q, b, _ = vectorized_positions(y, x, cfg)
    y2 = y.copy()
    y2[400:] *= 1.3
    q2, b2, _ = vectorized_positions(y2, x, cfg)
    assert np.array_equal(q[:400], q2[:400])
    assert np.allclose(np.nan_to_num(b[:400]), np.nan_to_num(b2[:400]))


def test_pnl_costs_and_formula():
    y, x = _sim(2, 400)
    q, b, _ = vectorized_positions(y, x, PairConfig())
    g0 = log_spread_pnl(y, x, q, b, 0)
    g5 = log_spread_pnl(y, x, q, b, 5)
    assert g5.sum() <= g0.sum()
    t = int(np.argmax(q != 0)) + 1
    manual = q[t - 1] * (np.log(y[t] / y[t - 1]) - b[t - 1] * np.log(x[t] / x[t - 1]))
    assert np.isclose(g0[t], manual)


def test_config_validation():
    with pytest.raises(ValueError):
        PairConfig(beta_window=50, z_window=60)
    with pytest.raises(ValueError):
        PairConfig(entry=2.0, stop=1.5)
    with pytest.raises(ValueError):
        PairStrategy(PairConfig()).on_bar(-1.0, 2.0)


def test_case_study_loader():
    ko = load_case_study("KO")
    assert ko.shape == (500,) and np.all(ko > 0)
    with pytest.raises(FileNotFoundError):
        load_case_study("XXX")
