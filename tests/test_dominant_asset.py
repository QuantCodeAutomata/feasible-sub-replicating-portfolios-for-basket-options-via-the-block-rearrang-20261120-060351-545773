"""Unit tests for Proposition 3.2 (src/dominant_asset.py)."""
import numpy as np
import pytest

from src.dominant_asset import analyse_dominant_asset
from src.marginals import LogNormalMarginal, UniformMarginal


def test_single_crossing_from_above_gives_valid_strikes():
    # Table 4, row 2: sigma = (10%, 10%, 120%), K = 110.
    x1 = LogNormalMarginal(100.0, 0.10, 0.5, 1 / 3)
    x2 = LogNormalMarginal(100.0, 0.10, 0.5, 1 / 3)
    x3 = LogNormalMarginal(100.0, 1.20, 0.5, 1 / 3)
    res = analyse_dominant_asset(x1, x2, x3, kb=110.0, n_grid=100_000)
    assert res.crossing_direction == "above"
    assert res.crossings in (1, 2)
    assert res.u_star is not None
    assert 0.0 < res.u_star < 1.0
    assert res.a1 == pytest.approx(res.a2, rel=1e-6)  # x1, x2 identical in law
    assert res.c > 0.0
    assert res.a1 + res.a2 + res.c == pytest.approx(110.0, abs=1e-5)
    # bound (13) must equal V(mu^c) under the single-crossing condition.
    if res.crossings == 1:
        assert res.bound13 == pytest.approx(res.v_mu_c, rel=1e-6)
    else:
        assert res.bound13 <= res.v_mu_c + 1e-9


def test_dominant_regime_does_not_apply_three_asset_table2_example():
    # Table 4, row 4: identical to Table 2 (Experiment 1): one crossing from
    # below, so the dominant-asset portfolio is not constructed.
    x1 = LogNormalMarginal(100.0, 0.355, 0.5, 1 / 3)
    x2 = LogNormalMarginal(100.0, 0.20, 0.5, 1 / 3)
    x3 = LogNormalMarginal(100.0, 0.25, 0.5, 1 / 3)
    res = analyse_dominant_asset(x1, x2, x3, kb=100.0, n_grid=100_000)
    assert res.crossing_direction == "below"
    assert res.u_star is None
    assert res.portfolio_value is None


def test_uniform_dominant_case_portfolio_value_matches_integral():
    x1 = UniformMarginal(0.0, 30.0, 1 / 3)
    x2 = UniformMarginal(0.0, 30.0, 1 / 3)
    x3 = LogNormalMarginal(100.0, 1.0, 0.5, 1 / 3)
    res = analyse_dominant_asset(x1, x2, x3, kb=70.0, n_grid=100_000)
    assert res.crossing_direction == "above"
    assert res.portfolio_value == pytest.approx(res.bound13, rel=1e-3)
