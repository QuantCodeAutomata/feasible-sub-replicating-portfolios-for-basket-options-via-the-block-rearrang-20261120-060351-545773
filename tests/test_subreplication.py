"""Unit tests for the pair decomposition, exact cash component (Proposition
2.3) and strike refinement (src/subreplication.py)."""
import numpy as np
import pytest

from src.subreplication import (
    analytical_cash,
    compare_with_strike_free,
    f_payoff,
    sub_portfolio,
)
from src.marginals import LogNormalMarginal


def test_f_payoff_identity_when_no_strikes():
    z = np.array([0.0, 5.0, 50.0])
    out = f_payoff(z, np.array([]))
    assert np.allclose(out, z)


def test_f_payoff_call_payoff_with_0_and_K():
    kb = 40.0
    z = np.array([0.0, 20.0, 40.0, 60.0])
    out = f_payoff(z, np.array([0.0, kb]))
    expected = np.maximum(z - kb, 0.0)
    assert np.allclose(out, expected)


def test_analytical_cash_trivial_linear_case():
    # With empty strike sets for every asset, f_i(z) = z, so the best
    # Psi = (sum x_i - K)^+ - sum x_i over the box vertices {0, K}^n is
    # attained at 0 (sum=0: 0 - 0 = 0) and must be <= 0 at every other vertex.
    strikes = [np.array([]), np.array([])]
    cash = analytical_cash(strikes, kb=10.0)
    assert cash <= 0.0


def test_sub_portfolio_basic_three_asset():
    rng = np.random.default_rng(0)
    marginals = [LogNormalMarginal(100.0, s, 0.5, 1 / 3) for s in (0.355, 0.20, 0.25)]
    n_atoms = 5000
    prices = np.column_stack([m.discretise(n_atoms) for m in marginals])
    for j in range(prices.shape[1]):
        rng.shuffle(prices[:, j])
    sp = sub_portfolio(prices, kb=100.0, marginals=marginals)
    assert np.isfinite(sp.value)
    # HV must not exceed the basket value under this (arbitrary) coupling
    bv = float(np.mean(np.maximum(prices.sum(axis=1) - 100.0, 0.0)))
    assert sp.value <= bv + 1e-6


def test_compare_with_strike_free_keeps_better_portfolio():
    strikes = [np.array([10.0, 50.0]), np.array([10.0, 50.0])]
    hv = 100.0  # deliberately very large: must dominate both fallbacks
    strikes_out, hv_out, cash_out = compare_with_strike_free(
        strikes, hv, cash=-5.0, kb=100.0, mean_x=100.0, calls_at_kb=[1.0, 1.0], n_assets=2
    )
    assert hv_out == hv
    assert cash_out == -5.0
    assert strikes_out is strikes


def test_compare_with_strike_free_linear_fallback_is_identity():
    # Force hv below both benchmarks: calls_value very large, linear_value also large
    strikes = [np.array([10.0, 50.0]), np.array([10.0, 50.0])]
    strikes_out, hv_out, cash_out = compare_with_strike_free(
        strikes, hv=-1000.0, cash=0.0, kb=100.0, mean_x=200.0, calls_at_kb=[1.0, 1.0], n_assets=2
    )
    assert hv_out == pytest.approx(100.0)  # mean_x - kb
    assert cash_out == pytest.approx(-100.0)  # -kb
    assert all(s.size == 0 for s in strikes_out)


def test_compare_with_strike_free_strike_free_fallback_uses_full_k():
    # linear_value (mean_x - kb) below calls_value, and hv below both: the
    # strike-free fallback sum_i (x_i - K)^+ uses the full basket strike K on
    # every asset (Remark 2.10), not K split across assets.
    strikes = [np.array([10.0, 50.0]), np.array([10.0, 50.0])]
    strikes_out, hv_out, cash_out = compare_with_strike_free(
        strikes, hv=-1000.0, cash=0.0, kb=100.0, mean_x=0.0, calls_at_kb=[10.0, 10.0], n_assets=2
    )
    assert hv_out == pytest.approx(20.0)  # sum of calls_at_kb
    assert cash_out == pytest.approx(0.0)
    for s in strikes_out:
        assert np.allclose(s, [0.0, 100.0])


def test_pointwise_domination_sum_calls_le_basket_call():
    # Direct check of the inequality underlying Remark 2.10's strike-free
    # portfolio: sum_i (x_i - K)^+ <= (sum_i x_i - K)^+ for any nonnegative
    # realisation and any K > 0.
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 200.0, size=(2000, 4))
    kb = 50.0
    lhs = np.maximum(x - kb, 0.0).sum(axis=1)
    rhs = np.maximum(x.sum(axis=1) - kb, 0.0)
    assert np.all(lhs <= rhs + 1e-12)
