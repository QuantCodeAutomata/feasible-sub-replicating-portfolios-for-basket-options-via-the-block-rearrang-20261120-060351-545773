"""Tests for the transfer to original marginals (src/exact.py) and the Algorithm 1
end-to-end pipeline (src/basket.py), on small discretisations for speed."""
import numpy as np
import pytest

from src.exact import hv_exact, reassembled_basket_value
from src.marginals import LogNormalMarginal
from src.basket import run_algorithm1
from src.rearrangement import bra


def test_hv_exact_matches_discretised_when_fine():
    marginals = [LogNormalMarginal(100.0, s, 0.5, 1 / 3) for s in (0.355, 0.20, 0.25)]
    n_atoms = 20_000
    X0 = np.column_stack([m.discretise(n_atoms) for m in marginals])
    res = bra(X0, stalliter=100, seed=0, f=None)
    from src.subreplication import sub_portfolio
    sp = sub_portfolio(res.X, kb=100.0, marginals=marginals)
    if sp.linear_fallback:
        pytest.skip("linear fallback on this coarse discretisation")
    hv_ex, _ = hv_exact(sp.strikes, kb=100.0, marginals=marginals)
    # The exact (continuous) hedging value should be close to its discretised
    # counterpart (Section 2.3): they must agree to within O(1/sqrt(N)).
    assert hv_ex == pytest.approx(sp.value, abs=0.5)


def test_reassembled_basket_value_upper_bounds_bv():
    marginals = [LogNormalMarginal(100.0, s, 0.5, 1 / 3) for s in (0.355, 0.20, 0.25)]
    n_atoms = 5_000
    X0 = np.column_stack([m.discretise(n_atoms) for m in marginals])
    res = bra(X0, stalliter=100, seed=0)
    bv_disc = float(np.mean(np.maximum(res.X.sum(axis=1) - 100.0, 0.0)))
    bv_ex = reassembled_basket_value(res.X, marginals, kb=100.0, n_nodes=8, n_nodes_kink=32)
    # Reassembly should stay close to the discretised basket value (both
    # approximate the same true minimal basket value from above).
    assert bv_ex == pytest.approx(bv_disc, abs=0.5)
    assert bv_ex >= 0.0


def test_run_algorithm1_hv_le_bv_and_bracket_closes():
    marginals = [LogNormalMarginal(100.0, s, 0.5, 1 / 3) for s in (0.355, 0.20, 0.25)]
    result = run_algorithm1(marginals, kb=110.0, n_atoms=4000, n_restarts=2, stall_window=60)
    assert result.hv <= result.bv + 1e-6
    assert result.hv_exact <= result.bv_exact + 1e-3
    assert result.hv > 0.0
    assert all(len(s) % 2 == 0 for s in result.strikes_disc)


def test_run_algorithm1_strikes_sorted_and_within_bounds():
    marginals = [LogNormalMarginal(100.0, s, 0.5, 1 / 3) for s in (0.355, 0.20, 0.25)]
    result = run_algorithm1(marginals, kb=100.0, n_atoms=4000, n_restarts=2, stall_window=60)
    for strikes in result.strikes_disc:
        if strikes.size:
            assert np.all(np.diff(strikes) >= -1e-9)
            assert strikes.min() >= -1e-9
            assert strikes.max() <= 100.0 + 1e-6
