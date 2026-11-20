"""Small, fast unit tests for src.neufeld_xiang on tiny synthetic inputs.

No paper-scale knot counts or Monte Carlo path counts are used here; see
``exp/exp4_neufeld_xiang_comparison.py`` for the full experiment.
"""

from __future__ import annotations

import numpy as np

from src.marginals import LogNormalMarginal
from src.neufeld_xiang import (
    equal_probability_knots,
    grid_lower_bound,
    grid_reassembly_bound,
)

T_HORIZON = 1.0


def _two_asset_lognormal() -> list[LogNormalMarginal]:
    return [
        LogNormalMarginal(s0=100.0, sigma=0.2, T=T_HORIZON, weight=0.5),
        LogNormalMarginal(s0=100.0, sigma=0.3, T=T_HORIZON, weight=0.5),
    ]


def test_grid_lower_bound_single_point_per_asset_recovers_trivial_payoff() -> None:
    """With a single knot per asset (placed at the marginal's mean), each
    hat function is a flat constant equal to that mean's node value, so the
    dual LP collapses to a single point constraint and the optimum must be
    exactly ``max(sum(means) - kb, 0)``."""
    marginals = _two_asset_lognormal()
    knots_list = [np.array([m.mean]) for m in marginals]
    total_mean = sum(m.mean for m in marginals)

    for kb in (total_mean - 50.0, total_mean + 50.0):
        res = grid_lower_bound(marginals, knots_list, kb=kb)
        expected = max(total_mean - kb, 0.0)
        assert abs(res.lower_bound - expected) < 1e-6, (res.lower_bound, expected, kb)


def test_grid_lower_bound_monotone_in_knot_count() -> None:
    """More knots can only enlarge the dual feasible set less (tighter grid
    approximation of each marginal's CDF), so the exact-vertex lower bound at
    a finer grid must not be smaller than at a coarser one, for the same
    strike (small m, 2 assets, finishes in well under a second)."""
    marginals = _two_asset_lognormal()
    kb = 95.0
    trunc = 1e-6

    knots_coarse = [equal_probability_knots(m, 4, trunc) for m in marginals]
    knots_fine = [equal_probability_knots(m, 8, trunc) for m in marginals]

    lb_coarse = grid_lower_bound(marginals, knots_coarse, kb=kb).lower_bound
    lb_fine = grid_lower_bound(marginals, knots_fine, kb=kb).lower_bound

    assert lb_fine >= lb_coarse - 1e-9, (lb_coarse, lb_fine)


def test_grid_reassembly_bound_is_at_least_the_lower_bound() -> None:
    """The Neufeld-Xiang comparison upper bound (Monte-Carlo reassembly of
    the BRA/RA antimonotone coupling with the true marginals) must be at
    least as large as the dual lower bound at the same grid, for a tiny
    synthetic example that finishes in a few seconds."""
    marginals = _two_asset_lognormal()
    kb = 95.0
    trunc = 1e-6

    knots = [equal_probability_knots(m, 8, trunc) for m in marginals]
    lb = grid_lower_bound(marginals, knots, kb=kb).lower_bound

    ub_mean, ub_se = grid_reassembly_bound(
        marginals, n_atoms=200, kb=kb, n_paths=2_000, n_repeats=2, seed=0
    )

    assert isinstance(ub_mean, float)
    assert ub_mean >= lb - 1e-6, (ub_mean, lb)
