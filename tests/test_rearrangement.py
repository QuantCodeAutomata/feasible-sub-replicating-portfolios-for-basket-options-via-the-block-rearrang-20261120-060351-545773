"""Unit tests for RA/BRA (src/rearrangement.py): the rearranged sample must
reduce the objective relative to the unrearranged (shuffled) baseline, and
must preserve each column's marginal atoms (a permutation, not a resample)."""
import numpy as np
import pytest

from src.rearrangement import basket_call_objective, bra, ra


def _random_matrix(seed=0, n=4000, d=3):
    rng = np.random.default_rng(seed)
    cols = [np.sort(rng.normal(size=n)) for _ in range(d)]
    X0 = np.column_stack(cols)
    for j in range(d):
        rng.shuffle(X0[:, j])
    return X0


def test_ra_reduces_variance_of_sum():
    X0 = _random_matrix(seed=1)
    baseline_var = np.var(X0.sum(axis=1))
    res = ra(X0, stalliter=100, seed=1)
    assert res.fval <= baseline_var
    # columns are permutations of the originals, not resamples
    for j in range(X0.shape[1]):
        assert np.allclose(np.sort(res.X[:, j]), np.sort(X0[:, j]))


def test_bra_reduces_variance_at_least_as_much_as_ra():
    X0 = _random_matrix(seed=2)
    res_ra = ra(X0, stalliter=100, seed=2)
    res_bra = bra(X0, stalliter=100, seed=2)
    assert res_bra.fval <= res_ra.fval + 1e-8


def test_bra_k_aware_objective_reduces_call_price():
    X0 = _random_matrix(seed=3)
    kb = 0.0
    obj = basket_call_objective(kb)
    baseline = obj(X0.sum(axis=1))
    res = bra(X0, stalliter=150, seed=3, f=obj)
    assert res.fval <= baseline


def test_antimonotone_two_columns_reaches_near_zero_variance_bound():
    # With two columns and enough iterations, BRA should approach the
    # Frechet-Hoeffding antimonotone lower bound on Var(X+Y).
    rng = np.random.default_rng(4)
    n = 3000
    a = np.sort(rng.normal(size=n))
    b = np.sort(rng.normal(size=n))
    X0 = np.column_stack([a, rng.permutation(b)])
    res = bra(X0, stalliter=200, seed=4)
    antimonotone_sum = a + b[::-1]
    lower_bound_var = np.var(antimonotone_sum)
    assert res.fval == pytest.approx(lower_bound_var, rel=0.05)
