"""Unit tests for src/marginals.py and src/pricing.py on synthetic inputs."""
import numpy as np
import pytest

from src.marginals import LogNormalMarginal, TruncatedNormalMixtureMarginal, UniformMarginal
from src.pricing import bs_call_price, bs_put_price, call_price, put_price


def test_bs_put_call_parity():
    c = bs_call_price(100.0, 105.0, 0.2, 0.5)
    p = bs_put_price(100.0, 105.0, 0.2, 0.5)
    assert c - p == pytest.approx(100.0 - 105.0, abs=1e-10)


def test_bs_call_price_zero_strike_is_forward():
    assert bs_call_price(100.0, 0.0, 0.2, 0.5) == pytest.approx(100.0)


def test_bs_call_price_decreasing_in_strike():
    strikes = np.array([80.0, 100.0, 120.0])
    prices = [bs_call_price(100.0, k, 0.2, 0.5) for k in strikes]
    assert prices[0] > prices[1] > prices[2] >= 0.0


def test_lognormal_discretisation_matches_mean():
    marg = LogNormalMarginal(s0=100.0, sigma=0.3, T=0.5, weight=1.0)
    sample = marg.discretise(20_000)
    assert sample.shape == (20_000,)
    assert np.all(np.diff(sample) >= 0.0)
    assert sample.mean() == pytest.approx(100.0, rel=2e-2)
    assert marg.mean == pytest.approx(100.0)


def test_lognormal_call_price_matches_empirical():
    marg = LogNormalMarginal(s0=100.0, sigma=0.355, T=0.5, weight=1 / 3)
    sample = marg.discretise(100_000)
    for k in (20.0, 33.3, 45.0):
        empirical = call_price(sample, k)
        exact = marg.call_price(k)
        assert empirical == pytest.approx(exact, abs=5e-2)


def test_uniform_marginal_quantile_and_call_price():
    marg = UniformMarginal(lo=0.0, hi=30.0, weight=1.0)
    assert marg.mean == pytest.approx(15.0)
    sample = marg.discretise(50_000)
    assert sample.min() >= 0.0 and sample.max() <= 30.0
    empirical = call_price(sample, 10.0)
    assert empirical == pytest.approx(marg.call_price(10.0), abs=1e-1)


def test_truncated_normal_mixture_cdf_bounds():
    mix = TruncatedNormalMixtureMarginal(
        weights=np.array([0.3, 0.7]), means=np.array([-2.0, 3.0]),
        stds=np.array([1.5, 2.0]), lo=-10.0, hi=10.0,
    )
    assert mix.cdf(np.array([-10.0]))[0] == pytest.approx(0.0, abs=1e-8)
    assert mix.cdf(np.array([10.0]))[0] == pytest.approx(1.0, abs=1e-8)
    sample = mix.discretise(20_000)
    assert sample.min() >= -10.0 and sample.max() <= 10.0
    assert sample.mean() == pytest.approx(mix.mean, abs=5e-2)


def test_empirical_put_call_parity():
    marg = LogNormalMarginal(s0=100.0, sigma=0.2, T=0.5, weight=1.0)
    sample = marg.discretise(50_000)
    k = 100.0
    assert call_price(sample, k) - put_price(sample, k) == pytest.approx(sample.mean() - k, abs=1e-6)
