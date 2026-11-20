"""Vanilla option pricing helpers used throughout the replication.

Both empirical (on a discretised sample) and closed-form (log-normal,
Black-Scholes) call/put prices are needed: the Block Rearrangement Algorithm
and the pair decomposition act on the N equiprobable atoms of each marginal
(Section 5 of the paper), while the transfer to the original marginals
(Section 2.3, Propositions 2.3 and 2.11) needs the exact continuous prices.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def call_price(sample: np.ndarray, strike: float) -> float:
    """Empirical call price ``E[(X - strike)^+]`` on an equiprobable sample."""
    return float(np.mean(np.maximum(sample - strike, 0.0)))


def put_price(sample: np.ndarray, strike: float) -> float:
    """Empirical put price ``E[(strike - X)^+]`` on an equiprobable sample."""
    return float(np.mean(np.maximum(strike - sample, 0.0)))


def bs_call_price(forward: float, strike: float, sigma: float, T: float) -> float:
    """Black-Scholes call price with zero interest rate and no dividends
    (paper, Section 5: ``S_0 exp(sigma W_T - 0.5 sigma^2 T)``, zero rates).

    ``forward`` is the time-0 forward (= spot, since rates are zero).
    """
    strike = float(strike)
    if strike <= 0.0:
        return float(forward - strike)
    if sigma <= 0.0 or T <= 0.0:
        return float(max(forward - strike, 0.0))
    vol = sigma * np.sqrt(T)
    d1 = (np.log(forward / strike) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    return float(forward * norm.cdf(d1) - strike * norm.cdf(d2))


def bs_put_price(forward: float, strike: float, sigma: float, T: float) -> float:
    """Black-Scholes put price via put-call parity (zero interest rate)."""
    return bs_call_price(forward, strike, sigma, T) - forward + strike
