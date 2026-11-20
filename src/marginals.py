"""Marginal-distribution classes and their mid-point quantile discretisation.

Section 5 of the paper discretises each marginal into ``N`` equiprobable
atoms placed at mid-point quantiles

    x^{(m)} = F^{-1}((m - 0.5) / N),  m = 1, ..., N,

and the Block Rearrangement Algorithm (BRA) operates on this ``N x n`` sample.
Each marginal class below exposes the exact (continuous) call price used for
the transfer back to the original marginals (Section 2.3) and the
discretisation used by the BRA (Section 5). Adapted from the structure of the
authors' companion code (``subreplicating/marginals.py`` and
``subreplicating/exact.py`` at https://github.com/MohamadAbouDaya/Subreplication),
rewritten here against Section 5 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import lognorm, norm, uniform

from .pricing import bs_call_price, bs_put_price


def midpoint_quantiles(n_atoms: int) -> np.ndarray:
    """Mid-point probability levels ``(m - 0.5) / N``, ``m = 1, ..., N``."""
    return (np.arange(1, n_atoms + 1) - 0.5) / n_atoms


@dataclass
class LogNormalMarginal:
    """Log-normal marginal ``S_T = S_0 exp(-0.5 sigma^2 T + sigma sqrt(T) Z)``
    (paper, Section 5, Table 1), weighted by ``weight`` so that the basket
    asset is ``x = weight * S_T``.
    """

    s0: float
    sigma: float
    T: float
    weight: float = 1.0

    @property
    def mean(self) -> float:
        return self.weight * self.s0

    def _scale(self) -> float:
        return self.s0 * np.exp(-0.5 * self.sigma**2 * self.T)

    def _s(self) -> float:
        return self.sigma * np.sqrt(self.T)

    def quantile(self, p: np.ndarray | float) -> np.ndarray:
        q = lognorm.ppf(p, s=self._s(), scale=self._scale())
        return self.weight * q

    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        """Continuous ``F_{X_i}(x)`` of the weighted asset, used for the
        right derivative ``C'_{X_i}^+(x) = -Q(X_i > x)`` in the pair
        decomposition (Section 2.2)."""
        x = np.asarray(x, dtype=float)
        if self.weight == 0.0:
            return np.where(x >= 0.0, 1.0, 0.0)
        return lognorm.cdf(x / self.weight, s=self._s(), scale=self._scale())

    def discretise(self, n_atoms: int) -> np.ndarray:
        return np.sort(self.quantile(midpoint_quantiles(n_atoms)))

    def call_price(self, strike: float) -> float:
        """Exact call price on the weighted asset: ``C_x(k) = weight * C_S(k/weight)``."""
        if self.weight == 0.0:
            return max(-strike, 0.0)
        return self.weight * bs_call_price(self.s0, strike / self.weight, self.sigma, self.T)

    def put_price(self, strike: float) -> float:
        if self.weight == 0.0:
            return max(strike, 0.0)
        return self.weight * bs_put_price(self.s0, strike / self.weight, self.sigma, self.T)


@dataclass
class UniformMarginal:
    """Uniform marginal on ``[lo, hi]``, weighted by ``weight``."""

    lo: float
    hi: float
    weight: float = 1.0

    @property
    def mean(self) -> float:
        return self.weight * 0.5 * (self.lo + self.hi)

    def quantile(self, p: np.ndarray | float) -> np.ndarray:
        return self.weight * uniform.ppf(p, loc=self.lo, scale=self.hi - self.lo)

    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.weight == 0.0:
            return np.where(x >= 0.0, 1.0, 0.0)
        return uniform.cdf(x / self.weight, loc=self.lo, scale=self.hi - self.lo)

    def discretise(self, n_atoms: int) -> np.ndarray:
        return np.sort(self.quantile(midpoint_quantiles(n_atoms)))

    def call_price(self, strike: float) -> float:
        k = strike / self.weight if self.weight else strike
        lo, hi = self.lo, self.hi
        if k <= lo:
            return self.weight * (0.5 * (lo + hi) - k)
        if k >= hi:
            return 0.0
        return self.weight * (hi - k) ** 2 / (2.0 * (hi - lo))

    def put_price(self, strike: float) -> float:
        return self.call_price(strike) - self.mean + strike


@dataclass
class TruncatedNormalMixtureMarginal:
    """Mixture of normal distributions truncated to ``[lo, hi]`` (Neufeld and
    Xiang (2022), third numerical experiment). ``shift`` translates the
    marginal (used in Experiment 5 to make the transformed asset non-negative).
    """

    weights: np.ndarray
    means: np.ndarray
    stds: np.ndarray
    lo: float
    hi: float
    shift: float = 0.0

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        self.means = np.asarray(self.means, dtype=float)
        self.stds = np.asarray(self.stds, dtype=float)
        assert abs(self.weights.sum() - 1.0) < 1e-8, "mixture weights must sum to one"

    def _component_mass(self) -> np.ndarray:
        """Truncated probability mass of each component on [lo, hi]."""
        za = (self.lo - self.means) / self.stds
        zb = (self.hi - self.means) / self.stds
        return norm.cdf(zb) - norm.cdf(za)

    def _norm_const(self) -> float:
        return float(np.sum(self.weights * self._component_mass()))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float)) - self.shift
        za = (self.lo - self.means) / self.stds
        out = np.zeros_like(x, dtype=float)
        for w, mu, sd, a in zip(self.weights, self.means, self.stds, za):
            out += w * (norm.cdf((x - mu) / sd) - norm.cdf(a))
        return out / self._norm_const()

    def quantile(self, p: np.ndarray | float) -> np.ndarray:
        """Inverse CDF by bisection (mixture CDFs have no closed-form inverse)."""
        p = np.atleast_1d(np.asarray(p, dtype=float))
        lo = np.full_like(p, self.lo)
        hi = np.full_like(p, self.hi)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            below = self.cdf(mid) < p
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return self.shift + 0.5 * (lo + hi)

    @property
    def mean(self) -> float:
        za = (self.lo - self.means) / self.stds
        zb = (self.hi - self.means) / self.stds
        phi_a, phi_b = norm.pdf(za), norm.pdf(zb)
        comp_mean = self.means + self.stds * (phi_a - phi_b) / self._component_mass()
        return self.shift + float(np.sum(self.weights * self._component_mass() * comp_mean) / self._norm_const())

    def discretise(self, n_atoms: int) -> np.ndarray:
        return np.sort(self.quantile(midpoint_quantiles(n_atoms)))

    def call_price(self, strike: float) -> float:
        """``E[(shift + Z - strike)^+]`` for truncated-mixture ``Z``, by
        piecewise integration of the component truncated normals."""
        k = strike - self.shift
        if k <= self.lo:
            return self.mean - strike
        if k >= self.hi:
            return 0.0
        za = (self.lo - self.means) / self.stds
        zb = (self.hi - self.means) / self.stds
        zk = (k - self.means) / self.stds
        phi_k, phi_b = norm.pdf(zk), norm.pdf(zb)
        # E[(Z_i - k)^+ 1{Z_i in [lo,hi]}] for each truncated-normal component
        term = self.stds * (phi_k - phi_b) - (k) * (norm.cdf(zb) - norm.cdf(zk))
        # express in absolute terms (component untruncated mean contribution)
        comp = self.stds * (phi_k - phi_b) - k * (norm.cdf(zb) - norm.cdf(zk))
        comp = self.means * (norm.cdf(zb) - norm.cdf(zk)) + self.stds * (phi_k - phi_b) - k * (
            norm.cdf(zb) - norm.cdf(zk)
        )
        value = float(np.sum(self.weights * comp)) / self._norm_const()
        return value

    def put_price(self, strike: float) -> float:
        return self.call_price(strike) - self.mean + strike


@dataclass
class AffineMarginal:
    """Affine transform ``Y = a * X + b`` of a base marginal (Experiment 5,
    ``y_i = s_i x_i + 10 |s_i|``): monotone increasing if ``a > 0``, monotone
    decreasing (quantile reversal) if ``a < 0``.
    """

    base: object
    a: float
    b: float = 0.0

    @property
    def mean(self) -> float:
        return self.a * self.base.mean + self.b

    def quantile(self, p: np.ndarray | float) -> np.ndarray:
        p = np.atleast_1d(np.asarray(p, dtype=float))
        if self.a >= 0:
            return self.a * self.base.quantile(p) + self.b
        return self.a * self.base.quantile(1.0 - p) + self.b

    def cdf(self, x: np.ndarray | float) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if self.a > 0:
            return self.base.cdf((x - self.b) / self.a)
        if self.a < 0:
            return 1.0 - self.base.cdf((x - self.b) / self.a)
        return np.where(self.b >= x, 1.0, 0.0)

    def discretise(self, n_atoms: int) -> np.ndarray:
        return np.sort(self.quantile(midpoint_quantiles(n_atoms)))

    def call_price(self, strike: float) -> float:
        if self.a > 0:
            return self.a * self.base.call_price((strike - self.b) / self.a)
        if self.a < 0:
            return (-self.a) * self.base.put_price((strike - self.b) / self.a)
        return max(self.b - strike, 0.0)

    def put_price(self, strike: float) -> float:
        return self.call_price(strike) - self.mean + strike


def lognormal_marginals(weights: np.ndarray, vols: np.ndarray, T: float, n_atoms: int) -> np.ndarray:
    """Discretised sample matrix ``(n_atoms, n_assets)`` of log-normal basket
    assets ``x_i = weight_i * S_{i,T}`` with ``S_{i,0} = 100`` (Table 1)."""
    cols = [LogNormalMarginal(100.0, float(v), float(T), float(w)).discretise(n_atoms)
            for w, v in zip(weights, vols)]
    return np.column_stack(cols)
