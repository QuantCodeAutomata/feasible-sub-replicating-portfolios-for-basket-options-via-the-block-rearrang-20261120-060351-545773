"""Closed-form bounds for basket options with one dominant asset (paper,
Section 3.2, Proposition 3.2).

When the co-monotone/antimonotone coupling ``mu^c`` (assets 1, 2 moving
together, asset 3 moving against their sum) makes the basket sum ``S^c``
cross the strike ``K`` only once, from above, the portfolio

    Pi_2^sub = min(x_1, a_1) + min(x_2, a_2) + (x_3 - c)^+ - a_1 - a_2

is optimal, with cash component exactly ``-(a_1 + a_2)`` (Proposition 3.2(i)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass
class DominantAssetResult:
    u_star: float | None
    a1: float | None
    a2: float | None
    c: float | None
    v_mu_c: float
    bound13: float | None
    crossings: int
    crossing_direction: str
    portfolio_value: float | None


def _s_curve(x1: object, x2: object, x3: object, u: np.ndarray) -> np.ndarray:
    """Basket sum along the co-monotone/antimonotone coupling,
    ``S^c(u) = F_1^{-1}(u) + F_2^{-1}(u) + F_3^{-1}(1-u)``."""
    return x1.quantile(u) + x2.quantile(u) + x3.quantile(1.0 - u)


def count_crossings(s_curve: np.ndarray, kb: float) -> tuple[int, str]:
    """Count sign changes of ``S^c(u) - K`` and report the direction of the
    first crossing ("above" if ``S^c`` starts above ``K``, "below" otherwise)."""
    sign = np.sign(s_curve - kb)
    sign[sign == 0] = 1.0
    changes = int(np.sum(np.diff(sign) != 0))
    direction = "above" if s_curve[0] > kb else "below"
    return changes, direction


def analyse_dominant_asset(x1, x2, x3, kb: float, n_grid: int = 200_000) -> DominantAssetResult:
    """Implements Proposition 3.2: find the first downcrossing ``u*`` of
    ``S^c(u) = K``, the optimal strikes ``a_1, a_2, c``, the portfolio value
    ``V``, the basket value ``V(mu^c)`` under the co-monotone/antimonotone
    coupling, and the bound (13) (the value attained at the first downcrossing,
    which coincides with ``V(mu^c)`` under the single-crossing-from-above
    condition of Proposition 3.2(iii))."""
    u_grid = (np.arange(1, n_grid + 1) - 0.5) / n_grid
    s_curve = _s_curve(x1, x2, x3, u_grid)
    v_mu_c = float(np.mean(np.maximum(s_curve - kb, 0.0)))

    crossings, direction = count_crossings(s_curve, kb)

    if direction != "above":
        return DominantAssetResult(None, None, None, None, v_mu_c, None, crossings, direction, None)

    def g(u: float) -> float:
        return float(x1.quantile(u) + x2.quantile(u) + x3.quantile(1.0 - u) - kb)

    # first downcrossing: smallest u in (0,1) with g(u) = 0 and g decreasing through it
    idx_candidates = np.flatnonzero(np.diff(np.sign(s_curve - kb)) < 0)
    if idx_candidates.size == 0:
        return DominantAssetResult(None, None, None, None, v_mu_c, None, crossings, direction, None)
    i0 = idx_candidates[0]
    lo, hi = u_grid[i0], u_grid[min(i0 + 1, n_grid - 1)]
    u_star = brentq(g, lo, hi, xtol=1e-14, rtol=1e-14)

    a1 = float(x1.quantile(u_star))
    a2 = float(x2.quantile(u_star))
    c = float(x3.quantile(1.0 - u_star))
    assert abs(a1 + a2 + c - kb) < 1e-6, "a1 + a2 + c must equal K (Proposition 3.2(ii))"

    portfolio_value = x3.call_price(c) - x1.put_price(a1) - x2.put_price(a2)

    # bound (13): integral over u in (0, u*) of (S^c(u) - K) du, i.e. the
    # value attained by stopping the co-monotone/antimonotone coupling at the
    # first downcrossing.
    mask = u_grid <= u_star
    bound13 = float(np.sum(np.where(mask, s_curve - kb, 0.0)) / n_grid)

    return DominantAssetResult(u_star, a1, a2, c, v_mu_c, bound13, crossings, direction, portfolio_value)
