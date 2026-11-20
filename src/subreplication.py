"""Pair decomposition, exact cash component, and strike refinement.

Implements Algorithm 1, steps 3-6 of the paper:

  * the pair-decomposition strike search (Section 2.2): for each asset ``i``,
    the auxiliary function ``phi_i`` built from the right derivative of the
    call price on ``X_i`` and the left derivative of the empirical call price
    on ``Y_i = sum_{j != i} X_j`` identifies the set ``A_i`` where the
    sub-replicating portfolio should hold a non-trivial position (Section 2.2);
  * the exact cash component by vertex enumeration (Proposition 2.3);
  * the derivative-free local search that refines the strikes (Section 2.2,
    "strike refinement").

Adapted from the structure of the authors' companion code
(``subreplicating/subreplication.py`` at
https://github.com/MohamadAbouDaya/Subreplication), rewritten here against
Proposition 2.3 and Section 2.2 of the paper.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .pricing import call_price


def _phi_breakpoints(marginal, y_col: np.ndarray, kb: float) -> np.ndarray:
    """Ordered boundaries ``K^1 < ... < K^{2n}`` of ``A = {x >= 0: phi(x) > 0}``
    for ``phi(x) = C'^+_X(x) + C'^-_Y(K - x) + 1 = F_X(x) - (1/N) #{m :
    Y^{(m)} >= K - x}`` (Section 2.2), where ``F_X`` is the *continuous*
    marginal CDF (right derivative of the continuous call price) and the
    ``Y``-term is the empirical left derivative of the rearranged sample.

    Between two consecutive jumps of the ``Y``-term, ``phi`` is continuous
    and strictly increasing in ``x`` (since ``F_X`` is strictly increasing
    and the ``Y``-term is constant there), so each such segment has at most
    one zero crossing, located in closed form at ``x* = F_X^{-1}(c)`` for the
    segment's constant level ``c``. The ``Y``-term only ever jumps upward in
    ``x``, so the segment boundaries themselves can only be down-crossings.
    A single left-to-right sweep over the segments therefore recovers the
    whole ordered boundary list without evaluating ``phi`` on a point grid.
    """
    n = y_col.size
    y_sorted = np.sort(y_col)
    jump_x = kb - y_sorted
    candidates = np.unique(np.concatenate([jump_x, [0.0]]))
    candidates = candidates[candidates >= 0.0]
    if candidates.size == 0:
        candidates = np.array([0.0])
    levels = (n - np.searchsorted(y_sorted, kb - candidates, side="left")) / n

    boundaries: list[float] = []
    state_positive = False
    m = candidates.size
    for j in range(m):
        a = candidates[j]
        b = candidates[j + 1] if j + 1 < m else np.inf
        c = levels[j]
        f_a = float(marginal.cdf(a))
        sign_at_a = (f_a - c) > 1e-12
        if sign_at_a != state_positive:
            boundaries.append(float(a))
            state_positive = sign_at_a
        if not state_positive and 0.0 < c < 1.0:
            x_star = float(marginal.quantile(c))
            if a < x_star < b:
                boundaries.append(x_star)
                state_positive = True
    return np.asarray(boundaries[: 2 * (len(boundaries) // 2)])


def _merge_close_boundaries(boundaries: np.ndarray, rel_tol: float = 0.01) -> np.ndarray:
    """Collapse clusters of near-duplicate crossings of ``A_i`` into single
    effective intervals.

    At finite ``N`` the empirical left derivative of the call price on
    ``Y_i`` is a step function with up to ``N`` jumps of size ``O(1/N)``,
    while the continuous marginal CDF is smooth; where their difference
    ``phi_i`` is close to a genuine zero crossing, this finite-``N`` noise
    can make it cross back and forth several times within a region whose
    width shrinks as ``N`` grows, instead of crossing exactly once. The
    paper's examples (Tables 2-4) always report a single interval per asset
    (``n_i = 1``), so a cluster of crossings separated by much less than the
    spacing between genuinely distinct crossing regions is numerical noise
    around one true boundary and is merged into one interval here; this
    keeps the vertex-enumeration set of Proposition 2.3 at its intended
    size (``|B_i| = 2 n_i + 2``) instead of growing with the discretisation
    noise.
    """
    if boundaries.size < 4:
        return boundaries
    pairs = boundaries.reshape(-1, 2).tolist()
    span = boundaries.max() - boundaries.min()
    if span <= 0:
        return boundaries
    gap_tol = rel_tol * span
    merged = [pairs[0]]
    for lo, hi in pairs[1:]:
        if lo - merged[-1][1] <= gap_tol:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return np.asarray(merged, dtype=float).reshape(-1)


def strike_candidates(prices: np.ndarray, kb: float, marginals: Sequence,
                       asset_max: np.ndarray | None = None) -> list[np.ndarray]:
    """Boundaries ``K_i^1 < ... < K_i^{2n_i}`` for every asset (paper, Section
    2.2, step 3). If ``K_i^{2n_i}`` exceeds the largest atom of ``X_i`` it is
    snapped to ``kb`` (step 3d). Clusters of near-duplicate crossings caused
    by finite-``N`` noise (see ``_merge_close_boundaries``) are collapsed
    before the strike set is used downstream."""
    n_atoms, n = prices.shape
    out = []
    for i in range(n):
        y_col = prices.sum(axis=1) - prices[:, i]
        bnds = _phi_breakpoints(marginals[i], y_col, kb)
        bnds = _merge_close_boundaries(bnds)
        cap = asset_max[i] if asset_max is not None else prices[:, i].max()
        if bnds.size > 0 and bnds[-1] > cap:
            bnds = bnds.copy()
            bnds[-1] = kb
        out.append(bnds)
    return out


def f_payoff(z: np.ndarray | float, strikes: np.ndarray) -> np.ndarray | float:
    """Payoff ``f_i(z) = z + sum_a [(z - K^{2a})^+ - (z - K^{2a-1})^+]``
    (paper, Section 2.2) of the strike set ``strikes = (K^1, ..., K^{2n})``."""
    z = np.asarray(z, dtype=float)
    out = z.copy()
    for a in range(0, len(strikes), 2):
        k_odd, k_even = strikes[a], strikes[a + 1]
        out = out - np.maximum(z - k_odd, 0.0) + np.maximum(z - k_even, 0.0)
    return out


def analytical_cash(strikes: Sequence[np.ndarray], kb: float) -> float:
    """Exact cash component ``lambda_bar`` by vertex enumeration (Proposition
    2.3): ``min`` over the Cartesian product of breakpoint sets ``B_i = {0}
    union strikes_i union {K}`` of ``(sum x_i - K)^+ - sum_i f_i(x_i)``.
    """
    n = len(strikes)
    boxes = [np.unique(np.concatenate([[0.0], np.asarray(s, dtype=float), [kb]])) for s in strikes]
    f_vals = [f_payoff(box, strikes[i]) for i, box in enumerate(boxes)]
    grids = np.meshgrid(*boxes, indexing="ij")
    sums = sum(grids)
    basket_payoff = np.maximum(sums - kb, 0.0)
    f_grids = np.meshgrid(*f_vals, indexing="ij")
    f_sum = sum(f_grids)
    psi = basket_payoff - f_sum
    return float(psi.min())


@dataclass
class SubPortfolioResult:
    strikes: list[np.ndarray]          # phi-crossing strikes K_i^1 < ... < K_i^{2n_i}
    cash: float                        # lambda_bar
    value: float                       # HV on the discretised marginals
    linear_fallback: bool               # True if Pi_1^sub (Proposition 3.1) was returned


def sub_portfolio(prices: np.ndarray, kb: float, marginals: Sequence,
                   asset_max: np.ndarray | None = None) -> SubPortfolioResult:
    """Build the sub-replicating portfolio for one BRA/RA output (Algorithm 1,
    steps 3-5): pair-decomposition strikes, exact cash (Proposition 2.3), and
    the resulting hedging value on the discretised marginals.
    """
    n_atoms, n = prices.shape
    strikes = strike_candidates(prices, kb, marginals, asset_max)
    if any(s.size == 0 for s in strikes):
        # Proposition 3.1: no asset has a non-empty A_i -> linear portfolio.
        cash = -kb
        fmeans = [float(np.mean(prices[:, i])) for i in range(n)]
        return SubPortfolioResult(strikes=[np.empty(0) for _ in range(n)], cash=cash,
                                   value=sum(fmeans) + cash, linear_fallback=True)
    cash = analytical_cash(strikes, kb)
    fmeans = [float(np.mean(f_payoff(prices[:, i], strikes[i]))) for i in range(n)]
    return SubPortfolioResult(strikes=strikes, cash=cash, value=sum(fmeans) + cash, linear_fallback=False)


def port_value_discretised(prices: np.ndarray, strikes: Sequence[np.ndarray], kb: float) -> tuple[float, float, float]:
    """Hedging value ``HV = sum_i E[f_i(X_i)] + cash`` on the discretised
    marginals, returning ``(hv, cash, options_component)``."""
    cash = analytical_cash(strikes, kb)
    fmeans = [float(np.mean(f_payoff(prices[:, i], strikes[i]))) for i in range(prices.shape[1])]
    return sum(fmeans) + cash, cash, sum(fmeans)


def _pack(strikes: Sequence[np.ndarray], free_mask: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([s[m] for s, m in zip(strikes, free_mask)])


def _unpack(x: np.ndarray, strikes: Sequence[np.ndarray], free_mask: Sequence[np.ndarray]) -> list[np.ndarray]:
    out = []
    pos = 0
    for s, m in zip(strikes, free_mask):
        s2 = s.copy()
        k = int(m.sum())
        s2[m] = x[pos:pos + k]
        pos += k
        out.append(np.sort(s2))
    return out


def refine_strikes(
    strikes0: Sequence[np.ndarray],
    kb: float,
    mean_fn,
    fixed_at_boundary: bool = True,
    max_rounds: int = 8,
) -> tuple[list[np.ndarray], float]:
    """Derivative-free local search alternating Nelder-Mead and Powell on the
    map ``{K_i^j} -> HV({K_i^j})`` (paper, Section 2.2 "strike refinement"),
    with the number of strikes and their sign pattern fixed. ``mean_fn(i,
    strikes_i)`` returns ``E[f_i(X_i)]`` either on the discretised marginal or
    on the exact one (used for ``HV`` and ``HV^ex`` respectively).

    Strikes equal to ``0`` or ``kb`` are held fixed when ``fixed_at_boundary``
    is set, matching the paper's description that boundary strikes stay put.
    """
    strikes = [np.sort(np.asarray(s, dtype=float)) for s in strikes0]
    free_mask = [np.ones(s.size, dtype=bool) for s in strikes]
    if fixed_at_boundary:
        for s, m in zip(strikes, free_mask):
            m[(np.abs(s) < 1e-9) | (np.abs(s - kb) < 1e-9)] = False

    def objective(x: np.ndarray) -> float:
        cand = _unpack(x, strikes, free_mask)
        cash = analytical_cash(cand, kb)
        total = cash + sum(mean_fn(i, cand[i]) for i in range(len(cand)))
        return -total

    x0 = _pack(strikes, free_mask)
    if x0.size == 0:
        cash = analytical_cash(strikes, kb)
        hv = cash + sum(mean_fn(i, strikes[i]) for i in range(len(strikes)))
        return strikes, hv

    bounds = [(0.0, kb)] * x0.size
    best_val = -objective(x0)
    x_best = x0.copy()
    for _ in range(max_rounds):
        improved = False
        for method in ("Nelder-Mead", "Powell"):
            res = minimize(objective, x_best, method=method, bounds=bounds,
                            options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 500, "maxfev": 2000}
                            if method == "Nelder-Mead"
                            else {"xtol": 1e-8, "ftol": 1e-12, "maxiter": 50, "maxfev": 1000})
            if -res.fun > best_val + 1e-9:
                best_val = -res.fun
                x_best = res.x
                improved = True
        if not improved:
            break
    strikes_final = _unpack(x_best, strikes, free_mask)
    return strikes_final, best_val


def compare_with_strike_free(strikes: list[np.ndarray], hv: float, kb: float,
                              mean_x: float, calls_at_kb: list[float]) -> tuple[list[np.ndarray], float]:
    """Remark 2.10: retain the refined portfolio only if it beats both the
    linear portfolio ``sum_i x_i - K`` (worth ``mean_X - K``) and the
    strike-free portfolio ``sum_i (x_i - K)^+`` (worth ``sum_i C_{X_i}(K)``).

    In the ``f_i(z) = z + sum_a [(z-K^{2a})^+ - (z-K^{2a-1})^+]`` basis, the
    empty strike set gives ``f_i(z) = z`` (the linear/identity position),
    while ``{0, K}`` gives ``f_i(z) = (z - K)^+`` (the strike-free call, since
    ``z - max(z, 0) + max(z - K, 0) = (z - K)^+`` for ``z >= 0``).
    """
    linear_value = mean_x - kb
    calls_value = float(sum(calls_at_kb))
    if hv >= linear_value and hv >= calls_value:
        return strikes, hv
    if linear_value >= calls_value:
        return [np.empty(0) for _ in strikes], linear_value
    return [np.array([0.0, kb]) for _ in strikes], calls_value
