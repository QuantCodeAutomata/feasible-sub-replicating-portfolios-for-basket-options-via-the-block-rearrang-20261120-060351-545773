"""Transfer of the discretised-marginal hedging value to the original
marginals (paper, Section 2.3): the exact lower bound ``HV^ex`` (Corollary
2.12, via the closed-form call prices) and the reassembly upper bound
``BV^ex`` (Proposition 2.11).

Adapted from the structure of the authors' companion code
(``subreplicating/exact.py`` at
https://github.com/MohamadAbouDaya/Subreplication), rewritten here against
Section 2.3 and Proposition 2.11 of the paper.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
from scipy.integrate import quad
from scipy.special import roots_legendre

from .subreplication import analytical_cash


def hv_exact(strikes: Sequence[np.ndarray], kb: float, marginals: Sequence) -> tuple[float, float]:
    """``HV^ex = sum_i {E[X_i] + sum_j (-1)^j C_{X_i}(K_i^j)} + lambda_bar``
    (Section 2.3). Returns ``(hv_ex, cash)``; ``lambda_bar`` is the same
    vertex-enumeration cash as for the discretised marginals because it only
    depends on the strikes and ``K``, not on the marginal law (Proposition 2.3)."""
    cash = analytical_cash(strikes, kb)
    total = cash
    for marg, strikes_i in zip(marginals, strikes):
        e_fi = marg.mean
        for j, k in enumerate(strikes_i, start=1):
            e_fi += (-1) ** j * marg.call_price(float(k))
        total += e_fi
    return total, cash


def exact_mean_fn(marginals: Sequence):
    """Factory for ``mean_fn(i, strikes_i) -> E[f_i(X_i)]`` on the exact
    marginals, for use with :func:`src.subreplication.refine_strikes`."""

    def mean_fn(i: int, strikes_i: np.ndarray) -> float:
        marg = marginals[i]
        value = marg.mean
        for j, k in enumerate(strikes_i, start=1):
            value += (-1) ** j * marg.call_price(float(k))
        return value

    return mean_fn


def _gauss_legendre_unit(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_legendre(n_nodes)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def reassembled_basket_value(
    prices_r: np.ndarray,
    marginals: Sequence,
    kb: float,
    n_nodes: int = 16,
    n_nodes_kink: int = 96,
) -> float:
    """Upper bound ``BV^ex`` by reassembling the BRA/RA coupling with the
    original marginals (paper, Proposition 2.11).

    Each row ``m`` of the rearranged sample fixes, for every asset ``i``, the
    probability cell ``[(rank_i(m)-1)/N, rank_i(m)/N]``; within the cell the
    orientation ``epsilon_{m,i}`` (traversed with ``t`` increasing or
    decreasing) is chosen asset-by-asset to minimise the row's contribution to
    the basket-call integral. Rows whose cell cannot straddle ``K`` need no
    orientation search; rows that do are evaluated with a finer
    Gauss-Legendre rule, and rows touching the first or last quantile cell of
    any asset (at most ``2n`` of them) are refined by adaptive quadrature.
    """
    n_atoms, n = prices_r.shape
    ranks = np.empty((n_atoms, n), dtype=np.int64)
    for i in range(n):
        s = np.sort(prices_r[:, i])
        ranks[:, i] = np.searchsorted(s, prices_r[:, i]) + 1

    lo = np.empty_like(prices_r)
    hi = np.empty_like(prices_r)
    for i in range(n):
        lo[:, i] = marginals[i].quantile((ranks[:, i] - 1) / n_atoms)
        hi[:, i] = marginals[i].quantile(ranks[:, i] / n_atoms)
    lo_sum, hi_sum = lo.sum(axis=1), hi.sum(axis=1)

    contributions = np.zeros(n_atoms)
    above = lo_sum >= kb
    straddle = (lo_sum < kb) & (hi_sum > kb)
    # rows below K contribute 0 and are left at their default value

    t_nodes, t_weights = _gauss_legendre_unit(n_nodes)

    def cell_mean(idx: np.ndarray, asset: int, nodes: np.ndarray) -> np.ndarray:
        p = (ranks[idx, asset, None] - 1 + nodes[None, :]) / n_atoms
        return marginals[asset].quantile(p.ravel()).reshape(len(idx), nodes.size)

    if above.any():
        idx = np.where(above)[0]
        total = np.zeros(len(idx))
        for i in range(n):
            total += cell_mean(idx, i, t_nodes) @ t_weights
        contributions[idx] = total - kb

    if straddle.any():
        idx = np.where(straddle)[0]
        extreme = np.zeros(len(idx), dtype=bool)
        for i in range(n):
            extreme |= (ranks[idx, i] == 1) | (ranks[idx, i] == n_atoms)

        t_fine, w_fine = _gauss_legendre_unit(n_nodes_kink)
        best = np.full(len(idx), np.inf)
        for combo in itertools.product((0, 1), repeat=n - 1):
            eps = (0,) + combo
            total = np.zeros((len(idx), n_nodes_kink))
            for i in range(n):
                nodes_i = t_fine if eps[i] == 0 else 1.0 - t_fine
                total += cell_mean(idx, i, nodes_i)
            payoff = np.maximum(total - kb, 0.0)
            integral = payoff @ w_fine
            best = np.minimum(best, integral)
        contributions[idx] = best

        if extreme.any():
            ex_idx = idx[extreme]
            for row in ex_idx:
                def integrand(t: float, row=row) -> float:
                    best_val = np.inf
                    for combo in itertools.product((0, 1), repeat=n - 1):
                        eps = (0,) + combo
                        s = 0.0
                        for i in range(n):
                            ti = t if eps[i] == 0 else 1.0 - t
                            p = (ranks[row, i] - 1 + ti) / n_atoms
                            s += float(marginals[i].quantile(p))
                        best_val = min(best_val, max(s - kb, 0.0))
                    return best_val

                val, _ = quad(integrand, 0.0, 1.0, limit=200)
                contributions[row] = val

    return float(np.mean(contributions))
