"""Rearrangement Algorithm (RA) and Block Rearrangement Algorithm (BRA).

Both approximate the dependence structure that minimises an objective of the
row sums of an ``N x n`` matrix of discretised marginal atoms, by repeatedly
permuting the rows within one column (RA, Puccetti and Ruschendorf (2012)) or
within a random subset of columns (BRA, paper Section 1 / Algorithm 1 step 2,
following Bernard, Bondarenko and Vanduffel (2018)) so that each column is
made antimonotone to the sum of the others (antimonotone coupling).

Adapted from the structure of the authors' companion code
(``subreplicating/rearrangement.py`` at
https://github.com/MohamadAbouDaya/Subreplication), rewritten here against the
block rearrangement procedure described in Section 1 and Remark 2.9 of the
paper.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def _variance_of_sums(row_sums: np.ndarray) -> float:
    """Default (variance) objective: ``Var(sum_i X_i)``."""
    return float(np.var(row_sums))


def basket_call_objective(strike: float) -> Callable[[np.ndarray], float]:
    """K-aware objective of Remark 2.9: ``E[(sum_i X_i - K)^+]``."""

    def objective(row_sums: np.ndarray) -> float:
        return float(np.mean(np.maximum(row_sums - strike, 0.0)))

    return objective


@dataclass
class RearrangementResult:
    X: np.ndarray       # rearranged N x n sample
    fval: float         # objective value achieved
    iterations: int


def _antimonotone_rearrange_pair(a: np.ndarray, b_sorted_desc_idx: np.ndarray) -> np.ndarray:
    """Return the permutation of ``a`` that is antimonotone to ``b`` (used by RA):
    the rank of ``a`` sorted ascending is matched to ``b`` sorted descending."""
    order_a = np.argsort(a, kind="mergesort")
    out = np.empty_like(order_a)
    out[order_a] = b_sorted_desc_idx
    return out


def ra(
    X0: np.ndarray,
    maxiter: int = 200_000,
    stalliter: int = 300,
    seed: int = 0,
    f: Callable[[np.ndarray], float] | None = None,
) -> RearrangementResult:
    """Rearrangement Algorithm (Puccetti and Ruschendorf, 2012): in each
    iteration pick one column at random and rearrange it to be antimonotone
    to the sum of the others, i.e. its rank order is reversed relative to the
    rank order of ``sum_{j != i} X_j``.
    """
    rng = np.random.default_rng(seed)
    n, d = X0.shape
    objective = f or _variance_of_sums
    X = X0.copy()
    for j in range(d):
        rng.shuffle(X[:, j])
    best_val = objective(X.sum(axis=1))
    no_improve = 0
    it = 0
    while it < maxiter and no_improve < stalliter:
        it += 1
        j = rng.integers(d)
        others = X.sum(axis=1) - X[:, j]
        order_others_desc = np.argsort(-others, kind="mergesort")
        # Antimonotone coupling: column j sorted ascending is paired with the
        # sum of the other columns sorted descending.
        col_sorted = np.sort(X[:, j])
        new_col = np.empty(n, dtype=X.dtype)
        new_col[order_others_desc] = col_sorted
        val = objective(others + new_col)
        if val < best_val - 1e-15:
            X[:, j] = new_col
            best_val = val
            no_improve = 0
        else:
            no_improve += 1
    return RearrangementResult(X=X, fval=best_val, iterations=it)


def bra(
    X0: np.ndarray,
    maxiter: int = 200_000,
    stalliter: int = 300,
    seed: int = 0,
    f: Callable[[np.ndarray], float] | None = None,
) -> RearrangementResult:
    """Block Rearrangement Algorithm: in each iteration randomly partition the
    columns into two blocks and rearrange the rows of one block so that its
    row sum is antimonotone to the row sum of the other block (paper, Section
    1; Bernard, Bondarenko and Vanduffel (2018)).  This generalises RA (whose
    blocks are single columns) and lets several columns move jointly, which
    reaches lower objective values for 3+ assets. The stall window
    ``stalliter`` ("a few hundred iterations", paper Section 5) stops the
    search once this many consecutive random partitions fail to improve the
    objective.
    """
    rng = np.random.default_rng(seed)
    n, d = X0.shape
    objective = f or _variance_of_sums
    X = X0.copy()
    for j in range(d):
        rng.shuffle(X[:, j])
    best_val = objective(X.sum(axis=1))
    no_improve = 0
    it = 0
    while it < maxiter and no_improve < stalliter:
        it += 1
        perm = rng.permutation(d)
        # random split into two non-empty blocks
        split = rng.integers(1, d) if d > 1 else 1
        block_a, block_b = perm[:split], perm[split:]
        sum_a = X[:, block_a].sum(axis=1)
        if len(block_b) == 0:
            no_improve += 1
            continue
        sum_b = X[:, block_b].sum(axis=1)
        order_b_desc = np.argsort(-sum_b, kind="mergesort")
        order_a_asc = np.argsort(sum_a, kind="mergesort")
        new_block_a = X[:, block_a][order_a_asc]
        rearranged_a = np.empty_like(new_block_a)
        rearranged_a[order_b_desc] = new_block_a
        candidate_sum = rearranged_a.sum(axis=1) + sum_b
        val = objective(candidate_sum)
        if val < best_val - 1e-15:
            X[:, block_a] = rearranged_a
            best_val = val
            no_improve = 0
        else:
            no_improve += 1
    return RearrangementResult(X=X, fval=best_val, iterations=it)
