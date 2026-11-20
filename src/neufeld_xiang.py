"""Grid-based comparison bounds `a la Neufeld and Xiang (2022), used in Table 5
and Table 6 to compare this paper's pair-decomposition bracket with a
multi-marginal optimal-transport method that replaces each marginal
constraint by finitely many moment constraints on a grid of knots (paper,
Section 5.2).

Neufeld and Xiang (2022) solve a semi-infinite linear programme by a
cutting-plane scheme with a commercial solver (Gurobi) and reconstruct an
upper bound from the primal transport plan by Monte Carlo. This module
reimplements the dual (sub-replication) side of their method with the
open-source HiGHS backend (``scipy.optimize.linprog``):

  * each marginal is replaced by piecewise-linear "hat" functions on ``m``
    knots (Section 5.2);
  * the dual lower bound is ``max sum_i sum_j f_{i,j} w_{i,j}`` subject to
    ``sum_i f_i(x_i) <= (sum_i x_i - K)^+`` at a finite vertex set, solved by
    constraint generation (cutting planes), exactly as Section 5.2's
    augmented-grid check describes: "the vertices of the cells cut out by the
    knots and by the hyperplane {sum_i x_i = K}";
  * the comparison upper bound is obtained, as in Proposition 2.11, by
    reassembling the antimonotone coupling found by BRA/RA on the
    knot-rounded marginal with the true marginals and evaluating by Monte
    Carlo (Section 5.2: "obtained by reassembling the discrete optimiser with
    the original marginals, evaluated by Monte Carlo").

Full enumeration of the knot grid's cells is only tractable for a few assets;
for six assets the vertex set used for constraint generation is a random
sample rather than the full Cartesian product (documented in
``results/RESULTS.md`` as a deviation from an exhaustive cutting-plane
scheme).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .rearrangement import basket_call_objective, bra, ra


def equal_probability_knots(marginal, m: int, truncation: float) -> np.ndarray:
    """``m`` knots at equally spaced quantile levels, truncated at the
    ``truncation`` / ``1 - truncation`` quantiles (paper, Section 5.2)."""
    levels = np.linspace(truncation, 1.0 - truncation, m)
    return np.sort(np.asarray(marginal.quantile(levels), dtype=float))


def equally_spaced_knots(marginal, m: int, truncation: float) -> np.ndarray:
    """``m`` knots equally spaced in the support, truncated at the
    ``truncation`` / ``1 - truncation`` quantiles (paper, Section 5.2)."""
    lo = float(np.asarray(marginal.quantile(truncation)).item())
    hi = float(np.asarray(marginal.quantile(1.0 - truncation)).item())
    return np.linspace(lo, hi, m)


def hat_weights(marginal, knots: np.ndarray) -> np.ndarray:
    """True-marginal probability mass ``w_j = E_true[phi_j(X)]`` of each
    piecewise-linear hat basis function centred at knot ``j`` (the dual LP's
    objective coefficients). ``phi_j`` is the standard tent function of
    height 1 at ``knots[j]`` and 0 at its neighbours, extended flat beyond
    the first/last knot, so ``sum_j w_j = 1`` and ``sum_j w_j knots[j] =
    E[X]`` restricted to the hat basis."""
    knots = np.asarray(knots, dtype=float)
    m = knots.size
    edges = np.concatenate([[-np.inf], knots, [np.inf]])
    # Evaluate E[phi_j(X)] via E[X] decomposition on each hat's support by a
    # fine quadrature-free closed form using quantile sampling is avoided:
    # instead use the identity E[phi_j(X)] = integral of phi_j dF, computed
    # from the marginal's CDF at the knots (phi_j piecewise affine).
    cdf = np.array([float(np.asarray(marginal.cdf(k)).item()) for k in knots])
    # E[(X - a) indicator(a<=X<=b)] type integrals are avoided by using a
    # dense quantile grid (consistent with the paper's discretisation, N
    # large) instead of closed-form per-component integrals, to stay
    # distribution-agnostic (log-normal, uniform, truncated mixtures).
    grid_p = (np.arange(1, 50_000 + 1) - 0.5) / 50_000
    sample = np.asarray(marginal.quantile(grid_p), dtype=float)
    weights = np.zeros(m)
    for j in range(m):
        lo = knots[j - 1] if j > 0 else -np.inf
        hi = knots[j + 1] if j < m - 1 else np.inf
        tent = np.zeros_like(sample)
        left = (sample >= lo) & (sample <= knots[j])
        right = (sample >= knots[j]) & (sample <= hi)
        if j > 0:
            tent[left] = (sample[left] - lo) / (knots[j] - lo)
        else:
            tent[sample <= knots[j]] = 1.0
        if j < m - 1:
            tent[right] = np.maximum(tent[right], (hi - sample[right]) / (hi - knots[j]))
        else:
            tent[sample >= knots[j]] = 1.0
        weights[j] = tent.mean()
    return weights


def _interp_row(x: np.ndarray, knots_list: list[np.ndarray], offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sparse row of the LP constraint matrix for the point ``x`` (one value
    per asset): interpolation weights on the (at most) two neighbouring
    knots of every asset, flat beyond the first/last knot."""
    cols, vals = [], []
    for i, knots in enumerate(knots_list):
        xi = x[i]
        m = knots.size
        j = np.searchsorted(knots, xi)
        if j <= 0:
            cols.append(offsets[i]); vals.append(1.0)
        elif j >= m:
            cols.append(offsets[i] + m - 1); vals.append(1.0)
        else:
            k0, k1 = knots[j - 1], knots[j]
            w1 = 0.0 if k1 == k0 else (xi - k0) / (k1 - k0)
            cols.append(offsets[i] + j - 1); vals.append(1.0 - w1)
            cols.append(offsets[i] + j); vals.append(w1)
    return np.array(cols), np.array(vals)


def _candidate_vertices(knots_list: list[np.ndarray], kb: float, n_corner: int,
                         n_hyperplane: int, seed: int) -> np.ndarray:
    """Exact test points for the grid LP (Section 5.2 / augmented-grid
    check): the full Cartesian product of knot-grid corners, plus, for every
    choice of one "solved" asset, the full Cartesian product of the other
    assets' knots with the solved asset's coordinate set exactly on the
    hyperplane ``sum_i x_i = K`` (clipped to its knot range). This is needed
    because the sub-replication inequality must also hold where the convex
    payoff ``(sum_i x_i - K)^+`` changes regime, not only at knot-grid
    corners. ``n_corner``/``n_hyperplane`` and ``seed`` are accepted for a
    stable call signature but the construction here is exact (no sampling);
    see :func:`grid_lower_bound` for the constraint-generation path used
    when these arguments are non-zero.
    """
    n = len(knots_list)
    grids = np.meshgrid(*knots_list, indexing="ij")
    pts = [np.column_stack([g.ravel() for g in grids])]

    for free in range(n):
        others = [i for i in range(n) if i != free]
        other_grids = np.meshgrid(*(knots_list[i] for i in others), indexing="ij")
        other_pts = np.column_stack([g.ravel() for g in other_grids])
        free_val = np.clip(kb - other_pts.sum(axis=1), knots_list[free][0], knots_list[free][-1])
        block = np.empty((other_pts.shape[0], n))
        block[:, others] = other_pts
        block[:, free] = free_val
        pts.append(block)
    return np.concatenate(pts, axis=0)


def _exact_vertex_chunks(knots_list: list[np.ndarray], kb: float, chunk_size: int = 200_000):
    """Generator yielding the same exact vertex set as :func:`_candidate_vertices`
    (full Cartesian corners, then the per-asset hyperplane-edge intersections)
    in memory-bounded chunks of at most ``chunk_size`` rows, using index
    arithmetic (``np.unravel_index``) instead of ``np.meshgrid`` so the full
    ``prod(sizes)``-sized array is never materialised at once. Used by the
    constraint-generation (cutting-plane) path of :func:`grid_lower_bound` to
    scan dimensions (e.g. 6 assets) for which the full exact vertex set would
    not fit in memory, while still checking every exact vertex for violations."""
    n = len(knots_list)
    sizes = np.array([k.size for k in knots_list], dtype=np.int64)

    total_corner = int(np.prod(sizes))
    for start in range(0, total_corner, chunk_size):
        end = min(start + chunk_size, total_corner)
        multi = np.unravel_index(np.arange(start, end), sizes)
        yield np.column_stack([knots_list[i][multi[i]] for i in range(n)])

    for free in range(n):
        others = [i for i in range(n) if i != free]
        other_sizes = sizes[others]
        total_other = int(np.prod(other_sizes))
        for start in range(0, total_other, chunk_size):
            end = min(start + chunk_size, total_other)
            multi = np.unravel_index(np.arange(start, end), other_sizes)
            other_pts = np.column_stack([knots_list[others[k]][multi[k]] for k in range(len(others))])
            free_val = np.clip(kb - other_pts.sum(axis=1), knots_list[free][0], knots_list[free][-1])
            block = np.empty((other_pts.shape[0], n))
            block[:, others] = other_pts
            block[:, free] = free_val
            yield block


def _sampled_vertices(knots_list: list[np.ndarray], kb: float, n_corner: int,
                       n_hyperplane: int, seed: int) -> np.ndarray:
    """Random bootstrap vertex sample used to seed the constraint-generation
    path of :func:`grid_lower_bound`: ``n_corner`` points drawn by picking a
    uniformly random knot index per asset, and ``n_hyperplane`` points drawn
    the same way except that one randomly chosen "free" asset per point is
    set exactly on the hyperplane ``sum_i x_i = K`` (clipped to its knot
    range), mirroring the two blocks of :func:`_candidate_vertices` but
    without enumerating the full Cartesian product. Unlike the exact method,
    a finite random sample is not guaranteed to contain every constraint-
    defining vertex; :func:`grid_lower_bound` corrects for this by scanning
    the full exact vertex set (in chunks, via :func:`_exact_vertex_chunks`)
    for violations of the sampled solution and feeding the worst violators
    back in, so the random sample only affects how many cutting-plane
    rounds are needed, not the correctness of a converged result.

    The ``2**n`` deterministic extreme corners (every combination of the
    first/last knot per asset) and, for every asset, one "axis" vertex per
    knot (that asset at the knot, every other asset at its median knot) are
    always included, regardless of ``n_corner``/``n_hyperplane``: a purely
    random sample can by chance leave some knot's hat variable without any
    constraint that bounds it from above given the others, which makes the
    bootstrap LP unbounded (observed empirically for widely spaced
    equally-spaced knots); these deterministic vertices rule that out by
    ensuring every hat variable appears with full weight in at least one
    constraint."""
    n = len(knots_list)
    rng = np.random.default_rng(seed)
    extremes = np.array([k[0] for k in knots_list]), np.array([k[-1] for k in knots_list])
    corner_bits = np.array(list(np.ndindex(*([2] * n))))
    blocks = [np.where(corner_bits == 0, extremes[0], extremes[1])]
    medians = np.array([k[k.size // 2] for k in knots_list])
    for i, knots in enumerate(knots_list):
        axis = np.tile(medians, (knots.size, 1))
        axis[:, i] = knots
        blocks.append(axis)
    if n_corner > 0:
        cols = [knots_list[i][rng.integers(0, knots_list[i].size, size=n_corner)] for i in range(n)]
        blocks.append(np.column_stack(cols))
    if n_hyperplane > 0:
        free_idx = rng.integers(0, n, size=n_hyperplane)
        rows = np.empty((n_hyperplane, n))
        for i in range(n):
            mask = free_idx != i
            idx = rng.integers(0, knots_list[i].size, size=n_hyperplane)
            rows[:, i] = knots_list[i][idx]
            rows[~mask, i] = np.nan  # placeholder for the free asset, filled below
        others_sum = np.nansum(rows, axis=1)
        for i in range(n):
            mask = free_idx == i
            if not mask.any():
                continue
            other_sum_i = others_sum[mask]  # sum over the (already-filled) other assets
            rows[mask, i] = np.clip(kb - other_sum_i, knots_list[i][0], knots_list[i][-1])
        blocks.append(rows)
    if not blocks:
        return np.empty((0, n))
    return np.concatenate(blocks, axis=0)


@dataclass
class GridBoundResult:
    lower_bound: float
    node_values: list[np.ndarray]
    n_constraints: int


def _hat_eval(x: np.ndarray, knots: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Vectorised evaluation of the piecewise-linear interpolant defined by
    ``values`` at the ``knots`` (flat beyond the first/last knot), at every
    point of ``x``. Same interpolation rule as :func:`_interp_row`, used here
    to batch-evaluate ``sum_i f_i(x_i)`` for constraint-violation scanning."""
    m = knots.size
    j = np.searchsorted(knots, x)
    out = np.empty_like(x)
    left = j <= 0
    right = j >= m
    mid = ~left & ~right
    out[left] = values[0]
    out[right] = values[-1]
    if np.any(mid):
        jm = j[mid]
        k0, k1 = knots[jm - 1], knots[jm]
        w1 = np.where(k1 == k0, 0.0, (x[mid] - k0) / (k1 - k0))
        out[mid] = values[jm - 1] * (1.0 - w1) + values[jm] * w1
    return out


def _worst_violations(
    knots_list: list[np.ndarray],
    kb: float,
    node_values: list[np.ndarray],
    n_keep: int,
    chunk_size: int = 200_000,
) -> tuple[np.ndarray, float]:
    """Scan the full exact vertex set (:func:`_exact_vertex_chunks`) in
    memory-bounded chunks for the points where the current dual candidate
    ``sum_i f_i(x_i) - (sum_i x_i - K)^+`` is most positive (i.e. most
    violated), returning the ``n_keep`` worst points found and the largest
    violation seen. Used by the constraint-generation loop of
    :func:`grid_lower_bound` so large-dimension grids never need the full
    Cartesian product materialised for the LP itself."""
    n = len(knots_list)
    best_pts = np.empty((0, n))
    best_viol = np.empty((0,))
    worst_overall = -np.inf
    for chunk in _exact_vertex_chunks(knots_list, kb, chunk_size=chunk_size):
        total = np.zeros(chunk.shape[0])
        for i in range(n):
            total += _hat_eval(chunk[:, i], knots_list[i], node_values[i])
        payoff = np.maximum(chunk.sum(axis=1) - kb, 0.0)
        viol = total - payoff
        worst_overall = max(worst_overall, float(viol.max()))
        if n_keep > 0:
            cand_pts = np.concatenate([best_pts, chunk], axis=0)
            cand_viol = np.concatenate([best_viol, viol], axis=0)
            keep = np.argsort(cand_viol)[-n_keep:]
            best_pts, best_viol = cand_pts[keep], cand_viol[keep]
    return best_pts, worst_overall


def _solve_grid_lp(verts: np.ndarray, knots_list: list[np.ndarray], offsets: np.ndarray,
                    n_vars: int, w: np.ndarray, kb: float):
    """Build the sparse LP constraint matrix for the vertex set ``verts``
    (one row per vertex, via :func:`_interp_row`) and solve the Neufeld-Xiang
    dual lower bound with HiGHS."""
    from scipy.sparse import coo_matrix

    rows, cols, vals, rhs = [], [], [], []
    for r, x in enumerate(verts):
        c, v = _interp_row(x, knots_list, offsets)
        rows.extend([r] * c.size)
        cols.extend(c.tolist())
        vals.extend(v.tolist())
        rhs.append(max(float(x.sum() - kb), 0.0))

    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), n_vars))
    res = linprog(c=-w, A_ub=A, b_ub=np.array(rhs), bounds=(None, None), method="highs")
    if not res.success:
        raise RuntimeError(f"Neufeld-Xiang grid LP failed to solve: {res.message}")
    return res


# Internal convergence tolerance for the constraint-generation loop below: a
# round whose worst scanned violation is not larger than this is accepted as
# converged to the exact-vertex LP optimum (not a paper parameter, callers
# control precision via ``n_corner``/``n_hyperplane``/``refine_rounds``).
_CUTTING_PLANE_TOL = 1e-7


def grid_lower_bound(
    marginals,
    knots_list: list[np.ndarray],
    kb: float,
    n_corner: int = 0,
    n_hyperplane: int = 0,
    refine_rounds: int = 0,
    seed: int = 0,
) -> GridBoundResult:
    """Semi-infinite LP lower bound of Neufeld and Xiang (2022): maximise
    ``sum_i sum_j f_{i,j} w_{i,j}`` subject to ``sum_i f_i(x_i) <= (sum_i x_i
    - K)^+`` at the exact vertex set of :func:`_candidate_vertices` (paper,
    Section 5.2).

    With the defaults ``n_corner=0, n_hyperplane=0`` the full exact vertex
    set is enumerated and solved in a single LP, exactly as before -- the
    only tractable option for a handful of assets, and exact because, within
    each hyperrectangular grid cell, ``sum_i f_i(x_i)`` is affine and
    ``(sum_i x_i - K)^+`` is affine on each side of the ``sum_i x_i = K``
    hyperplane, so the inequality's worst case on a cell is always attained
    at one of :func:`_candidate_vertices`'s corners/hyperplane-edge points.

    For more assets (e.g. 6) or more knots per asset, the full Cartesian
    product is combinatorially intractable (``prod(sizes)`` rows): passing
    ``n_corner>0`` and/or ``n_hyperplane>0`` switches to a constraint-
    generation (cutting-plane) scheme instead: a random vertex sample of
    that size (:func:`_sampled_vertices`) bootstraps a small LP, then for up
    to ``refine_rounds`` rounds the *full* exact vertex set is scanned in
    memory-bounded chunks (:func:`_worst_violations`, never materialising
    ``prod(sizes)`` rows at once) for points violating the current dual
    candidate; the worst violators are added to the active vertex set and
    the LP is re-solved. The loop stops early once a round's worst scanned
    violation is within ``_CUTTING_PLANE_TOL``, at which point the result is
    the same exact-vertex optimum the brute-force method would have found.
    If it instead runs out of ``refine_rounds`` with a residual violation,
    the returned bound is only an upper estimate of that exact optimum
    (some exact-vertex constraints remain unchecked); call with more
    ``refine_rounds`` for a certified result.
    """
    n = len(marginals)
    sizes = [k.size for k in knots_list]
    offsets = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    n_vars = int(sum(sizes))
    w = np.concatenate([hat_weights(marginals[i], knots_list[i]) for i in range(n)])

    if n_corner == 0 and n_hyperplane == 0:
        verts = _candidate_vertices(knots_list, kb, n_corner, n_hyperplane, seed)
        res = _solve_grid_lp(verts, knots_list, offsets, n_vars, w, kb)
        n_constraints = verts.shape[0]
    else:
        verts = _sampled_vertices(knots_list, kb, n_corner, n_hyperplane, seed)
        res = _solve_grid_lp(verts, knots_list, offsets, n_vars, w, kb)
        for _ in range(max(refine_rounds, 0)):
            node_values = [res.x[offsets[i]:offsets[i] + sizes[i]] for i in range(n)]
            cap = max(n_corner + n_hyperplane, 1000)
            new_pts, worst = _worst_violations(knots_list, kb, node_values, n_keep=cap)
            if worst <= _CUTTING_PLANE_TOL:
                break
            verts = np.concatenate([verts, new_pts], axis=0)
            res = _solve_grid_lp(verts, knots_list, offsets, n_vars, w, kb)
        n_constraints = verts.shape[0]

    node_values = [res.x[offsets[i]:offsets[i] + sizes[i]] for i in range(n)]
    return GridBoundResult(lower_bound=float(w @ res.x), node_values=node_values, n_constraints=n_constraints)


def grid_reassembly_bound(
    marginals,
    n_atoms: int,
    kb: float,
    n_paths: int,
    n_repeats: int,
    seed: int = 0,
) -> tuple[float, float]:
    """Comparison upper bound: run BRA/RA directly on the K-aware objective
    (Remark 2.9) at ``n_atoms`` discretisation atoms (the grid-based method's
    own discretisation of the marginals) and reassemble with the true
    marginals by Monte Carlo (paper, Proposition 2.11 / Section 5.2
    description of the Neufeld-Xiang upper bound).
    """
    n = len(marginals)
    X0 = np.column_stack([m.discretise(n_atoms) for m in marginals])
    objective = basket_call_objective(kb)
    best_X, best_val = X0, np.inf
    for algo in (bra, ra):
        res = algo(X0, stalliter=300, seed=seed, f=objective)
        if res.fval < best_val:
            best_val, best_X = res.fval, res.X

    ranks = np.empty((n_atoms, n), dtype=np.int64)
    for i in range(n):
        ranks[:, i] = np.searchsorted(np.sort(best_X[:, i]), best_X[:, i]) + 1

    rng = np.random.default_rng(seed)
    means = np.empty(n_repeats)
    for r in range(n_repeats):
        row_idx = rng.integers(0, n_atoms, size=n_paths)
        total = np.zeros(n_paths)
        for i in range(n):
            p = (ranks[row_idx, i] - 1 + rng.random(n_paths)) / n_atoms
            total += np.asarray(marginals[i].quantile(p), dtype=float)
        means[r] = np.mean(np.maximum(total - kb, 0.0))
    se = float(means.std(ddof=1) / np.sqrt(n_repeats)) if n_repeats > 1 else 0.0
    return float(means.mean()), se
