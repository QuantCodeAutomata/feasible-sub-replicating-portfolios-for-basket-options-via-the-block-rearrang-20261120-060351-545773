"""Experiment 5: comparison with Neufeld-Xiang (2022) multi-marginal optimal
transport on truncated normal mixture marginals (Table 6), plus a
joint-mixability check with six unit-weight copies (paper, Section 5.3).

Algebraic identity used to convert the basket-call bracket on the
non-negative transformed assets ``y_i`` into a bracket for ``E[|<s,X> - t|]``:

    |z - t| = 2*(z - t)^+ - (z - t)                                  ... (i)

for any real random variable ``z`` and constant ``t`` (check: if ``z >= t``,
both sides equal ``z - t``; if ``z < t``, both sides equal ``t - z``).

Let ``z = <s, X> = sum_i s_i x_i`` (the direction-``s`` projection of the
base marginals), ``y_i = s_i x_i + 10|s_i|`` and ``kb = t + 10*sum_i|s_i|``.
Then

    sum_i y_i - kb = sum_i s_i x_i + 10*sum_i|s_i| - t - 10*sum_i|s_i|
                    = sum_i s_i x_i - t = z - t                       ... (ii)

so, as random variables, ``(sum_i y_i - kb)^+ = (z - t)^+`` and
``sum_i y_i - kb = z - t``. Substituting (ii) into (i):

    |z - t| = 2*(sum_i y_i - kb)^+ - (sum_i y_i - kb).

Taking expectations and noting ``E[sum_i y_i - kb] = sum_i E[y_i] - kb`` is a
constant fixed by the marginals alone (linear, hence coupling-independent):

    E[|z - t|] = 2*E[(sum_i y_i - kb)^+] - (sum_i E[y_i] - kb).

Writing ``cost(V) = 2*V - (sum_i E[y_i] - kb)`` for any value ``V`` that
brackets the basket-call price ``E[(sum_i y_i - kb)^+]`` on the ``y_i``
(e.g. ``V = hv_exact`` or ``V = bv_exact`` from Algorithm 1, or the
Neufeld-Xiang grid bounds), ``cost`` is monotone increasing in ``V``; so a
sub-/super-replication bracket ``hv_exact <= BV_true <= bv_exact`` on the
``y``-basket maps to a bracket ``cost(hv_exact) <= E[|<s,X> - t|] <=
cost(bv_exact)`` on the cost scale, preserving the ordering.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from data.generate_synthetic import generate_marginals
from src import config
from src.basket import run_algorithm1
from src.marginals import AffineMarginal
from src.neufeld_xiang import equal_probability_knots, grid_lower_bound, grid_reassembly_bound

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_y_marginals(base_marginals: list) -> list[AffineMarginal]:
    """``y_i = s_i * x_i + 10 * |s_i|`` (paper, Section 5.3): a monotone
    affine transform making every ``y_i`` non-negative on ``x_i in [-10, 10]``."""
    return [
        AffineMarginal(base=x_i, a=s_i, b=10.0 * abs(s_i))
        for x_i, s_i in zip(base_marginals, config.EXP5_DIRECTION)
    ]


def cost_transform(v: float, mean_sum: float, kb: float) -> float:
    """``cost(V) = 2*V - (sum_i E[y_i] - kb)`` (see module docstring)."""
    return 2.0 * v - (mean_sum - kb)


def run_delta_row(
    delta: float,
    base_marginals: list,
    n_restarts: int,
    nx_knots: int,
    mc_paths: int,
    mc_repeats: int,
) -> dict:
    # Section 5.3 states E<s,X> = -1.0557 for the authors' undisclosed mixture
    # parameters. Since the mixtures used here are a substitute (data/generate_
    # synthetic.py; the authors' exact parameters could not be retrieved, see
    # results/RESULTS.md), deltas are offsets from this run's *actual* E<s,X>
    # rather than from the paper's constant, so that t = E<s,X> + delta still
    # probes the distribution around its true centre instead of landing, by
    # construction, deep in the money or deep out of the money.
    y_marginals = build_y_marginals(base_marginals)
    mean_sum = sum(y.mean for y in y_marginals)
    e_sx_actual = mean_sum - 10.0 * sum(abs(s) for s in config.EXP5_DIRECTION)
    t = e_sx_actual + delta
    kb = t + 10.0 * sum(abs(s) for s in config.EXP5_DIRECTION)

    seed_salt = int(delta * 1000)
    t0 = time.time()
    res = run_algorithm1(
        y_marginals, kb, n_atoms=config.N_ATOMS_RUN, n_restarts=n_restarts,
        stall_window=config.BRA_STALL_WINDOW, seed_salt=seed_salt,
    )
    this_paper_lower = cost_transform(res.hv_exact, mean_sum, kb)
    this_paper_upper = cost_transform(res.bv_exact, mean_sum, kb)

    knots = [equal_probability_knots(y_i, nx_knots, 1e-12) for y_i in y_marginals]
    nx_lb = grid_lower_bound(y_marginals, knots, kb).lower_bound
    # Reduced Monte Carlo path/repeat count relative to the paper's Table 6
    # run (Section 5.3), controlled by mc_paths/mc_repeats (see --full).
    nx_ub_mean, nx_ub_se = grid_reassembly_bound(
        y_marginals, n_atoms=config.N_ATOMS_RUN, kb=kb,
        n_paths=mc_paths, n_repeats=mc_repeats, seed=seed_salt,
    )
    nx_lower = cost_transform(nx_lb, mean_sum, kb)
    nx_upper = cost_transform(nx_ub_mean, mean_sum, kb)
    nx_upper_se = 2.0 * nx_ub_se  # additive constant in cost_transform has zero variance

    elapsed = time.time() - t0
    logger.info(
        "delta=%.2f t=%.4f kb=%.4f: this_paper=[%.5f, %.5f] nx=[%.5f, %.5f +/- %.5f] (%.1fs)",
        delta, t, kb, this_paper_lower, this_paper_upper, nx_lower, nx_upper, nx_upper_se, elapsed,
    )
    return {
        "delta": delta,
        "t": t,
        "kb": kb,
        "e_sx_actual": round(e_sx_actual, 6),
        "this_paper_lower": round(this_paper_lower, 6),
        "this_paper_upper": round(this_paper_upper, 6),
        "nx_lower": round(nx_lower, 6),
        "nx_upper": round(nx_upper, 6),
        "nx_upper_se": round(nx_upper_se, 6),
        "elapsed_sec": round(elapsed, 2),
    }


def run_joint_mixability_check(n_restarts: int) -> dict:
    """Six unit-weight copies at ``K = 5`` (paper, Section 5.3): the paper
    claims these marginals are jointly mixable to numerical accuracy, i.e.
    ``hv_exact`` and ``bv_exact`` should agree closely with each other and
    with the strike-free theoretical value ``max(sum_i E[m_i] - kb, 0)``."""
    marginals = generate_marginals(6, config.EXP5_SEED + config.EXP5_JOINT_SEED_OFFSET)
    kb = 5.0
    t0 = time.time()
    res = run_algorithm1(
        marginals, kb, n_atoms=config.N_ATOMS_RUN, n_restarts=n_restarts,
        stall_window=config.BRA_STALL_WINDOW, seed_salt=0,
    )
    elapsed = time.time() - t0
    theoretical_value = max(sum(m.mean for m in marginals) - kb, 0.0)
    bracket_width = res.bv_exact - res.hv_exact
    small_bracket = bracket_width < 1e-3
    logger.info(
        "Joint-mixability check (K=5): HV^ex=%.6f BV^ex=%.6f theoretical=%.6f "
        "bracket_width=%.2e small_bracket=%s (%.1fs)",
        res.hv_exact, res.bv_exact, theoretical_value, bracket_width, small_bracket, elapsed,
    )
    return {
        "kb": kb,
        "hv_exact": round(res.hv_exact, 6),
        "bv_exact": round(res.bv_exact, 6),
        "theoretical_value": round(theoretical_value, 6),
        "bracket_width": round(bracket_width, 8),
        "small_bracket": small_bracket,
        "elapsed_sec": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="Use the paper's knot count and larger restart/Monte-Carlo settings")
    args = parser.parse_args()

    n_restarts = config.EXP5_RESTARTS_PAPER if args.full else config.EXP5_RESTARTS_RUN
    joint_restarts = config.EXP5_JOINT_RESTARTS_PAPER if args.full else config.EXP5_JOINT_RESTARTS_RUN
    nx_knots = config.EXP5_NX_KNOTS_PAPER if args.full else config.EXP5_NX_KNOTS_RUN
    mc_paths = config.EXP5_MC_PATHS_PAPER if args.full else config.EXP5_MC_PATHS_RUN
    mc_repeats = config.EXP5_MC_REPEATS_PAPER if args.full else config.EXP5_MC_REPEATS_RUN
    logger.info(
        "Experiment 5 (Table 6): n_restarts=%d, nx_knots=%d, mc_paths=%d, mc_repeats=%d",
        n_restarts, nx_knots, mc_paths, mc_repeats,
    )

    base_marginals = generate_marginals(config.EXP5_N_MARGINALS, config.EXP5_SEED)
    rows = [
        run_delta_row(delta, base_marginals, n_restarts, nx_knots, mc_paths, mc_repeats)
        for delta in config.EXP5_DELTAS
    ]

    joint_row = None
    if config.EXP5_SIX_MARGINAL_CHECK:
        joint_row = run_joint_mixability_check(joint_restarts)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "table6_truncated_mixture.json"
    out_path.write_text(json.dumps({
        "n_restarts": n_restarts,
        "nx_knots": nx_knots,
        "mc_paths": mc_paths,
        "mc_repeats": mc_repeats,
        "rows": rows,
        "joint_mixability": joint_row,
    }, indent=2))
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
