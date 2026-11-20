"""Experiment 4: comparison with the Neufeld-Xiang (2022) grid bounds (Table 5).

Compares this paper's pair-decomposition bracket ``[HV^ex, BV^ex]`` (Algorithm
1, already produced by Experiments 1 and 2) against the grid-based dual lower
bound and Monte-Carlo-reassembled upper bound of Neufeld and Xiang (2022),
Section 5.2, for both the three-asset (Example 1) and six-asset (Example 2)
log-normal baskets, and runs the paper's augmented-grid validation (Section
5.2): adding this paper's own BRA-derived strikes to a coarse grid should
recover ``HV^ex`` almost exactly.

Six-asset note: the exact vertex set of ``grid_lower_bound`` (the full
Cartesian product of per-asset knot grids, plus hyperplane-edge slices) has
size of order ``m**6``, which is combinatorially intractable to assemble into
a single LP for knot counts beyond a handful. ``src.config`` therefore uses a
reduced ``EXP4_KNOTS_6ASSET_*_RUN = 6`` (see the comment there) for the
default (non-``--full``) run of this script; this script additionally caps
the Monte-Carlo reassembly discretisation and path count for the six-asset
case well below the three-asset defaults so every heavy call finishes in well
under two minutes (logged explicitly below).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from src import config
from src.basket import run_algorithm1
from src.marginals import LogNormalMarginal
from src.neufeld_xiang import (
    equal_probability_knots,
    equally_spaced_knots,
    grid_lower_bound,
    grid_reassembly_bound,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Script-own (not config.py) reductions for the six-asset Monte-Carlo
# reassembly bound, kept separate from EXP4_MC_PATHS_RUN/N_ATOMS_RUN so as
# not to collide with other experiments that reuse those config values: at
# the full N_ATOMS_RUN=50,000 atoms and EXP4_MC_PATHS_RUN=200,000 paths,
# BRA/RA on six columns plus Monte Carlo would not reliably finish within
# two minutes per strike on modest hardware.
SIX_ASSET_REASSEMBLY_ATOMS_RUN = 2_000
SIX_ASSET_REASSEMBLY_PATHS_RUN = 20_000
SIX_ASSET_REASSEMBLY_REPEATS_RUN = 3


def build_marginals_3asset() -> list[LogNormalMarginal]:
    """Log-normal marginals of Table 1, Example 1 (copied from
    ``exp/exp1_three_asset_basket.py::build_marginals``)."""
    return [
        LogNormalMarginal(s0=config.EXP1_S0, sigma=sigma, T=config.T_HORIZON, weight=w)
        for sigma, w in zip(config.EXP1_SIGMAS, config.EXP1_WEIGHTS)
    ]


def build_marginals_6asset() -> list[LogNormalMarginal]:
    """Log-normal marginals of Table 1, Example 2 (copied from
    ``exp/exp2_six_asset_basket.py::build_marginals``)."""
    return [
        LogNormalMarginal(s0=config.EXP2_S0, sigma=sigma, T=config.T_HORIZON, weight=w)
        for sigma, w in zip(config.EXP2_SIGMAS, config.EXP2_WEIGHTS)
    ]


def load_or_compute_bracket(
    results_path: Path,
    marginals: list[LogNormalMarginal],
    strikes: tuple[float, ...],
    n_restarts: int,
    n_atoms: int,
) -> dict[float, dict[str, float]]:
    """Per-strike ``{K: {"HV_ex": ..., "BV_ex": ...}}`` bracket from this
    paper's own Algorithm 1, reusing ``results_path`` if it already exists
    (produced by Experiment 1/2) to avoid recomputing an expensive BRA run,
    and falling back to :func:`run_algorithm1` only if it does not."""
    if results_path.exists():
        logger.info("Reusing existing bracket from %s", results_path)
        data = json.loads(results_path.read_text())
        return {row["K"]: {"HV_ex": row["HV_ex"], "BV_ex": row["BV_ex"]} for row in data["rows"]}

    logger.info("%s not found, computing bracket via run_algorithm1", results_path)
    out: dict[float, dict[str, float]] = {}
    for kb in strikes:
        t0 = time.time()
        res = run_algorithm1(marginals, kb, n_atoms=n_atoms, n_restarts=n_restarts,
                              stall_window=config.BRA_STALL_WINDOW, seed_salt=int(kb))
        logger.info("run_algorithm1(K=%.1f): HV_ex=%.5f BV_ex=%.5f (%.1fs)",
                     kb, res.hv_exact, res.bv_exact, time.time() - t0)
        out[kb] = {"HV_ex": res.hv_exact, "BV_ex": res.bv_exact}
    return out


def nx_bounds_for_strike(
    marginals: list[LogNormalMarginal],
    kb: float,
    m_eqprob: int,
    m_eqspace: int,
    eqprob_trunc: float,
    eqspace_trunc: float,
    mc_atoms: int,
    mc_paths: int,
    mc_repeats: int,
    label: str,
) -> dict[str, float]:
    """Neufeld-Xiang equal-probability-knot and equally-spaced-knot lower
    bounds, and the Monte-Carlo reassembly upper bound, for one strike."""
    row: dict[str, float] = {"K": kb}

    t0 = time.time()
    knots_eqprob = [equal_probability_knots(m, m_eqprob, eqprob_trunc) for m in marginals]
    lb_eqprob = grid_lower_bound(marginals, knots_eqprob, kb=kb)
    logger.info("[%s] K=%.1f eqprob m=%d: LB=%.6f n_constraints=%d (%.1fs)",
                 label, kb, m_eqprob, lb_eqprob.lower_bound, lb_eqprob.n_constraints, time.time() - t0)
    row["NX_eqprob_LB"] = lb_eqprob.lower_bound
    row["NX_eqprob_m"] = m_eqprob

    t0 = time.time()
    knots_eqspace = [equally_spaced_knots(m, m_eqspace, eqspace_trunc) for m in marginals]
    lb_eqspace = grid_lower_bound(marginals, knots_eqspace, kb=kb)
    logger.info("[%s] K=%.1f eqspace m=%d: LB=%.6f n_constraints=%d (%.1fs)",
                 label, kb, m_eqspace, lb_eqspace.lower_bound, lb_eqspace.n_constraints, time.time() - t0)
    row["NX_eqspace_LB"] = lb_eqspace.lower_bound
    row["NX_eqspace_m"] = m_eqspace

    t0 = time.time()
    ub_mean, ub_se = grid_reassembly_bound(marginals, n_atoms=mc_atoms, kb=kb,
                                            n_paths=mc_paths, n_repeats=mc_repeats, seed=int(kb))
    logger.info("[%s] K=%.1f reassembly UB=%.6f (SE=%.6f, atoms=%d, paths=%d, repeats=%d) (%.1fs)",
                 label, kb, ub_mean, ub_se, mc_atoms, mc_paths, mc_repeats, time.time() - t0)
    row["NX_reassembly_UB"] = ub_mean
    row["NX_reassembly_SE"] = ub_se
    return row


def run_comparison(full: bool) -> list[dict]:
    rows: list[dict] = []

    # --- three-asset (Example 1) ---
    m3_eqprob = config.EXP4_KNOTS_3ASSET_EQPROB_PAPER if full else config.EXP4_KNOTS_3ASSET_EQPROB_RUN
    m3_eqspace = config.EXP4_KNOTS_3ASSET_EQSPACE_PAPER if full else config.EXP4_KNOTS_3ASSET_EQSPACE_RUN
    mc_paths3 = config.EXP4_MC_PATHS_PAPER if full else config.EXP4_MC_PATHS_RUN
    n_restarts3 = config.EXP1_RESTARTS_PAPER if full else config.EXP1_RESTARTS_RUN

    marginals3 = build_marginals_3asset()
    bracket3 = load_or_compute_bracket(RESULTS_DIR / "table2_three_asset.json", marginals3,
                                        config.EXP1_STRIKES, n_restarts3, config.N_ATOMS_RUN)

    for kb in config.EXP1_STRIKES:
        row = nx_bounds_for_strike(
            marginals3, kb, m3_eqprob, m3_eqspace,
            config.EXP4_EQPROB_TRUNCATION, config.EXP4_EQSPACE_TRUNCATION,
            mc_atoms=config.N_ATOMS_RUN, mc_paths=mc_paths3, mc_repeats=config.EXP4_MC_REPEATS,
            label="3asset",
        )
        row["n_assets"] = 3
        row["this_paper_HV_ex"] = bracket3[kb]["HV_ex"]
        row["this_paper_BV_ex"] = bracket3[kb]["BV_ex"]
        rows.append(row)

    # --- six-asset (Example 2) ---
    m6_eqprob = config.EXP4_KNOTS_6ASSET_EQPROB_PAPER if full else config.EXP4_KNOTS_6ASSET_EQPROB_RUN
    m6_eqspace = config.EXP4_KNOTS_6ASSET_EQSPACE_PAPER if full else config.EXP4_KNOTS_6ASSET_EQSPACE_RUN
    mc_paths6 = config.EXP4_MC_PATHS_PAPER if full else SIX_ASSET_REASSEMBLY_PATHS_RUN
    mc_atoms6 = config.N_ATOMS_RUN if full else SIX_ASSET_REASSEMBLY_ATOMS_RUN
    mc_repeats6 = config.EXP4_MC_REPEATS if full else SIX_ASSET_REASSEMBLY_REPEATS_RUN
    n_restarts6 = config.EXP2_RESTARTS_PAPER if full else config.EXP2_RESTARTS_RUN

    if not full:
        logger.info(
            "Six-asset default (non-full) settings reduced below config._RUN for timing: "
            "mc_atoms=%d (vs N_ATOMS_RUN=%d), mc_paths=%d (vs EXP4_MC_PATHS_RUN=%d), "
            "mc_repeats=%d (vs EXP4_MC_REPEATS=%d)",
            mc_atoms6, config.N_ATOMS_RUN, mc_paths6, config.EXP4_MC_PATHS_RUN,
            mc_repeats6, config.EXP4_MC_REPEATS,
        )

    marginals6 = build_marginals_6asset()
    bracket6 = load_or_compute_bracket(RESULTS_DIR / "table3_six_asset.json", marginals6,
                                        config.EXP2_STRIKES, n_restarts6, config.N_ATOMS_RUN)

    for kb in config.EXP2_STRIKES:
        row = nx_bounds_for_strike(
            marginals6, kb, m6_eqprob, m6_eqspace,
            config.EXP4_EQPROB_TRUNCATION, config.EXP4_EQSPACE_TRUNCATION,
            mc_atoms=mc_atoms6, mc_paths=mc_paths6, mc_repeats=mc_repeats6,
            label="6asset",
        )
        row["n_assets"] = 6
        row["this_paper_HV_ex"] = bracket6[kb]["HV_ex"]
        row["this_paper_BV_ex"] = bracket6[kb]["BV_ex"]
        rows.append(row)

    return rows


def run_augmented_grid_check(full: bool) -> dict:
    """Section 5.2 augmented-grid validation: a coarse equal-probability grid
    for the three-asset example at K=100, augmented with this paper's own
    BRA-derived strikes (from ``results/table2_three_asset.json``), should
    recover ``HV^ex`` from the same file to within
    ``config.EXP4_AUGMENTED_TOLERANCE``; repeating with a refined knot count
    should not change the result materially."""
    kb = 100.0
    table2_path = RESULTS_DIR / "table2_three_asset.json"
    if not table2_path.exists():
        raise FileNotFoundError(
            f"{table2_path} is required for the augmented-grid check (run exp1 first)"
        )
    data = json.loads(table2_path.read_text())
    row100 = next(r for r in data["rows"] if r["K"] == kb)
    hv_ex_reference = row100["HV_ex"]
    bra_strikes = [row100["strikes_asset1"], row100["strikes_asset2"], row100["strikes_asset3"]]

    marginals3 = build_marginals_3asset()
    trunc = config.EXP4_EQPROB_TRUNCATION

    out: dict = {"K": kb, "reference_HV_ex": hv_ex_reference}
    for label, m_coarse in (
        ("coarse", config.EXP4_AUGMENTED_COARSE_KNOTS),
        ("refined", config.EXP4_AUGMENTED_REFINED_KNOTS),
    ):
        knots_list = []
        for i, marginal in enumerate(marginals3):
            base = equal_probability_knots(marginal, m_coarse, trunc)
            augmented = sorted(set(base.tolist()) | set(bra_strikes[i]))
            knots_list.append(np.asarray(augmented, dtype=float))

        t0 = time.time()
        res = grid_lower_bound(marginals3, knots_list, kb=kb)
        elapsed = time.time() - t0
        diff = abs(res.lower_bound - hv_ex_reference)
        logger.info("Augmented grid (%s, m=%d+strikes): LB=%.9f vs HV_ex=%.9f diff=%.3e (%.1fs)",
                     label, m_coarse, res.lower_bound, hv_ex_reference, diff, elapsed)
        out[f"{label}_m"] = m_coarse
        out[f"{label}_LB"] = res.lower_bound
        out[f"{label}_diff_from_HV_ex"] = diff
        out[f"{label}_within_tolerance"] = bool(diff <= config.EXP4_AUGMENTED_TOLERANCE)

    out["coarse_vs_refined_diff"] = abs(out["coarse_LB"] - out["refined_LB"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="Use the paper's knot counts and Monte Carlo path counts")
    args = parser.parse_args()

    logger.info("Experiment 4 (Table 5): full=%s", args.full)

    rows = run_comparison(full=args.full)
    RESULTS_DIR.mkdir(exist_ok=True)
    comparison_path = RESULTS_DIR / "table5_nx_comparison.json"
    comparison_path.write_text(json.dumps({"full": args.full, "rows": rows}, indent=2))
    logger.info("Saved %s", comparison_path)

    augmented = run_augmented_grid_check(full=args.full)
    augmented_path = RESULTS_DIR / "table5_augmented_grid.json"
    augmented_path.write_text(json.dumps(augmented, indent=2))
    logger.info("Saved %s", augmented_path)


if __name__ == "__main__":
    main()
