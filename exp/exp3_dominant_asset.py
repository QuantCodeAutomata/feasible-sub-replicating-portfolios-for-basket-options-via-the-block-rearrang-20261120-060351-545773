"""Experiment 3: closed-form lower bounds for basket options with one
dominant asset (Table 4).

Tests Proposition 3.2: when the co-monotone/antimonotone coupling ``mu^c``
makes the basket sum cross the strike ``K`` only once from above, the
portfolio ``Pi_2^sub = min(x_1, a_1) + min(x_2, a_2) + (x_3 - c)^+ - a_1 - a_2``
is optimal, with the bound (13) equal to ``V(mu^c)`` (Corollary 2.6). Row 3
has a double crossing (bound (13) still attainable, below ``V(mu^c)``); row 4
is the three-asset example of Table 2, where the dominant-asset regime does
not apply.

For each row, ``V(mu^c)``, the bound (13) and the basket value ``BV`` found
by running the Block Rearrangement Algorithm / Rearrangement Algorithm
directly on the K-aware objective (Remark 2.9) are compared.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src import config
from src.dominant_asset import analyse_dominant_asset
from src.marginals import LogNormalMarginal, UniformMarginal
from src.rearrangement import basket_call_objective, bra, ra

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def bv_via_rearrangement(marginals, kb: float, n_atoms: int, n_restarts: int) -> float:
    """Smallest basket value ``BV`` found by BRA/RA directly minimising the
    K-aware objective ``E[(sum_i X_i - K)^+]`` (Remark 2.9), used here as an
    independent cross-check of the closed-form bound (13)."""
    X0 = np.column_stack([m.discretise(n_atoms) for m in marginals])
    objective = basket_call_objective(kb)
    best = np.inf
    for seed in range(n_restarts):
        for algo in (bra, ra):
            res = algo(X0, stalliter=config.BRA_STALL_WINDOW, seed=seed, f=objective)
            best = min(best, res.fval)
    return float(best)


def run_row(name: str, x1, x2, x3, kb: float, n_atoms: int, n_restarts: int) -> dict:
    res = analyse_dominant_asset(x1, x2, x3, kb, n_grid=max(n_atoms, 100_000))
    bv = bv_via_rearrangement([x1, x2, x3], kb, n_atoms, n_restarts)
    row = {
        "row": name,
        "K": kb,
        "crossings": res.crossings,
        "crossing_direction": res.crossing_direction,
        "u_star": res.u_star,
        "a1": res.a1,
        "a2": res.a2,
        "c": res.c,
        "V_mu_c": round(res.v_mu_c, 6),
        "bound13": round(res.bound13, 6) if res.bound13 is not None else None,
        "portfolio_value": round(res.portfolio_value, 6) if res.portfolio_value is not None else None,
        "BV": round(bv, 6),
    }
    logger.info("%s: crossings=%d dir=%s V(mu^c)=%s bound13=%s BV=%.6f",
                name, res.crossings, res.crossing_direction, row["V_mu_c"], row["bound13"], bv)
    return row


def run(n_atoms: int, n_restarts: int) -> list[dict]:
    rows = []

    # Row 1: X_1, X_2 ~ U[0, 30] (unweighted), X_3 log-normal with mean 40 and
    # volatility 100% over T=1 (Table 4 caption), K=70.
    x1 = UniformMarginal(config.EXP3_ROW1_UNIFORM_LO, config.EXP3_ROW1_UNIFORM_HI, 1.0)
    x2 = UniformMarginal(config.EXP3_ROW1_UNIFORM_LO, config.EXP3_ROW1_UNIFORM_HI, 1.0)
    x3 = LogNormalMarginal(config.EXP3_ROW1_X3_MEAN, config.EXP3_ROW1_X3_SIGMA, config.EXP3_ROW1_X3_T, 1.0)
    rows.append(run_row("row1_uniform_uniform_lognormal", x1, x2, x3,
                         config.EXP3_ROW1_K, n_atoms, n_restarts))

    # Row 2: log-normal, sigma = (10%, 10%, 120%), K=110.
    s1, s2, s3 = config.EXP3_ROW2_SIGMAS
    x1 = LogNormalMarginal(config.EXP3_S0, s1, config.T_HORIZON, 1 / 3)
    x2 = LogNormalMarginal(config.EXP3_S0, s2, config.T_HORIZON, 1 / 3)
    x3 = LogNormalMarginal(config.EXP3_S0, s3, config.T_HORIZON, 1 / 3)
    rows.append(run_row("row2_single_crossing", x1, x2, x3, config.EXP3_ROW2_K, n_atoms, n_restarts))

    # Row 3: log-normal, sigma = (20%, 20%, 60%), K=105 (double crossing).
    s1, s2, s3 = config.EXP3_ROW3_SIGMAS
    x1 = LogNormalMarginal(config.EXP3_S0, s1, config.T_HORIZON, 1 / 3)
    x2 = LogNormalMarginal(config.EXP3_S0, s2, config.T_HORIZON, 1 / 3)
    x3 = LogNormalMarginal(config.EXP3_S0, s3, config.T_HORIZON, 1 / 3)
    rows.append(run_row("row3_double_crossing", x1, x2, x3, config.EXP3_ROW3_K, n_atoms, n_restarts))

    # Row 4: the three-asset example of Table 2/Experiment 1 (K=100): the
    # dominant-asset regime does not apply (crossing from below).
    s1, s2, s3 = config.EXP3_ROW4_SIGMAS
    x1 = LogNormalMarginal(config.EXP3_S0, s1, config.T_HORIZON, 1 / 3)
    x2 = LogNormalMarginal(config.EXP3_S0, s2, config.T_HORIZON, 1 / 3)
    x3 = LogNormalMarginal(config.EXP3_S0, s3, config.T_HORIZON, 1 / 3)
    rows.append(run_row("row4_table2_example", x1, x2, x3, config.EXP3_ROW4_K, n_atoms, n_restarts))

    return rows


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=int, default=config.N_ATOMS_RUN)
    parser.add_argument("--restarts", type=int, default=config.EXP3_RESTARTS)
    args = parser.parse_args()

    logger.info("Experiment 3 (Table 4): n_atoms=%d, n_restarts=%d", args.atoms, args.restarts)
    rows = run(n_atoms=args.atoms, n_restarts=args.restarts)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "table4_dominant_asset.json"
    out_path.write_text(json.dumps({"n_atoms": args.atoms, "n_restarts": args.restarts, "rows": rows}, indent=2))
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
