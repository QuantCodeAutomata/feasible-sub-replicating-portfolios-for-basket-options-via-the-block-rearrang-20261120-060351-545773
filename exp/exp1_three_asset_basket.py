"""Experiment 1: three-asset basket call sub-replication via BRA (Table 2).

Reproduces Algorithm 1 of the paper (discretisation, BRA/RA restarts, pair
decomposition, exact cash component, strike refinement, transfer to the
original log-normal marginals) for the three-asset example of Table 1
(Example 1) and the basket strikes of Section 5.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from src import config
from src.basket import run_algorithm1
from src.marginals import LogNormalMarginal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_marginals() -> list[LogNormalMarginal]:
    """Log-normal marginals of Table 1, Example 1, weighted by ``w_i = 1/3``."""
    return [
        LogNormalMarginal(s0=config.EXP1_S0, sigma=sigma, T=config.T_HORIZON, weight=w)
        for sigma, w in zip(config.EXP1_SIGMAS, config.EXP1_WEIGHTS)
    ]


def run(n_restarts: int, n_atoms: int) -> list[dict]:
    marginals = build_marginals()
    rows = []
    for kb in config.EXP1_STRIKES:
        t0 = time.time()
        res = run_algorithm1(marginals, kb, n_atoms=n_atoms, n_restarts=n_restarts,
                              stall_window=config.BRA_STALL_WINDOW, seed_salt=int(kb))
        elapsed = time.time() - t0
        logger.info("K=%.1f: HV=%.5f BV=%.5f HV^ex=%.5f BV^ex=%.5f (%.1fs)",
                     kb, res.hv, res.bv, res.hv_exact, res.bv_exact, elapsed)
        rows.append({
            "K": kb,
            "strikes_asset1": [round(float(s), 4) for s in res.strikes_disc[0]] if not res.linear_fallback else [],
            "strikes_asset2": [round(float(s), 4) for s in res.strikes_disc[1]] if not res.linear_fallback else [],
            "strikes_asset3": [round(float(s), 4) for s in res.strikes_disc[2]] if not res.linear_fallback else [],
            "cash": round(res.cash_disc, 4),
            "HV": round(res.hv, 5),
            "BV": round(res.bv, 5),
            "HV_ex": round(res.hv_exact, 5),
            "BV_ex": round(res.bv_exact, 5),
            "bracket_width": round(res.bv_exact - res.hv_exact, 8),
            "linear_fallback": res.linear_fallback,
            "elapsed_sec": round(elapsed, 2),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use the paper's R=64 restarts")
    parser.add_argument("--restarts", type=int, default=None, help="Override the restart count")
    parser.add_argument("--atoms", type=int, default=config.N_ATOMS_RUN, help="Discretisation atoms per marginal")
    args = parser.parse_args()

    n_restarts = args.restarts or (config.EXP1_RESTARTS_PAPER if args.full else config.EXP1_RESTARTS_RUN)
    logger.info("Experiment 1 (Table 2): n_restarts=%d, n_atoms=%d", n_restarts, args.atoms)

    rows = run(n_restarts=n_restarts, n_atoms=args.atoms)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "table2_three_asset.json"
    out_path.write_text(json.dumps({"n_restarts": n_restarts, "n_atoms": args.atoms, "rows": rows}, indent=2))
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
