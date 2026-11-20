"""Algorithm 1 end to end: discretisation, BRA/RA restarts, pair-decomposition
strikes, exact cash, strike refinement, and transfer to the original
marginals (paper, Algorithm 1 and Sections 2.2-2.3).

This is the single shared pipeline called by every experiment that reproduces
a basket-call sub-replication (Experiments 1, 2 and parts of 3-5): no
experiment script re-implements discretisation, rearrangement or the cash
component.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from . import config
from .exact import exact_mean_fn, hv_exact, reassembled_basket_value
from .rearrangement import basket_call_objective, bra, ra
from .subreplication import (
    analytical_cash,
    compare_with_strike_free,
    f_payoff,
    port_value_discretised,
    refine_strikes,
    strike_candidates,
    sub_portfolio,
)

logger = logging.getLogger(__name__)


@dataclass
class BasketResult:
    kb: float
    strikes_disc: list[np.ndarray]       # strikes refined on the discretised marginals
    cash_disc: float
    hv: float
    bv: float
    strikes_exact: list[np.ndarray]      # strikes refined with HV^ex as objective
    cash_exact: float
    hv_exact: float
    bv_exact: float
    linear_fallback: bool
    diagnostics: dict = field(default_factory=dict)


def _restart_seeds(n_restarts: int, salt: int) -> np.ndarray:
    return np.arange(n_restarts) + 1000 * salt


def best_rearrangement(
    marginals,
    kb: float,
    n_atoms: int,
    n_restarts: int,
    stall_window: int = config.BRA_STALL_WINDOW,
    seed_salt: int = 0,
) -> tuple[np.ndarray, float]:
    """Run BRA and RA with both the variance and the K-aware objective
    (Remark 2.9) over ``n_restarts`` random initialisations and keep the
    output with the highest resulting hedging value ``HV`` (Algorithm 1, step 5).
    """
    X0 = np.column_stack([m.discretise(n_atoms) for m in marginals])
    asset_max = X0.max(axis=0)
    objectives = {"variance": None, "k_aware": basket_call_objective(kb)}
    best_hv = -np.inf
    best_prices = None
    best_tag = None
    seeds = _restart_seeds(n_restarts, seed_salt)
    for seed in seeds:
        for algo_name, algo in (("bra", bra), ("ra", ra)):
            for obj_name, obj in objectives.items():
                res = algo(X0, stalliter=stall_window, seed=int(seed), f=obj)
                sp = sub_portfolio(res.X, kb, marginals, asset_max)
                if sp.value > best_hv:
                    best_hv = sp.value
                    best_prices = res.X
                    best_tag = f"{algo_name}/{obj_name}/seed={seed}"
    logger.info("best rearrangement for K=%s: %s, HV=%.6f", kb, best_tag, best_hv)
    return best_prices, best_hv


def run_algorithm1(marginals, kb: float, n_atoms: int = config.N_ATOMS_RUN,
                    n_restarts: int = 20, stall_window: int = config.BRA_STALL_WINDOW,
                    seed_salt: int = 0) -> BasketResult:
    """Full Algorithm 1 pipeline for one basket strike ``kb``."""
    prices_r, _ = best_rearrangement(marginals, kb, n_atoms, n_restarts, stall_window, seed_salt)
    asset_max = prices_r.max(axis=0)
    sp = sub_portfolio(prices_r, kb, marginals, asset_max)

    row_sums = prices_r.sum(axis=1)
    bv = float(np.mean(np.maximum(row_sums - kb, 0.0)))
    bv_ex = reassembled_basket_value(prices_r, marginals, kb,
                                      n_nodes=config.QUAD_NODES_REGULAR, n_nodes_kink=config.QUAD_NODES_KINK)
    mean_x = sum(m.mean for m in marginals)
    n_assets = len(marginals)
    # Remark 2.10's strike-free portfolio is sum_i (x_i - K)^+, using the same
    # basket strike K on every weighted asset x_i (see compare_with_strike_free
    # docstring for why this dominates the linear portfolio but not the basket
    # payoff, hence is a valid sub-replicator worth sum_i C_{X_i}(K)).
    calls_at_kb = [m.call_price(kb) for m in marginals]
    calls_at_kb_exact = calls_at_kb  # same exact Black-Scholes/closed-form call price

    if sp.linear_fallback:
        # Proposition 3.1 gives the linear portfolio sum_i x_i - K as *a*
        # valid sub-replicator when the pair decomposition fails for some
        # asset, but Remark 2.10 requires comparing it with the strike-free
        # portfolio sum_i w_i (S_i - K)^+ (worth sum_i C_{X_i}(K/n), always
        # >= 0) and keeping the larger (tighter) lower bound; the upper bound
        # (BV, BV^ex) is unaffected by which lower-bound portfolio is used and
        # is always computed from the same reassembly formulas as the general
        # case.
        strikes_final, hv_final, cash_final = compare_with_strike_free(
            sp.strikes, mean_x - kb, -kb, kb, mean_x, calls_at_kb, n_assets)
        return BasketResult(
            kb=kb, strikes_disc=strikes_final, cash_disc=cash_final, hv=hv_final, bv=bv,
            strikes_exact=strikes_final, cash_exact=cash_final, hv_exact=hv_final, bv_exact=bv_ex,
            linear_fallback=True,
            diagnostics={"note": "Proposition 3.1 linear fallback compared with Remark 2.10 "
                                  "strike-free portfolios; upper bound from the usual reassembly"},
        )

    def mean_fn_disc(i: int, strikes_i: np.ndarray) -> float:
        return float(np.mean(f_payoff(prices_r[:, i], strikes_i)))

    strikes_ref, _ = refine_strikes(sp.strikes, kb, mean_fn_disc, max_rounds=config.REFINE_MAX_ROUNDS)
    hv, cash, _ = port_value_discretised(prices_r, strikes_ref, kb)

    hv_ex, cash_ex = hv_exact(strikes_ref, kb, marginals)
    strikes_ex_ref, obj_ex = refine_strikes(strikes_ref, kb, exact_mean_fn(marginals),
                                             max_rounds=config.REFINE_MAX_ROUNDS)
    hv_ex_ref, cash_ex_ref = hv_exact(strikes_ex_ref, kb, marginals)
    if hv_ex_ref > hv_ex + config.HV_EXACT_IMPROVEMENT_TOL:
        strikes_ex_final, hv_ex_final, cash_ex_final = strikes_ex_ref, hv_ex_ref, cash_ex_ref
    else:
        strikes_ex_final, hv_ex_final, cash_ex_final = strikes_ref, hv_ex, cash_ex

    strikes_final, hv_final, cash_final = compare_with_strike_free(
        strikes_ref, hv, cash, kb, mean_x, calls_at_kb, n_assets)
    # Remark 2.10 applies equally to the exact bracket for the original
    # marginals: the exact lower bound must also beat the linear and
    # strike-free floors, evaluated with exact (continuous-marginal) call
    # prices rather than the discretised ones.
    strikes_ex_final, hv_ex_final, cash_ex_final = compare_with_strike_free(
        strikes_ex_final, hv_ex_final, cash_ex_final, kb, mean_x, calls_at_kb_exact, n_assets)

    return BasketResult(
        kb=kb, strikes_disc=strikes_final, cash_disc=cash_final, hv=hv_final, bv=bv,
        strikes_exact=strikes_ex_final, cash_exact=cash_ex_final, hv_exact=hv_ex_final, bv_exact=bv_ex,
        linear_fallback=False,
        diagnostics={"hv_bv_gap": bv - hv_final, "hv_ex_bv_ex_gap": bv_ex - hv_ex_final},
    )
