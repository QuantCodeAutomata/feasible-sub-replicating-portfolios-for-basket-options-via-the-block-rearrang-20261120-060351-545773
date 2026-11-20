# AGENTS.md

Guide for contributors and coding agents working on this repository.

## What this repository implements

A replication of selected numerical experiments from *Feasible
sub-replicating portfolios for basket options via the block rearrangement
algorithm* (see `README.md` for the full citation and scope). The shared
algorithm (Algorithm 1 of the paper: Block Rearrangement Algorithm, pair
decomposition, exact cash component, strike refinement, transfer to the
original marginals) lives in `src/` and is reused, unchanged, by every
experiment script in `exp/`.

## Layout

- `src/` — flat modules: `config.py` (all paper-derived parameters),
  `marginals.py` (log-normal / uniform / mixture marginal classes and their
  mid-point quantile discretisation), `pricing.py` (vanilla option pricing,
  empirical and closed-form), `rearrangement.py` (RA and BRA), `exact.py`
  (transfer to the original marginals, Corollary 2.12 / Proposition 2.11),
  `subreplication.py` (pair decomposition, cash component, strike
  refinement), `basket.py` (the end-to-end Algorithm 1 pipeline),
  `dominant_asset.py` (closed-form bounds of Proposition 3.2), and
  `neufeld_xiang.py` (grid-based comparison method of Section 5.2).
- `exp/` — one script per reproduced table, each calling only the `src/`
  functions above.
- `data/generate_synthetic.py` — generates the truncated-normal-mixture
  marginals used by Experiment 5; no market data is used anywhere.
- `tests/` — unit tests for every `src/` module, run on small synthetic
  inputs; no network access required.
- `results/` — all tables, metrics and plots produced by the experiment
  scripts, plus `results/RESULTS.md` with the paper comparison.

## Commands

```bash
pip install -r requirements.txt
python -m pytest                       # all tests (no market_data marker is used: no market data)
python -m exp.exp1_three_asset_basket --full   # Table 2, paper's R = 64 restarts
python -m exp.exp2_six_asset_basket --full     # Table 3, paper's R = 20 restarts
python -m exp.exp3_dominant_asset
python -m exp.exp4_neufeld_xiang --full
python -m exp.exp5_truncated_mixture
```

`reproduce.sh` runs the test suite and every experiment at its default
(time-bounded) configuration.

## Conventions

- Every experiment parameter lives in `src/config.py` as a module-level
  `UPPER_CASE` variable with a comment naming its paper section and unit;
  no parameter is hard-coded inside `exp/` or `src/` logic.
- No real market data: this paper's experiments are defined entirely on
  parametric or synthetically generated marginal distributions.
- Re-running an experiment script overwrites its files in `results/`; the
  numbers and verdicts written by hand in `results/RESULTS.md` must be
  checked against the newly written files and updated after a rerun.
- Keep new modules flat under `src/`; experiment scripts only in `exp/`;
  tests only in `tests/`.
