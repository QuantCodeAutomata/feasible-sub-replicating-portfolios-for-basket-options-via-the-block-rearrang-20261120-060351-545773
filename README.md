# Feasible sub-replicating portfolios for basket options via the block rearrangement algorithm

Replication of selected numerical experiments from *Feasible sub-replicating
portfolios for basket options via the block rearrangement algorithm*
(working paper, indexed source used for this replication).

The paper constructs **feasible** (finitely many strikes) sub-replicating
portfolios of vanilla options for basket calls when only the marginal laws of
the basket constituents are known and the dependence structure is unknown.
Algorithm 1 combines:

1. the Block Rearrangement Algorithm (BRA) / Rearrangement Algorithm (RA) to
   approximate the dependence structure minimising the basket value,
2. a pair-decomposition rule (Section 2.2) that reads off, for each asset, a
   small set of candidate strikes from the rearranged sample,
3. an exact cash component by vertex enumeration (Proposition 2.3),
4. a derivative-free local search that refines the strikes, and
5. a transfer step (Section 2.3, Propositions 2.3 and 2.11) that brackets the
   true (continuous-marginal) value between a lower bound ``HV^ex`` and an
   upper bound ``BV^ex``.

No real market data is used anywhere in this repository: every experiment is
defined on parametric marginals (log-normal, uniform, or truncated normal
mixtures) exactly as specified by the paper's Table 1 and Section 5, so there
is no `data/data_loader.py` and no API keys are required.

## Repository layout

```
src/            Algorithm 1 and its building blocks (shared by every experiment)
exp/            One script per reproduced experiment, each calling src/ only
tests/          Unit tests for src/ components
data/           data/generate_synthetic.py: truncated-normal-mixture marginals for Experiment 5
results/        Saved tables/metrics/plots and RESULTS.md (paper comparison)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest
```

All tests run on synthetic/parametric inputs; none depend on a network call.

## Experiments

Each experiment can be run at a reduced (default, fast) configuration or at
the paper's full configuration with `--full`:

```bash
python -m exp.exp1_three_asset_basket             # Table 2, three-asset basket call
python -m exp.exp1_three_asset_basket --full       # paper's R = 64 restarts

python -m exp.exp2_six_asset_basket                # Table 3, six-asset basket call
python -m exp.exp2_six_asset_basket --full          # paper's R = 20 restarts

python -m exp.exp3_dominant_asset                  # Table 4, closed-form dominant-asset bounds

python -m exp.exp4_neufeld_xiang                   # Table 5, comparison with Neufeld-Xiang (2022)
python -m exp.exp4_neufeld_xiang --full

python -m exp.exp5_truncated_mixture               # Table 6, truncated-normal-mixture marginals
```

`reproduce.sh` runs the tests and every experiment at its default (reduced)
configuration.

## Results

See `results/RESULTS.md` for the paper comparison, verdicts, deviations, and
items not reproduced.
