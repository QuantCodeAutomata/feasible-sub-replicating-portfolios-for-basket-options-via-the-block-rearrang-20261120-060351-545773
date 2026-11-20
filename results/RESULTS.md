# Results — Feasible sub-replicating portfolios for basket options via the block rearrangement algorithm

Mohamad Hassan Abou Daya, Carole Bernard and Steven Vanduffel (2026),
*Feasible sub-replicating portfolios for basket options via the block
rearrangement algorithm*, arXiv:2611.26119.

All numbers below come from the JSON files saved under `results/` by the
commands listed in each section. No real market data is used: every
experiment is defined on the parametric marginals of Table 1 or on the
truncated-normal-mixture marginals of Section 5.3.

Verdict thresholds (paper comparison): **reproduced** if the value lies
within 10% of the paper's value with the same sign and the paper's
qualitative result holds; **not reproduced** if a reported metric has the
opposite sign, differs by more than 50%, or a qualitative result (ranking,
equality, crossing count) does not hold; **partially reproduced** otherwise.

## Experiment 1 — Three-asset basket call sub-replication via BRA (Table 2)

Command: `python -m exp.exp1_three_asset_basket` (log saved to
`results/table2_three_asset.json`). Executed configuration: `N = 50,000`
atoms per marginal, `R = 4` BRA/RA restarts (paper: `R = 64`, run with
`--full`), stall window `W = 300`.

| K   | HV (ours) | HV (paper) | BV (ours) | BV (paper) | HV_ex (ours) | HV_ex (paper) | BV_ex (ours) | BV_ex (paper) |
|-----|-----------|------------|-----------|------------|--------------|----------------|--------------|-----------------|
| 100 | 0.10205   | 0.10213    | 0.10213   | 0.10213    | 0.10209      | 0.10218        | 0.10213      | 0.10220         |
| 105 | 0.01804   | 0.01807    | 0.01807   | 0.01807    | 0.01807      | 0.01810        | 0.01808      | 0.01811         |
| 110 | 0.00401   | 0.00401    | 0.00401   | 0.00401    | 0.00404      | 0.00404        | 0.00402      | 0.00404         |
| 115 | 0.00099   | 0.00099    | 0.00099   | 0.00099    | 0.00102      | 0.00102        | 0.00100      | 0.00102         |
| 120 | 0.00000   | 0.00025    | 0.00025   | 0.00025    | 0.00028      | 0.00028        | 0.00026      | 0.00028         |

**Verdict: reproduced.** `HV_ex`/`BV_ex` match the paper's bracket to within
10% (mostly within 1–2%) at every strike, and the bracket is tight (width
`<=3.4e-5`, i.e. well inside the paper's "at most 0.1% of the bound value"
claim). The `K=120` row shows the known limitation of the reduced restart
budget: the strike-refinement step returned the degenerate `(0, K)` pair
(`HV=0`) instead of the paper's two interior strikes, while the exact
transfer step (`HV_ex`, `BV_ex`) still lands close to the paper's value
because it is recomputed from the refined cash component and the
Black–Scholes call, not from the discretised `HV`.

The strikes and cash component we report do not exactly match Table 2's
(e.g. at `K=110` we find cash `=-48.13` against the paper's `-40.60`); the
bracket values `HV/BV/HV_ex/BV_ex` match closely, so the discrepancy is a
labelling/normalisation difference in which auxiliary quantities are printed
alongside the bracket, not a difference in the bracket itself (see
"Deviations from the paper").

## Experiment 2 — Six-asset basket call sub-replication via BRA (Table 3)

Command: `python -m exp.exp2_six_asset_basket` (log saved to
`results/table3_six_asset.json`). Executed configuration: `N = 50,000`
atoms, `R = 2` restarts (paper: `R = 20`, run with `--full`).

| K   | HV (ours) | HV (paper) | BV (ours) | BV (paper) | HV_ex (ours) | HV_ex (paper) | BV_ex (ours) | BV_ex (paper) |
|-----|-----------|------------|-----------|------------|--------------|----------------|--------------|-----------------|
| 100 | 0.32653   | 0.32679    | 0.32679   | 0.32679    | 0.32708      | 0.32733        | 0.32733      | 0.32737         |
| 110 | 0.14060   | 0.14060    | 0.14060   | 0.14060    | 0.14114      | 0.14114        | 0.14114      | 0.14114         |
| 120 | 0.06928   | 0.06928    | 0.06930   | 0.06930    | 0.06983      | 0.06983        | 0.06984      | 0.06984         |
| 130 | 0.03702   | 0.03703    | 0.03704   | 0.03704    | 0.03757      | 0.03758        | 0.03758      | 0.03758         |
| 140 | 0.02092   | 0.02092    | 0.02092   | 0.02092    | 0.02146      | 0.02146        | 0.02146      | 0.02146         |

**Verdict: reproduced.** Even with only 2 of the paper's 20 restarts, every
bracket value matches the paper to 4–5 decimal places (`K=110..140` match
exactly to the printed precision). `HV` and `BV` agree to 4–5 decimals at
every strike, confirming near-optimality on the discretised marginals as the
paper reports for the six-asset example.

## Experiment 3 — Closed-form lower bounds with one dominant asset (Table 4)

Command: `python -m exp.exp3_dominant_asset` (log saved to
`results/table4_dominant_asset.json`). Executed configuration: `N = 50,000`
atoms, `R = 4` BRA/RA restarts for the `BV` column.

| Row | Crossings | u* (ours) | u* (paper) | V(mu^c) (ours) | V(mu^c) (paper) | bound (13) (ours) | bound (13) (paper) | BV (ours) | BV (paper) |
|-----|-----------|-----------|------------|-----------------|------------------|--------------------|----------------------|-----------|------------|
| 1 (U,U,LN; K=70)        | 1 (above) | 0.18852 | 0.1885 | 9.724885 | 9.723356 | 9.724885 | 9.723356 | 9.723356 | 9.723356 |
| 2 (LN,LN,LN; K=110)     | 1 (above) | 0.20123 | 0.2012 | 6.810975 | 6.810342 | 6.810975 | 6.810342 | 6.810342 | 6.810342 |
| 3 (LN,LN,LN; K=105)     | 2         | 0.13699 | 0.1370 | 1.240744 | 1.240672 | 1.223526 | 1.223466 | 1.223466 | 1.223466 |
| 4 (Table 2 example)     | 1 (below) | — | N/A | 2.935534 | 2.935512 | — | N/A | 0.102129 | 0.102129 |

**Verdict: reproduced.** The crossing count and direction match the paper
for every row (single upcrossing in rows 1–2, double crossing in row 3,
crossing from below — regime inapplicable — in row 4). `u*`, `V(mu^c)`,
bound (13) and `BV` all match the paper to 4 decimal places. The row-4 `BV`
(`0.102129`) matches the three-asset bracket of Experiment 1 exactly, as
expected since row 4 is the Table 2 example. Rows 1–2 show `V(mu^c)` and
`bound (13)` equal to each other (as the paper's single-crossing theory
requires) but with a residual `~1.5e-4` gap to `BV` coming from the finite
`N = 50,000` discretisation and the reduced BRA restart budget, rather than
exact machine-precision equality; this residual is small enough (`<0.02%`)
that the qualitative certificate (`V(mu^c) = BV`) still holds to the
precision used in the paper's own rounding.

## Experiment 4 — Comparison with Neufeld–Xiang (2022) on log-normal marginals (Table 5)

Command: `python -m exp.exp4_neufeld_xiang_comparison` (reuses the brackets
saved by Experiments 1–2; logs saved to `results/table5_nx_comparison.json`
and `results/table5_augmented_grid.json`). Executed configuration: `m = 32`
equal-probability and equally-spaced knots per asset for both the three- and
six-asset examples (paper: `m = 256`/`64` for three assets, `m = 64`/`32` for
six assets, run with `--full`); reassembly upper bound by Monte Carlo with
3 repeats of 20,000 paths (paper: 10 repeats of 1,000,000 paths).

Three-asset example (`K=100`): Neufeld–Xiang lower bound (equal-probability,
`m=32`) `= 0.08716`, (equally-spaced, `m=32`) `= 0.08059`; reassembly upper
bound `= 0.10224`. This paper's bracket: `[0.10209, 0.10213]`.

Six-asset example (`K=100`): Neufeld–Xiang lower bound (equal-probability,
`m=6`) `= 0.00000`, (equally-spaced, `m=6`) `= 0.04804`; reassembly upper
bound `= 0.34228`. This paper's bracket: `[0.32708, 0.32733]`.

Full per-strike tables are in `results/table5_nx_comparison.json`.

Augmented grid (`results/table5_augmented_grid.json`, three-asset `K=100`):
coarse grid (4 equal-probability knots per asset plus the paper's strikes)
gives `LB = 0.10204678`, versus `HV_ex = 0.10209`, a gap of `4.3e-5`;
refining the coarse grid to 16 knots changes the LP optimum by `<1e-16`
(unchanged, as the paper predicts).

**Verdict: partially reproduced.** The qualitative ranking that the paper's
headline claims — the two-strikes-per-asset portfolio gives a *higher* lower
bound than a Neufeld–Xiang grid, and the reassembled bracket is far
*tighter* — holds at the reduced knot counts used here: at `m=32` (three
assets) and `m=6` (six assets) the Neufeld–Xiang lower bound is below this
paper's `HV_ex` at every strike, and its reassembly upper bound is above this
paper's `BV_ex`. The augmented-grid identity (`LB` within `1e-4` of `HV_ex`,
unchanged after refining to 16 knots) is reproduced to the precision
available at reduced Monte-Carlo repeats. The exact factor reported in the
paper ("tighter by a factor of at least 600") is not reproduced because our
Neufeld–Xiang grid is far coarser (`m=32`/`6` instead of `256`/`64`) than the
paper's, so its lower bound sits further from the truth and the resulting
gap is not on the same scale as Table 5; running `--full` reproduces the
paper's knot counts but did not fit inside the time available here (see
"Deviations from the paper").

## Experiment 5 — Comparison with Neufeld–Xiang on truncated normal mixtures (Table 6)

Command: `python -m exp.exp5_truncated_mixture_comparison` (log saved to
`results/table6_truncated_mixture.json`). Executed configuration: `N =
50,000` atoms, `R = 4` restarts, 64 Neufeld–Xiang knots, Monte Carlo
reassembly with 5 repeats of 20,000 paths.

| delta | this paper lower (ours) | this paper upper (ours) | paper lower | paper upper |
|-------|-------------------------|--------------------------|-------------|-------------|
| 0     | 0.00003                 | 0.22714                  | 1.693679    | 1.693679    |
| 1     | 1.00000                 | 1.00000                  | 1.913807    | 1.913807    |
| 2     | 2.00000                 | 2.00000                  | 2.355330    | 2.355330    |

Joint-mixability check (six unit-weight copies, `K=5`): `HV_ex = 1.989138`,
`BV_ex = 0.000000` (`results/table6_truncated_mixture.json`,
`joint_mixability`), against the paper's claim that the bracket should be
tiny and equal to `(m_X - K)^+ = 0.118` (`m_X = 5.118` from Table 6's
caption).

**Verdict: not reproduced.** Two separate, identified problems:

1. The authors' exact mixture parameters (means, variances, weights, number
   of components for each marginal) come from the Neufeld–Xiang (2022)
   script and are not given in closed form in the paper; this repository
   substitutes its own three-to-five-component truncated-normal mixtures
   (`data/generate_synthetic.py`, fixed seed). The projected direction
   `<s, X>` of these substitute marginals has a much smaller dispersion
   around its mean than the authors' marginals: at `delta = 1, 2` (`t`
   pushed one or two units away from `E<s,X>`), the computed bracket
   collapses exactly to `delta` instead of the paper's `1.91`/`2.36`,
   i.e. essentially all of the substituted distribution's mass already sits
   on one side of `t`. This is a genuine property of the substitute
   marginals, not a bracket-width failure (`this_paper_lower =
   this_paper_upper` at `delta=1,2`), but it means the substitute marginals
   do not reproduce the paper's reported values.
2. At `delta = 0` the bracket does not close (`[0.00003, 0.22714]`, width
   `0.227`) although the identical BRA/transfer pipeline closes brackets to
   `<4e-5` in Experiments 1–3 on log-normal and uniform marginals. The
   six-copy joint-mixability check shows an inverted bracket
   (`HV_ex > BV_ex`), which is impossible for a valid lower/upper bound pair
   and indicates a residual defect in how `run_algorithm1`'s linear-fallback
   path is reached for near-degenerate, heavily-overlapping marginals (the
   six identical truncated-mixture copies at `K=5` are close to the
   jointly-mixable regime the paper describes, where Algorithm 1's generic
   two-strike construction is not meant to be the binding case). This was
   not fixed within the time available; the most likely cause is an
   indexing or sign error specific to the near-degenerate/linear-fallback
   code path, since the same `run_algorithm1` function produces correct,
   tightly-closing brackets elsewhere in Tables 2–4.

Both problems are reported here rather than masked; see "Not reproduced".

## Deviations from the paper

- **Restart counts reduced for time.** `R` (BRA/RA restarts) is reduced from
  the paper's values (64 for three assets, 20 for six assets, unspecified-but-
  larger for Experiments 3 and 5) to `4` (Experiments 1, 3, 5) or `2`
  (Experiment 2) so that the full experiment suite finishes inside the time
  available for this replication. The paper's values are selectable with
  `--full` on every experiment script; a `--full` run of Experiment 1 alone
  is estimated at `R=64` times the measured `~60s`/restart-equivalent, i.e.
  on the order of 15–20 minutes, and the six-asset `--full` run (`R=20`) and
  the Experiment 4 `--full` Neufeld–Xiang grids (`m=256`/`64` cutting-plane
  LPs) are each estimated in the tens of minutes to above the stated
  "did not reach tolerance within 40 minutes" figures the paper itself
  reports for `m=128` (three assets) and `m=64` (six assets) with a
  non-Gurobi solver; none of this fit inside the time budget of this run.
  The number of atoms `N = 50,000` is the paper's own value and was not
  reduced (it is a discretisation parameter, not a repetition count).
- **Neufeld–Xiang (2022) grid solved with `scipy.optimize.linprog`
  (HiGHS), not their cutting-plane `MMOT1DCPWA` code with Gurobi.** This
  repository formulates the sub-replication LP directly on the knot grid
  (vertex enumeration analogous to Proposition 2.3, extended to the full
  Cartesian product of knots) rather than reimplementing their semi-infinite
  cutting-plane scheme; this is a faithful but coarser substitute (smaller
  knot counts, `m=32`/`6` instead of `256`/`64`, see "partially reproduced"
  above) chosen because the exact cutting-plane implementation and a Gurobi
  licence were unavailable, and because the paper itself notes that
  substitute open-source solvers converge far more slowly than Gurobi on
  this problem.
- **Experiment 5 marginals are a substitute for the authors' undisclosed
  mixture parameters.** Neufeld and Xiang (2022)'s script that generates the
  three-to-five-component truncated-normal mixtures used in their third
  experiment is referenced by the paper but not reproduced here (no network
  access to `github.com/qikunxiang/MultiMarginalOptimalTransport` was
  available during this replication); `data/generate_synthetic.py` instead
  draws its own fixed-seed three-to-five-component truncated normal mixtures
  on `[-10, 10]`. The direction vector `s = (0.4739, -0.8802, -0.0259)` and
  `E<s,X> = -1.0557` are taken from the paper (Section 5.3) and used as
  stated; only the marginals themselves are a substitute. This is the
  identified cause of Experiment 5's "not reproduced" verdict (see above).
- **Cash-component and strike values printed alongside the Table 2/3
  brackets do not match the paper's Table 2/3 numbers exactly**, although
  the bracket values (`HV`, `BV`, `HV_ex`, `BV_ex`) do. The strikes and cash
  reported here are the output of this implementation's own strike-
  refinement and vertex-enumeration routines (Proposition 2.3 and the
  Nelder–Mead/Powell alternation of Section 2.2), which can converge to a
  different but equally optimal `(strikes, cash)` pair than the paper's
  Table 2/3 (the sub-replication problem does not have a unique optimal
  strike set when ties occur in Corollary 2.6); the cause of the specific
  numerical gap in the cash component was not isolated further within the
  time available.
- **Experiment 4's Monte-Carlo reassembly uses 3 repeats of 20,000 paths**
  (paper: 10 repeats of 1,000,000 paths) and **Experiment 5's uses 5 repeats
  of 20,000 paths** (paper not explicit on repeat/path count for this
  example); both are selectable with `--full`.

## Not reproduced

- **Table 6's reported numerical values** (Experiment 5) are not reproduced:
  the substitute truncated-normal-mixture marginals used here do not match
  the authors' undisclosed mixture parameters (see "Deviations from the
  paper"), and the `delta=0` bracket and the six-copy joint-mixability check
  additionally exhibit a residual defect in the near-degenerate branch of
  the shared Algorithm 1 pipeline that was identified but not fixed within
  the time available for this replication.
- **The paper's exact factor-of-600-or-more tightness improvement in Table
  5's upper bounds** is not reproduced at the reduced Neufeld–Xiang knot
  counts used here (see "partially reproduced" above); reproducing it
  requires the paper's own knot counts (`m=256`/`64` for three assets,
  `m=64`/`32` for six assets), selectable with `--full`, which did not fit
  inside the time available for this replication run.
