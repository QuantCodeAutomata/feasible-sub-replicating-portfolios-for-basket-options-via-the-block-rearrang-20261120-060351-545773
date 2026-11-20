"""Paper-derived parameters for every experiment. No classes or dicts: every
module-level variable is used directly by ``exp/*`` and carries a one-line
source note (paper section/equation, or "not specified in the paper").
"""

# ---------------------------------------------------------------------------
# Common discretisation (paper, Section 5: "each marginal discretised into
# N = 50,000 equiprobable atoms placed at mid-point quantiles")
# ---------------------------------------------------------------------------
N_ATOMS_PAPER = 50_000   # atoms per marginal, paper Section 5
N_ATOMS_RUN = 50_000     # atoms actually used; kept at the paper value (method parameter, not reduced)

T_HORIZON = 0.5          # option maturity in years, Table 1 / Section 5
RISK_FREE_RATE = 0.0     # paper uses zero interest rates throughout Section 5

# Local-search tolerances for the strike refinement (Section 2.2): the paper
# states the procedure alternates Nelder-Mead and Powell "until neither
# method improves HV" but does not give numerical tolerances -> chosen
# conservatively tight (not specified in the paper).
REFINE_MAX_ROUNDS = 8
REFINE_XATOL = 1e-6
REFINE_FATOL = 1e-10
HV_EXACT_IMPROVEMENT_TOL = 1e-6  # paper, Section 2.3: "retain the second run if it raises HV^ex by more than 1e-6"

# Reassembly quadrature (Proposition 2.11): node counts are not specified in
# the paper beyond "more nodes on rows where the basket crosses K"; chosen so
# that re-running with 4x as many nodes changes BV^ex by < 1e-10, as the
# paper reports for its own (unspecified) choice.
QUAD_NODES_REGULAR = 16
QUAD_NODES_KINK = 96

# Stall window for BRA/RA (paper, Section 5: "a few hundred iterations");
# the exact value is not specified in the paper.
BRA_STALL_WINDOW = 300
BRA_MAX_ITER = 2_000_000

# ---------------------------------------------------------------------------
# Experiment 1: three-asset basket call (Table 2 / Example 1, Table 1)
# ---------------------------------------------------------------------------
EXP1_SIGMAS = (0.355, 0.20, 0.25)     # log-normal volatilities, Table 1 Example 1
EXP1_S0 = 100.0                       # S_{i,0}, Table 1
EXP1_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)  # equal basket weights w_i = 1/3
EXP1_STRIKES = (100.0, 105.0, 110.0, 115.0, 120.0)  # basket strikes K, Section 5
EXP1_RESTARTS_PAPER = 64              # R restarts used in the tables, Section 5
# Restarts actually run: measured at ~70s per restart (4 BRA/RA calls x pair
# decomposition) at N=50,000 atoms, R=64 would take about 6 hours for the five
# strikes of Table 2, exceeding the time available for this replication.
# HV already agrees with R=64 to 3-4 decimal places at R=1 in a timing probe
# (Section 5 reports R=64 only as the value used to build the tables, not as
# a value required for convergence); R is therefore reduced to 4. Run
# `python -m exp.exp1_three_asset_basket --full` for the paper's R=64.
EXP1_RESTARTS_RUN = 4

# ---------------------------------------------------------------------------
# Experiment 2: six-asset basket call (Table 3 / Example 2, Table 1)
# ---------------------------------------------------------------------------
EXP2_SIGMAS = (0.915, 0.755, 0.825, 0.355, 0.20, 0.15)  # Table 1 Example 2
EXP2_S0 = 100.0
EXP2_WEIGHTS = tuple(1 / 6 for _ in range(6))
EXP2_STRIKES = (100.0, 110.0, 120.0, 130.0, 140.0)
EXP2_RESTARTS_PAPER = 20              # R restarts used in the tables, Section 5
# Restarts actually run: measured at ~180s per restart at N=50,000 atoms for
# six assets (paper, Section 5.4: "each strike takes about 2.5 minutes with
# 4 restarts" on a sixteen-core machine); R=20 for five strikes would take
# roughly 5 hours on the single process available here, exceeding the time
# available for this replication. HV already agrees with the paper's value to
# 3-4 decimal places at R=1 in a timing probe, so R is reduced to 2. Run
# `python -m exp.exp2_six_asset_basket --full` for the paper's R=20.
EXP2_RESTARTS_RUN = 2

# ---------------------------------------------------------------------------
# Experiment 3: dominant-asset closed-form bounds (Table 4)
# ---------------------------------------------------------------------------
# Row 1: X_1, X_2 ~ U[0, 30] (unweighted) and X_3 log-normal with mean 40 and
# volatility 100% over T = 1 (paper, Table 4 caption, Section 5.1) -- this row
# is an explicit exception to the "S_{i,0}=100, w_i=1/3, T=0.5" setting used
# by the other (fully log-normal) rows.
EXP3_ROW1_UNIFORM_LO = 0.0
EXP3_ROW1_UNIFORM_HI = 30.0
EXP3_ROW1_X3_MEAN = 40.0   # Table 4 caption: "X_3 is log-normal with mean 40"
EXP3_ROW1_X3_SIGMA = 1.00  # "and volatility 100% over T=1"
EXP3_ROW1_X3_T = 1.0
EXP3_ROW1_K = 70.0

EXP3_ROW2_SIGMAS = (0.10, 0.10, 1.20)
EXP3_ROW2_K = 110.0

EXP3_ROW3_SIGMAS = (0.20, 0.20, 0.60)
EXP3_ROW3_K = 105.0

EXP3_ROW4_SIGMAS = EXP1_SIGMAS  # identical to the Table 2 / Experiment 1 example
EXP3_ROW4_K = 100.0
EXP3_S0 = 100.0
EXP3_RESTARTS = 4  # Table 4 caption: "BV is the smallest basket value found by RA and BRA (... four restarts each)"

# ---------------------------------------------------------------------------
# Experiment 4: comparison with Neufeld-Xiang (2022) multi-marginal OT (Table 5)
# ---------------------------------------------------------------------------
EXP4_KNOTS_3ASSET_EQPROB_PAPER = 256   # equal-probability knots per asset, three-asset example
EXP4_KNOTS_3ASSET_EQSPACE_PAPER = 64   # equally spaced knots per asset, three-asset example
EXP4_KNOTS_6ASSET_EQPROB_PAPER = 64    # equal-probability knots per asset, six-asset example
EXP4_KNOTS_6ASSET_EQSPACE_PAPER = 32   # equally spaced knots per asset, six-asset example
# Knot counts actually solved: the full grids require a commercial LP solver
# (paper, Section 5.2, footnote: Gurobi) and did not reach the paper's
# tolerance with the open-source HiGHS backend within the time available;
# reduced grids are used (see results/RESULTS.md) while the cutting-plane
# method, the tolerance, and the knot-placement rules are unchanged.
EXP4_KNOTS_3ASSET_EQPROB_RUN = 32
EXP4_KNOTS_3ASSET_EQSPACE_RUN = 32
# 6-asset knot counts reduced further than the 3-asset case: the exact
# vertex set of src.neufeld_xiang.grid_lower_bound (full Cartesian product of
# knot-grid corners, plus hyperplane-edge slices) has size ~ m**6 + 6*m**5,
# so even m=16 (~23M candidate points) is intractable to assemble into a
# single LP within the time budget of this replication. A diagnostic sweep
# (see AGENTS.md / the Experiment 4 report) found that at m=16 the exact
# dual LP value is *provably* 0.0 for the Example-2 strikes (K >= 100, at or
# above the basket's own mean of ~100): since dropping LP constraints can
# only enlarge the feasible region of this maximisation, a bootstrapped
# sample-LP optimum of exactly 0.0 sandwiches the true exact-vertex optimum
# to 0.0 too (0.0 is always feasible, so the true optimum is also >= 0.0).
# This was confirmed both by that duality argument and by a full chunked
# scan of the exact vertex set finding zero constraint violation for the
# zero candidate. Escaping this degenerate "0.0 plateau" (as the 3-asset
# case does once m reaches ~16-24 at its K=100) would require m of the same
# order for 6 assets, which is combinatorially unreachable here. m=6 is
# therefore used instead: it keeps the exact method's constraint count
# (~34,375 at m=5, ~93,312 at m=6) solvable within seconds and, while it
# also returns 0.0, this is the largest knot count for which the exact
# method is cheap enough to re-verify quickly if a future LP/solver change
# makes a finer grid tractable.
EXP4_KNOTS_6ASSET_EQPROB_RUN = 6
EXP4_KNOTS_6ASSET_EQSPACE_RUN = 6
EXP4_EQPROB_TRUNCATION = 1e-12   # truncation quantile for equal-probability knots, Section 5.2
EXP4_EQSPACE_TRUNCATION = 1e-8   # truncation quantile for equally spaced knots, Section 5.2
EXP4_LP_TOLERANCE = 1e-6         # cutting-plane tolerance, Section 5.2
EXP4_MC_PATHS_PAPER = 10_000_000  # Monte Carlo paths per repetition for the NX upper bound, Section 5.2
EXP4_MC_PATHS_RUN = 200_000        # reduced path count for the executed run (see results/RESULTS.md)
EXP4_MC_REPEATS = 10              # repetitions of the Monte Carlo upper bound, Section 5.2
EXP4_AUGMENTED_COARSE_KNOTS = 4   # coarse equal-probability grid for the augmented-grid check, Section 5.2
EXP4_AUGMENTED_REFINED_KNOTS = 16  # refined grid, Section 5.2
EXP4_AUGMENTED_TOLERANCE = 3e-9  # expected agreement with HV^ex, Section 5.2

# ---------------------------------------------------------------------------
# Experiment 5: comparison on truncated normal mixture marginals (Table 6)
# ---------------------------------------------------------------------------
# The exact mixture parameters are generated by the Neufeld-Xiang (2022)
# script and are not given in closed form in the paper; the parameters below
# are the three-to-five-component truncated-normal mixtures generated for
# this replication by data/generate_synthetic.py with a fixed seed, matching
# the paper's description (mixtures of 3-5 normals truncated to [-10, 10]).
EXP5_TRUNCATION_LO = -10.0
EXP5_TRUNCATION_HI = 10.0
EXP5_N_MARGINALS = 3
EXP5_SEED = 12345
EXP5_DIRECTION = (0.4739, -0.8802, -0.0259)  # unit direction s, Section 5.3
EXP5_E_SX = -1.0557                          # E[<s,X>], Section 5.3
EXP5_DELTAS = (0.0, 1.0, 2.0)                 # shift offsets t = E<s,X> + delta, Section 5.3
EXP5_NX_KNOTS_PAPER = 64   # equal-probability knots per marginal, Section 5.3
EXP5_NX_KNOTS_RUN = 64
EXP5_SIX_MARGINAL_CHECK = True  # reproduce the joint-mixability check with six copies, Section 5.3

# Restart / Monte Carlo settings for Experiment 5: not specified explicitly
# for Table 6 in the paper (Section 5.3 gives the direction s, E[<s,X>] and
# deltas, but not a restart count or Monte Carlo path count); chosen
# analogously to the restart/path-count reductions used in Experiments 1-4
# for a fast default run, with a larger setting under `--full`.
EXP5_RESTARTS_RUN = 4             # run_algorithm1 restarts per delta, fast run
EXP5_RESTARTS_PAPER = 20          # run_algorithm1 restarts per delta, --full run
EXP5_MC_PATHS_RUN = 20_000        # grid_reassembly_bound Monte Carlo paths, fast run
EXP5_MC_PATHS_PAPER = 10_000_000  # grid_reassembly_bound Monte Carlo paths, --full run (matches EXP4_MC_PATHS_PAPER)
EXP5_MC_REPEATS_RUN = 5           # grid_reassembly_bound Monte Carlo repeats, fast run
EXP5_MC_REPEATS_PAPER = 10        # grid_reassembly_bound Monte Carlo repeats, --full run (matches EXP4_MC_REPEATS)
EXP5_JOINT_RESTARTS_RUN = 4       # run_algorithm1 restarts, six-copy joint-mixability check, fast run
EXP5_JOINT_RESTARTS_PAPER = 20    # run_algorithm1 restarts, six-copy joint-mixability check, --full run
EXP5_JOINT_SEED_OFFSET = 1        # seed offset added to EXP5_SEED for the six-copy generator, Section 5.3
