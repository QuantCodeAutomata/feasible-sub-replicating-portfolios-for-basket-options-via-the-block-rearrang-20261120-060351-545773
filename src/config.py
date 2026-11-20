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
EXP1_RESTARTS_RUN = 64                # restarts actually run (full paper value)

# ---------------------------------------------------------------------------
# Experiment 2: six-asset basket call (Table 3 / Example 2, Table 1)
# ---------------------------------------------------------------------------
EXP2_SIGMAS = (0.915, 0.755, 0.825, 0.355, 0.20, 0.15)  # Table 1 Example 2
EXP2_S0 = 100.0
EXP2_WEIGHTS = tuple(1 / 6 for _ in range(6))
EXP2_STRIKES = (100.0, 110.0, 120.0, 130.0, 140.0)
EXP2_RESTARTS_PAPER = 20              # R restarts used in the tables, Section 5
EXP2_RESTARTS_RUN = 8                 # reduced restart count for the executed run (see results/RESULTS.md)

# ---------------------------------------------------------------------------
# Experiment 3: dominant-asset closed-form bounds (Table 4)
# ---------------------------------------------------------------------------
# Row 1: two Uniform[0,30] marginals and one log-normal marginal; the paper
# does not give the log-normal volatility explicitly for this row, so the
# Example-1 value for the third (dominant) asset is reused (not specified in
# the paper; documented as an assumption in results/RESULTS.md).
EXP3_ROW1_UNIFORM_LO = 0.0
EXP3_ROW1_UNIFORM_HI = 30.0
EXP3_ROW1_SIGMA3 = 1.20
EXP3_ROW1_K = 70.0

EXP3_ROW2_SIGMAS = (0.10, 0.10, 1.20)
EXP3_ROW2_K = 110.0

EXP3_ROW3_SIGMAS = (0.20, 0.20, 0.60)
EXP3_ROW3_K = 105.0

EXP3_ROW4_SIGMAS = EXP1_SIGMAS  # identical to the Table 2 / Experiment 1 example
EXP3_ROW4_K = 100.0
EXP3_S0 = 100.0
EXP3_RESTARTS = 32  # restarts for the BV cross-check via BRA/RA on Table 4 rows (not specified in the paper)

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
EXP4_KNOTS_6ASSET_EQPROB_RUN = 16
EXP4_KNOTS_6ASSET_EQSPACE_RUN = 16
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
