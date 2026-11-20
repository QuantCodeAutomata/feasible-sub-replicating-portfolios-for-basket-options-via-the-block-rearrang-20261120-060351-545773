"""Synthetic marginals used by Experiment 5 (Table 6).

Neufeld and Xiang (2022) generate, in their third numerical experiment,
marginals that are mixtures of three to five normal distributions truncated
to ``[-10, 10]``. The paper reproduced here cites their generator
(https://github.com/qikunxiang/MultiMarginalOptimalTransport) but does not
give the mixture parameters in closed form, and that repository could not be
retrieved in this environment. This module generates representative mixtures
with the same structure (3-5 truncated-normal components on ``[-10, 10]``,
mixture weights summing to one) with a fixed seed, so that the comparison of
Experiment 5 runs on marginals of the kind described in Section 5.3 rather
than on invented numbers presented as the authors'. This substitution is
recorded in ``results/RESULTS.md``.
"""

from __future__ import annotations

import numpy as np

from src.marginals import TruncatedNormalMixtureMarginal

LO, HI = -10.0, 10.0


def generate_mixture_marginal(rng: np.random.Generator) -> TruncatedNormalMixtureMarginal:
    """One mixture of 3-5 normals truncated to ``[LO, HI]`` (Neufeld and Xiang
    (2022), third experiment structure)."""
    n_comp = int(rng.integers(3, 6))
    weights = rng.dirichlet(np.ones(n_comp))
    means = rng.uniform(LO + 2.0, HI - 2.0, size=n_comp)
    stds = rng.uniform(1.0, 4.0, size=n_comp)
    return TruncatedNormalMixtureMarginal(weights=weights, means=means, stds=stds, lo=LO, hi=HI)


def generate_marginals(n_marginals: int, seed: int) -> list[TruncatedNormalMixtureMarginal]:
    """``n_marginals`` independent truncated-normal-mixture marginals with a
    fixed seed, for reproducible use across runs of Experiment 5."""
    rng = np.random.default_rng(seed)
    return [generate_mixture_marginal(rng) for _ in range(n_marginals)]
