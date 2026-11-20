"""Tests for Experiment 5 (Table 6): truncated-mixture marginals, the
AffineMarginal non-negativity construction, and the cost-transform identity
used in ``exp/exp5_truncated_mixture_comparison.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.generate_synthetic import generate_marginals
from src.marginals import AffineMarginal


@pytest.mark.parametrize("s", [0.4739, -0.8802, -0.0259, 1.0, -1.0])
def test_affine_marginal_nonnegative(s: float) -> None:
    """``y = s*x + 10*|s|`` is non-negative for ``x`` truncated to [-10, 10],
    for both positive and negative ``s`` (Section 5.3 construction)."""
    (x,) = generate_marginals(1, seed=7)
    y = AffineMarginal(base=x, a=s, b=10.0 * abs(s))
    sample = y.discretise(2000)
    assert (sample >= -1e-9).all()


def test_cost_transform_identity() -> None:
    """Numerically confirm the identity underlying the cost-scale conversion
    used in ``run_delta_row``:

        |z - t| = 2*(z - t)^+ - (z - t)                                (i)

    for ``z = sum_i s_i x_i``, and that the bridge through ``y_i = s_i*x_i +
    10*|s_i|``, ``kb = t + 10*sum_i|s_i|`` holds *pathwise*:

        sum_i y_i - kb == z - t                                        (ii)

    so that in particular ``(sum_i y_i - kb)^+ = (z - t)^+``, which is what
    justifies evaluating the basket-call bracket on the (non-negative) ``y``
    marginals instead of directly on ``z - t``. We deliberately do *not* test
    ``|z - t| == 2*(z - kb)^+ - (z - kb)``, which is false in general since
    ``kb != t`` (``kb = t + 10*sum|s_i|``); the correct bridge goes through
    ``sum_i y_i``, not through ``z`` itself.
    """
    marginals = generate_marginals(3, seed=7)
    s = (0.4739, -0.8802, -0.0259)
    t = -1.0557 + 1.0  # arbitrary delta, matches config.EXP5_DELTAS[1]
    kb = t + 10.0 * sum(abs(si) for si in s)

    n_atoms = 20_000
    x_samples = [m.discretise(n_atoms) for m in marginals]  # independent per-marginal quantile grids
    rng = np.random.default_rng(0)
    # Independent coupling is sufficient here: both sides of (i) and (ii) are
    # pointwise identities, so they hold under any coupling, including a
    # random shuffle of each marginal's discretised atoms.
    for col in x_samples:
        rng.shuffle(col)
    z = sum(si * col for si, col in zip(s, x_samples))

    y_sum = sum(
        si * col + 10.0 * abs(si) for si, col in zip(s, x_samples)
    )

    # (ii): pathwise identity sum_i y_i - kb == z - t
    np.testing.assert_allclose(y_sum - kb, z - t, atol=1e-8)

    # (i): general identity |z-t| = 2*(z-t)^+ - (z-t), exact pathwise
    lhs = np.abs(z - t)
    rhs = 2.0 * np.clip(z - t, 0.0, None) - (z - t)
    np.testing.assert_allclose(lhs, rhs, atol=1e-8)

    # Combined, on expectation scale (what the script actually computes):
    # E[|z-t|] == 2*E[(sum_y - kb)^+] - (E[sum_y] - kb)
    mean_abs = float(np.mean(np.abs(z - t)))
    cost = 2.0 * float(np.mean(np.clip(y_sum - kb, 0.0, None))) - (float(np.mean(y_sum)) - kb)
    assert abs(mean_abs - cost) < 1e-6


def test_generate_marginals_smoke() -> None:
    marginals = generate_marginals(3, seed=42)
    assert len(marginals) == 3
    for m in marginals:
        assert np.isfinite(m.mean)
        sample = m.discretise(500)
        assert (np.diff(sample) >= 0.0).all()
