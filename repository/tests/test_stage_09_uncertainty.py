"""Tests for ``pipeline.stage_09_progress.uncertainty``.

We pin three contracts:

1. :func:`wilson_interval` matches a textbook value for a known
   ``(successes, n)`` pair within a small tolerance.
2. :func:`f_score` matches the textbook harmonic-mean definition.
3. :func:`bootstrap_ci` produces an interval that brackets the true mean of a
   normal distribution with high probability, and is reproducible across runs
   given the same RNG.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.stage_09_progress.uncertainty import bootstrap_ci, f_score, wilson_interval


def test_wilson_interval_textbook_value() -> None:
    # 8/10 successes, 95 % CI: ≈ (0.490, 0.943) per Wilson's formula.
    lo, p, hi = wilson_interval(8, 10)
    assert p == pytest.approx(0.8)
    assert lo == pytest.approx(0.490, abs=5e-3)
    assert hi == pytest.approx(0.943, abs=5e-3)


def test_wilson_interval_n_zero_returns_zero() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0, 0.0)


def test_wilson_interval_clamps_to_unit_interval() -> None:
    lo_z, p_z, hi_z = wilson_interval(0, 5)
    lo_o, p_o, hi_o = wilson_interval(5, 5)
    assert 0.0 <= lo_z <= p_z <= hi_z <= 1.0
    assert 0.0 <= lo_o <= p_o <= hi_o <= 1.0


def test_wilson_interval_clamps_overshoot_successes() -> None:
    # successes > n must not produce a value outside [0, 1].
    lo, p, hi = wilson_interval(20, 10)
    assert 0.0 <= lo <= p <= hi <= 1.0
    # and it must be treated as p = 1.
    assert p == 1.0


def test_f_score_definition_matches_harmonic_mean() -> None:
    # 2 * 0.8 * 0.5 / (0.8 + 0.5) = 0.6153846...
    assert f_score(0.8, 0.5) == pytest.approx(2 * 0.8 * 0.5 / (0.8 + 0.5))


def test_f_score_is_zero_when_either_component_is_zero() -> None:
    assert f_score(0.0, 0.9) == 0.0
    assert f_score(0.9, 0.0) == 0.0


def test_f_score_handles_non_finite() -> None:
    assert f_score(float("nan"), 0.5) == 0.0
    assert f_score(0.5, float("inf")) == 0.0


def test_bootstrap_ci_brackets_true_mean() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=1.0, scale=0.1, size=200)
    lo, pt, hi = bootstrap_ci(
        data,
        statistic=lambda a: float(np.mean(a)),
        n_boot=500,
        rng=np.random.default_rng(7),
    )
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo < pt < hi
    # 95 % CI should bracket 1.0 with overwhelming probability for
    # N(1, 0.1) and 200 samples.
    assert lo <= 1.0 <= hi


def test_bootstrap_ci_is_reproducible_under_seeded_rng() -> None:
    data = np.linspace(0, 1, 50)
    lo_a, pt_a, hi_a = bootstrap_ci(
        data, statistic=lambda a: float(np.mean(a)), n_boot=200, rng=np.random.default_rng(123)
    )
    lo_b, pt_b, hi_b = bootstrap_ci(
        data, statistic=lambda a: float(np.mean(a)), n_boot=200, rng=np.random.default_rng(123)
    )
    assert (lo_a, pt_a, hi_a) == (lo_b, pt_b, hi_b)


def test_bootstrap_ci_too_few_samples_returns_nan_triple() -> None:
    lo, pt, hi = bootstrap_ci(
        np.asarray([1.0]), statistic=lambda a: float(np.mean(a)), n_boot=200, rng=np.random.default_rng(1)
    )
    assert math.isnan(lo) and math.isnan(pt) and math.isnan(hi)


def test_bootstrap_ci_filters_non_finite_input() -> None:
    data = np.asarray([1.0, 2.0, float("nan"), float("inf"), 3.0, 4.0])
    lo, pt, hi = bootstrap_ci(
        data, statistic=lambda a: float(np.mean(a)), n_boot=200, rng=np.random.default_rng(1)
    )
    assert math.isfinite(lo) and math.isfinite(hi)
    # mean of [1,2,3,4] is 2.5
    assert lo <= 2.5 <= hi
