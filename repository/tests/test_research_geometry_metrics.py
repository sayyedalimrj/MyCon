"""Deterministic synthetic/mocked tests; NOT real building benchmarks."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from pipeline.stage_08_bim_eval.research_metrics import (
    alignment_residuals, compare_clouds, gravity_error_deg,
)


def test_identical_clouds_have_perfect_metrics() -> None:
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    result = compare_clouds(points, points, threshold_m=0.01)
    assert result["f_score_at_threshold"] == 1.0
    assert result["accuracy_mean_m"] == pytest.approx(0)
    assert result["completeness_mean_m"] == pytest.approx(0)
    assert result["chamfer_l1_sum_m"] == pytest.approx(0)
    assert result["sampled_scan_points"] == 4


def test_shift_yields_nonzero_distance_and_no_false_completeness() -> None:
    src = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=float)
    dst = src + [0, 0, 0.2]
    r = compare_clouds(src, dst, threshold_m=0.05, max_points=2)
    assert r["accuracy_mean_m"] == pytest.approx(0.2)
    assert r["completeness_mean_m"] == pytest.approx(0.2)
    assert r["precision_at_threshold"] == 0
    assert r["f_score_at_threshold"] == 0
    assert r["chamfer_l1_sum_m"] == pytest.approx(0.4)
    assert r["chamfer_l2_squared_sum_m2"] == pytest.approx(0.08)


def test_reproducible_sampling_and_reject_invalid_inputs() -> None:
    rng = np.random.default_rng(1)
    points = rng.normal(size=(100, 3))
    a = compare_clouds(points, points, max_points=20, random_seed=11)
    b = compare_clouds(points, points, max_points=20, random_seed=11)
    assert a == b
    with pytest.raises(ValueError):
        compare_clouds(points, points, threshold_m=0)
    with pytest.raises(ValueError):
        compare_clouds([[np.nan, 0, 0]], points)
    with pytest.raises(ValueError):
        compare_clouds([], points)


def test_external_anchor_error_and_up_vector_angle() -> None:
    scan = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    bim = [[0, 0, 0.1], [1, 0, 0.1], [0, 1, 0.1]]
    r = alignment_residuals(scan, bim)
    assert r["count"] == 3
    assert r["registration_error_rmse_m"] == pytest.approx(.1)
    assert gravity_error_deg([0, 0, 1], [0, 0, 1]) == pytest.approx(0)
    assert gravity_error_deg([0, 0, 1], [1, 0, 0]) == pytest.approx(90)
    with pytest.raises(ValueError):
        gravity_error_deg([0, 0, 0], [0, 0, 1])


pytestmark = pytest.mark.geometry
