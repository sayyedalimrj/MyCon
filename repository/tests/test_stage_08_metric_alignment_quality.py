from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from pipeline.stage_08_bim_eval.metric_alignment_quality import (
    enrich_metric_alignment_report_file,
    estimate_similarity_umeyama,
    evaluate_anchor_geometry,
    leave_one_out_residuals,
    robust_similarity_ransac,
)


def _transform(points: np.ndarray, scale: float, translation: np.ndarray) -> np.ndarray:
    return scale * points + translation


def test_similarity_umeyama_recovers_uniform_scale_and_translation() -> None:
    scan = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    bim = _transform(scan, 2.0, np.array([10.0, -3.0, 1.5]))

    fit = estimate_similarity_umeyama(scan, bim)

    assert fit.status == "ok"
    assert math.isclose(fit.scale, 2.0, rel_tol=1e-6)
    assert fit.rmse_m < 1e-8
    assert fit.max_residual_m < 1e-8


def test_anchor_geometry_rejects_collinear_anchors() -> None:
    scan = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    result = evaluate_anchor_geometry(scan)

    assert result["passed"] is False
    assert any("degenerate_anchor_geometry_rank" in failure for failure in result["failures"])


def test_leave_one_out_reports_small_residuals_for_clean_anchors() -> None:
    scan = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    bim = _transform(scan, 1.5, np.array([2.0, 3.0, -1.0]))

    result = leave_one_out_residuals(scan, bim)

    assert result["status"] == "ok"
    assert result["max_residual_m"] < 1e-8


def test_robust_similarity_ransac_rejects_outlier() -> None:
    scan = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=float,
    )
    bim = _transform(scan, 2.0, np.array([1.0, 2.0, 3.0]))
    bim[-1] = np.array([100.0, 100.0, 100.0])

    fit = robust_similarity_ransac(scan, bim, residual_threshold_m=0.05)

    assert fit.status == "ok"
    assert len(fit.inlier_indices) == 4
    assert 4 not in fit.inlier_indices
    assert any("ransac_rejected_outliers" in warning for warning in fit.warnings)


def test_report_enrichment_marks_insufficient_anchors_failed(tmp_path: Path) -> None:
    report = tmp_path / "metric_alignment_report.json"
    report.write_text(
        json.dumps(
            {
                "status": "skipped_insufficient_anchors",
                "confidence": "low",
                "usable_registration_anchor_count": 1,
                "quality_gate": {"passed": True, "failures": [], "warnings": [], "thresholds": {}},
            }
        ),
        encoding="utf-8",
    )

    enriched = enrich_metric_alignment_report_file(report)

    assert enriched["quality_gate"]["passed"] is False
    assert enriched["confidence"] == "low"
    assert any("insufficient_registration_anchors" in failure for failure in enriched["quality_gate"]["failures"])
