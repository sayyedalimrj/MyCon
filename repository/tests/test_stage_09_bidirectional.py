"""Tests for ``pipeline.stage_09_progress.bidirectional_metrics``.

These tests exercise the four cases that matter for scan-to-BIM progress:

- Perfect overlap → accuracy = completeness = F-score = 1.
- Half overlap   → accuracy = 1, completeness = 0.5 (the construction
  semantics: the scan agrees with the BIM where it covers, but only covers
  half of the surface).
- Disjoint clouds → accuracy = completeness = 0.
- Empty side → no zero-division, structured ``notes`` returned.
"""

from __future__ import annotations

import math

import numpy as np

from pipeline.stage_09_progress.bidirectional_metrics import (
    BidirectionalResult,
    compute_bidirectional,
)


def test_perfect_overlap_yields_unit_metrics() -> None:
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    r = compute_bidirectional(scan_points=pts, bim_points=pts, tau_m=0.05)
    assert r.accuracy == 1.0
    assert r.completeness == 1.0
    assert r.f_score == 1.0
    assert r.accuracy_count_evaluated == 4
    assert r.completeness_count_evaluated == 4


def test_half_coverage_distinguishes_accuracy_from_completeness() -> None:
    scan = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    bim = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    r = compute_bidirectional(scan_points=scan, bim_points=bim, tau_m=0.05)
    # Every scan point hits a BIM point exactly → accuracy = 1.
    assert r.accuracy == 1.0
    # Only half of BIM points are within tau of any scan point → completeness = 0.5.
    assert r.completeness == 0.5
    # Harmonic mean of (1, 0.5) = 2/3.
    assert math.isclose(r.f_score, 2.0 / 3.0, rel_tol=1e-9)


def test_disjoint_clouds_yield_zero_metrics() -> None:
    scan = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    bim = np.array([[100, 0, 0], [101, 0, 0]], dtype=float)
    r = compute_bidirectional(scan_points=scan, bim_points=bim, tau_m=0.05)
    assert r.accuracy == 0.0
    assert r.completeness == 0.0
    assert r.f_score == 0.0


def test_empty_scan_returns_zero_with_note() -> None:
    bim = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    r = compute_bidirectional(scan_points=np.zeros((0, 3)), bim_points=bim, tau_m=0.05)
    assert r.accuracy == 0.0
    assert r.completeness == 0.0
    assert "no_scan_points_for_element" in r.notes


def test_empty_bim_returns_zero_with_note() -> None:
    scan = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    r = compute_bidirectional(scan_points=scan, bim_points=np.zeros((0, 3)), tau_m=0.05)
    assert r.accuracy == 0.0
    assert r.completeness == 0.0
    assert "no_bim_points_for_element" in r.notes


def test_to_row_serializes_all_documented_keys() -> None:
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    r = compute_bidirectional(scan_points=pts, bim_points=pts, tau_m=0.05)
    row = r.to_row()
    expected_keys = {
        "bidirectional_tau_m",
        "accuracy",
        "completeness",
        "f_score",
        "accuracy_n_evaluated",
        "accuracy_n_in_tolerance",
        "completeness_n_evaluated",
        "completeness_n_in_tolerance",
        "accuracy_wilson_lo",
        "accuracy_wilson_hi",
        "completeness_wilson_lo",
        "completeness_wilson_hi",
        "f_score_wilson_lo",
        "f_score_wilson_hi",
        "bidirectional_notes",
    }
    assert set(row.keys()) == expected_keys


def test_wilson_endpoints_form_valid_interval() -> None:
    scan = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    bim = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    r = compute_bidirectional(scan_points=scan, bim_points=bim, tau_m=0.05)
    ac_lo, _, ac_hi = r.accuracy_wilson
    cm_lo, _, cm_hi = r.completeness_wilson
    f_lo, f_pt, f_hi = r.f_score_wilson
    assert 0.0 <= ac_lo <= r.accuracy <= ac_hi <= 1.0
    assert 0.0 <= cm_lo <= r.completeness <= cm_hi <= 1.0
    # F-score CI envelope must contain the point estimate.
    assert f_lo <= f_pt <= f_hi


def test_result_is_immutable_dataclass() -> None:
    pts = np.array([[0, 0, 0]], dtype=float)
    r = compute_bidirectional(scan_points=pts, bim_points=pts, tau_m=0.05)
    assert isinstance(r, BidirectionalResult)
    try:
        r.accuracy = 0.5  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("BidirectionalResult should be frozen")
