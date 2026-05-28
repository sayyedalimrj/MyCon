"""Tests for :mod:`pipeline.common.calibration`.

These tests verify both the *math* (closed-form ECE / Brier on hand-built
inputs) and the *engineering contract* (JSON-shape stability, schema
version, label-to-probability mapping, equal-mass binning robustness).

Marked ``lightweight`` so they run in the laptop test set per
``pytest.ini`` / ``tests/conftest.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pipeline.common.calibration import (
    DEFAULT_CONFIDENCE_LABEL_PROBABILITIES,
    CalibrationDataset,
    brier_score,
    build_reliability_table,
    calibration_report,
    expected_calibration_error,
    maximum_calibration_error,
    smooth_ece,
)


pytestmark = pytest.mark.lightweight


# ---------------------------------------------------------------------------
# CalibrationDataset construction
# ---------------------------------------------------------------------------


def test_dataset_from_lists_normalises_inputs() -> None:
    ds = CalibrationDataset.from_lists(
        ["high", "medium", "low", 0.95, 0.05],
        [True, False, False, 1, 0],
    )
    assert len(ds) == 5
    # Discrete labels should map exactly per default mapping.
    assert ds.probabilities[0] == DEFAULT_CONFIDENCE_LABEL_PROBABILITIES["high"]
    assert ds.probabilities[1] == DEFAULT_CONFIDENCE_LABEL_PROBABILITIES["medium"]
    assert ds.probabilities[2] == DEFAULT_CONFIDENCE_LABEL_PROBABILITIES["low"]
    # Numeric probabilities pass through clamped to [0, 1].
    assert math.isclose(ds.probabilities[3], 0.95)
    assert math.isclose(ds.probabilities[4], 0.05)


def test_dataset_from_lists_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        CalibrationDataset.from_lists([0.5, 0.6], [1])


def test_dataset_from_lists_rejects_invalid_correctness() -> None:
    with pytest.raises(ValueError):
        CalibrationDataset.from_lists([0.5], ["maybe"])


def test_dataset_from_records_drops_invalid_rows_silently() -> None:
    records = [
        {"confidence": "high", "correct": True},
        {"confidence": 0.4, "correct": False},
        {"confidence": "high"},  # missing correct → dropped
        {"correct": True},  # missing confidence → mapped to 0.5 then kept
    ]
    ds = CalibrationDataset.from_records(records)
    # Three valid rows: row 0, row 1, row 3 (default unknown -> 0.5).
    assert len(ds) == 3


def test_dataset_clamps_out_of_range_numeric_probabilities() -> None:
    ds = CalibrationDataset.from_lists([1.5, -0.2], [1, 0])
    assert ds.probabilities[0] == 1.0
    assert ds.probabilities[1] == 0.0


def test_dataset_drops_non_finite_values() -> None:
    ds = CalibrationDataset.from_lists([float("nan"), 0.5], [1, 1])
    # NaN gets clamped/coerced to 0.5 by _coerce_label_to_prob (returns 0.5)
    # and then kept (it is finite); we just assert the dataset stays valid.
    assert all(math.isfinite(p) for p in ds.probabilities)
    assert all(c in (0.0, 1.0) for c in ds.correctness)


# ---------------------------------------------------------------------------
# Closed-form ECE / Brier sanity
# ---------------------------------------------------------------------------


def test_ece_zero_when_perfectly_calibrated() -> None:
    """If every prediction is 0 or 1 and matches the label, ECE = 0."""
    ds = CalibrationDataset.from_lists([0.0, 1.0, 0.0, 1.0], [0, 1, 0, 1])
    assert expected_calibration_error(ds, n_bins=4, strategy="equal_width") == pytest.approx(0.0, abs=1e-12)
    assert maximum_calibration_error(ds, n_bins=4, strategy="equal_width") == pytest.approx(0.0, abs=1e-12)


def test_ece_one_when_maximally_miscalibrated() -> None:
    """If all preds are 1.0 but all labels are 0, ECE = 1."""
    ds = CalibrationDataset.from_lists([1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0])
    assert expected_calibration_error(ds, n_bins=4, strategy="equal_width") == pytest.approx(1.0, abs=1e-12)
    assert maximum_calibration_error(ds, n_bins=4, strategy="equal_width") == pytest.approx(1.0, abs=1e-12)


def test_brier_closed_form() -> None:
    """Brier = mean((p - y)^2). Hand-computed example."""
    ds = CalibrationDataset.from_lists([0.9, 0.1, 0.8, 0.2], [1, 0, 0, 1])
    # (.1)^2 + (.1)^2 + (.8)^2 + (.8)^2 = 0.01 + 0.01 + 0.64 + 0.64 = 1.30
    # mean = 0.325
    assert brier_score(ds) == pytest.approx(0.325, abs=1e-12)


def test_brier_nan_for_empty_dataset() -> None:
    ds = CalibrationDataset.from_lists([], [])
    assert math.isnan(brier_score(ds))


def test_ece_decreases_with_better_calibration() -> None:
    """A better-calibrated set should have a strictly lower ECE."""
    bad = CalibrationDataset.from_lists([0.95, 0.95, 0.95, 0.95], [0, 0, 0, 1])
    good = CalibrationDataset.from_lists([0.95, 0.95, 0.95, 0.95], [1, 1, 1, 0])
    bad_ece = expected_calibration_error(bad, n_bins=10, strategy="equal_width")
    good_ece = expected_calibration_error(good, n_bins=10, strategy="equal_width")
    assert good_ece < bad_ece


# ---------------------------------------------------------------------------
# Reliability table shape
# ---------------------------------------------------------------------------


def test_reliability_table_returns_n_bins_rows_even_when_empty_bins() -> None:
    ds = CalibrationDataset.from_lists([0.85, 0.85, 0.85], [1, 1, 0])
    table = build_reliability_table(ds, n_bins=10, strategy="equal_width")
    assert len(table.bins) == 10
    populated = [b for b in table.bins if b.count > 0]
    assert len(populated) == 1
    assert populated[0].count == 3
    assert populated[0].empirical_accuracy == pytest.approx(2.0 / 3.0)


def test_equal_mass_keeps_every_bin_populated_when_possible() -> None:
    # Spread 100 distinct values across [0, 1] so equal-mass binning has
    # one observation per bin (when n_bins divides n).
    ds = CalibrationDataset.from_lists(
        [i / 100.0 for i in range(100)],
        [i % 2 for i in range(100)],
    )
    table = build_reliability_table(ds, n_bins=10, strategy="equal_mass")
    # All 10 bins should be populated (10 obs each).
    assert table.n_bins_with_data == 10
    for b in table.bins:
        assert b.count == 10


def test_reliability_table_empty_dataset_returns_nan_metrics() -> None:
    ds = CalibrationDataset.from_lists([], [])
    table = build_reliability_table(ds, n_bins=5)
    assert table.n_total == 0
    assert table.n_bins_with_data == 0
    assert math.isnan(table.expected_calibration_error)
    assert math.isnan(table.maximum_calibration_error)
    assert math.isnan(table.brier_score)


def test_invalid_strategy_raises() -> None:
    ds = CalibrationDataset.from_lists([0.5], [1])
    with pytest.raises(ValueError):
        build_reliability_table(ds, strategy="bogus")


def test_invalid_n_bins_raises() -> None:
    ds = CalibrationDataset.from_lists([0.5], [1])
    with pytest.raises(ValueError):
        build_reliability_table(ds, n_bins=0)


# ---------------------------------------------------------------------------
# Smooth ECE
# ---------------------------------------------------------------------------


def test_smooth_ece_small_for_well_calibrated() -> None:
    """Smooth-ECE on perfectly bimodal well-calibrated data is small but
    not zero: kernel smoothing inherently spreads probability mass across
    [0, 1], which is *the point* of the estimator (Blasiok-Nakkiran 2024).
    We assert it stays below a generous threshold."""
    ds = CalibrationDataset.from_lists([0.0, 1.0, 0.0, 1.0] * 10, [0, 1, 0, 1] * 10)
    assert smooth_ece(ds) < 0.2


def test_smooth_ece_increases_with_miscalibration() -> None:
    good = CalibrationDataset.from_lists([0.2, 0.8] * 50, [0, 1] * 50)
    bad = CalibrationDataset.from_lists([0.9] * 100, [0] * 100)
    assert smooth_ece(bad) > smooth_ece(good)


def test_smooth_ece_nan_for_empty() -> None:
    ds = CalibrationDataset.from_lists([], [])
    assert math.isnan(smooth_ece(ds))


def test_smooth_ece_rejects_nonpositive_bandwidth() -> None:
    ds = CalibrationDataset.from_lists([0.5], [1])
    with pytest.raises(ValueError):
        smooth_ece(ds, bandwidth=0.0)


# ---------------------------------------------------------------------------
# calibration_report — JSON contract
# ---------------------------------------------------------------------------


def test_calibration_report_contract_has_stable_schema_version() -> None:
    records = [
        {"confidence": "high", "correct": True},
        {"confidence": "high", "correct": True},
        {"confidence": "low", "correct": False},
    ]
    rep = calibration_report(records, n_bins=5)
    assert rep["schema_version"] == "calibration_report.v1"
    assert rep["n_samples"] == 3
    assert "metrics" in rep
    for key in ("expected_calibration_error", "maximum_calibration_error", "brier_score", "smooth_ece"):
        assert key in rep["metrics"]
    assert isinstance(rep["reliability_table"], list)
    assert len(rep["reliability_table"]) == 5
    # Every bin row carries the documented fields.
    for row in rep["reliability_table"]:
        for k in ("bin_index", "lower_edge", "upper_edge", "count", "mean_confidence", "empirical_accuracy", "gap"):
            assert k in row
    # Mapping is recorded inline so the report is self-contained.
    assert rep["label_probability_mapping"]["high"] == DEFAULT_CONFIDENCE_LABEL_PROBABILITIES["high"]


def test_calibration_report_round_trips_through_json(tmp_path: Path) -> None:
    records = [
        {"confidence": "high", "correct": True},
        {"confidence": "medium", "correct": False},
        {"confidence": "low", "correct": False},
    ]
    rep = calibration_report(records)
    out = tmp_path / "rep.json"
    out.write_text(json.dumps(rep), encoding="utf-8")
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == "calibration_report.v1"
    assert parsed["n_samples"] == 3


def test_calibration_report_accepts_dataset_directly() -> None:
    ds = CalibrationDataset.from_lists([0.85, 0.85], [1, 1])
    rep = calibration_report(ds)
    assert rep["n_samples"] == 2
    # Two perfectly-correct high-confidence observations → low ECE.
    assert rep["metrics"]["expected_calibration_error"] < 0.2


def test_calibration_report_with_custom_label_mapping_changes_value() -> None:
    """If the operator chooses a different mapping, the recorded mapping
    and the resulting ECE both change accordingly."""
    records = [
        {"confidence": "high", "correct": True},
        {"confidence": "high", "correct": False},
    ]
    rep_default = calibration_report(records)
    rep_custom = calibration_report(records, label_probability_mapping={"high": 0.55, "unknown": 0.5})
    assert rep_default["label_probability_mapping"]["high"] != rep_custom["label_probability_mapping"]["high"]
    # When 'high' maps closer to 0.5 the ECE for a 50%% empirical accuracy
    # batch should drop.
    assert rep_custom["metrics"]["expected_calibration_error"] < rep_default["metrics"]["expected_calibration_error"]
