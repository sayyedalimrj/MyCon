"""Tests for :mod:`pipeline.service.calibration_api`.

Exercise the pure read/write functions directly. The router is a thin
HTTP wrapper around the same functions; covering the functions covers
the contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.common.hitl import CorrectionStore
from pipeline.service.calibration_api import (
    CALIBRATION_REPORT_BASENAME,
    CalibrationApiError,
    CalibrationArtefactPaths,
    DEFAULT_TARGET_KINDS,
    get_latest_report,
    run_calibration_report,
)


pytestmark = pytest.mark.lightweight


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_corrections(jsonl_path: Path, *, n_correct: int = 5, n_wrong: int = 1) -> int:
    """Append a deterministic mix of correct + incorrect corrections.

    All records use predicted_confidence='high' and predicted_value='accept';
    overruled rows set corrected_value='reject', confirmed rows set it
    to 'accept'. Returns the total number of records written.
    """
    store = CorrectionStore(jsonl_path)
    n_total = 0
    for i in range(n_correct):
        store.append(
            {
                "target_kind": "element_acceptance",
                "target_id": f"correctElem{i:02d}aaaaaaaaaa",
                "predicted_value": "accept",
                "predicted_confidence": "high",
                "corrected_value": "accept",
                "reviewer_id": "alice",
                "timestamp_utc": f"2026-05-01T0{i}:00:00Z",
                "rationale": "ok",
                "run_id": "test_run",
            }
        )
        n_total += 1
    for i in range(n_wrong):
        store.append(
            {
                "target_kind": "element_acceptance",
                "target_id": f"wrongElem{i:02d}bbbbbbbbbbbb",
                "predicted_value": "accept",
                "predicted_confidence": "high",
                "corrected_value": "reject",
                "reviewer_id": "bob",
                "timestamp_utc": f"2026-05-02T0{i}:00:00Z",
                "rationale": "missing rebar capping",
                "run_id": "test_run",
            }
        )
        n_total += 1
    return n_total


def _paths(tmp_path: Path) -> CalibrationArtefactPaths:
    return CalibrationArtefactPaths.under_run_dir(tmp_path)


# ---------------------------------------------------------------------------
# Path layout + constants
# ---------------------------------------------------------------------------


def test_basename_constants_are_locked() -> None:
    assert CALIBRATION_REPORT_BASENAME == "calibration_report.json"
    assert DEFAULT_TARGET_KINDS == ("element_acceptance",)


def test_under_run_dir_uses_canonical_layout(tmp_path: Path) -> None:
    paths = CalibrationArtefactPaths.under_run_dir(tmp_path)
    assert paths.corrections_jsonl == (tmp_path / "reports" / "hitl_corrections.jsonl").resolve()
    assert paths.report_json == (tmp_path / "reports" / "calibration_report.json").resolve()


# ---------------------------------------------------------------------------
# run_calibration_report — happy paths
# ---------------------------------------------------------------------------


def test_run_writes_report_json_and_returns_metrics(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.corrections_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_total = _seed_corrections(paths.corrections_jsonl, n_correct=5, n_wrong=1)
    response = run_calibration_report(paths)
    assert response["schema_version"] == "calibration_run_response.v1"
    assert response["n_replayed_records"] == n_total
    assert response["n_effective_records"] == n_total
    assert response["n_conflicts"] == 0
    # Report shape is the standard calibration_report.v1
    report = response["report"]
    assert report["schema_version"] == "calibration_report.v1"
    assert report["n_samples"] == n_total
    assert 0.0 <= report["metrics"]["expected_calibration_error"] <= 1.0
    # Persisted on disk
    assert paths.report_json.exists()
    on_disk = json.loads(paths.report_json.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "calibration_report.v1"


def test_run_attaches_provenance_block(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.corrections_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _seed_corrections(paths.corrections_jsonl)
    response = run_calibration_report(paths, n_bins=5, strategy="equal_mass")
    prov = response["report"]["calibration_run_provenance"]
    assert prov["schema_version"] == "calibration_run_provenance.v1"
    assert prov["n_replayed_records"] == 6
    assert prov["target_kinds"] == ["element_acceptance"]
    assert prov["filter_run_id"] is None
    assert prov["corrections_jsonl"] == str(paths.corrections_jsonl.resolve())


def test_run_is_idempotent(tmp_path: Path) -> None:
    """Replaying the same log twice produces the same metrics."""
    paths = _paths(tmp_path)
    paths.corrections_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _seed_corrections(paths.corrections_jsonl, n_correct=4, n_wrong=2)
    a = run_calibration_report(paths)["report"]["metrics"]
    b = run_calibration_report(paths)["report"]["metrics"]
    assert a == b


def test_run_filters_by_target_kinds(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.corrections_jsonl.parent.mkdir(parents=True, exist_ok=True)
    store = CorrectionStore(paths.corrections_jsonl)
    # Three element-acceptance corrections + two activity-completion corrections.
    for i in range(3):
        store.append(
            {
                "target_kind": "element_acceptance",
                "target_id": f"elemA{i:02d}aaaaaaaaaaaa",
                "predicted_value": "accept",
                "predicted_confidence": "high",
                "corrected_value": "accept",
                "reviewer_id": "alice",
                "timestamp_utc": f"2026-05-01T0{i}:00:00Z",
                "rationale": "ok",
                "run_id": "r1",
            }
        )
    for i in range(2):
        store.append(
            {
                "target_kind": "activity_completion",
                "target_id": f"A0{i:02d}",
                "predicted_value": "accept",
                "predicted_confidence": "medium",
                "corrected_value": "reject",
                "reviewer_id": "alice",
                "timestamp_utc": f"2026-05-02T0{i}:00:00Z",
                "rationale": "behind schedule",
                "run_id": "r1",
            }
        )
    # Default DEFAULT_TARGET_KINDS=('element_acceptance',) keeps 3 records.
    response = run_calibration_report(paths)
    assert response["n_replayed_records"] == 3


def test_run_target_kinds_none_includes_all(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.corrections_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _seed_corrections(paths.corrections_jsonl, n_correct=3, n_wrong=1)
    response = run_calibration_report(paths, target_kinds=None)
    assert response["report"]["calibration_run_provenance"]["target_kinds"] is None


def test_run_creates_parent_dir_if_missing(tmp_path: Path) -> None:
    """Replay must work for a brand-new run dir."""
    nested = tmp_path / "deeply" / "nested" / "run"
    paths = CalibrationArtefactPaths.under_run_dir(nested)
    paths.corrections_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _seed_corrections(paths.corrections_jsonl)
    run_calibration_report(paths)
    assert paths.report_json.exists()


# ---------------------------------------------------------------------------
# run_calibration_report — error paths
# ---------------------------------------------------------------------------


def test_run_raises_not_found_when_no_log(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with pytest.raises(CalibrationApiError) as exc_info:
        run_calibration_report(paths)
    assert exc_info.value.code == "not_found"


def test_run_rejects_invalid_n_bins(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.corrections_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _seed_corrections(paths.corrections_jsonl)
    with pytest.raises(CalibrationApiError) as exc_info:
        run_calibration_report(paths, n_bins=0)
    assert exc_info.value.code == "invalid_input"


def test_run_rejects_invalid_strategy(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.corrections_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _seed_corrections(paths.corrections_jsonl)
    with pytest.raises(CalibrationApiError) as exc_info:
        run_calibration_report(paths, strategy="bogus")
    assert exc_info.value.code == "invalid_input"


def test_run_rejects_blank_target_kind_entries(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.corrections_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _seed_corrections(paths.corrections_jsonl)
    with pytest.raises(CalibrationApiError) as exc_info:
        run_calibration_report(paths, target_kinds=["element_acceptance", ""])
    assert exc_info.value.code == "invalid_input"


# ---------------------------------------------------------------------------
# get_latest_report
# ---------------------------------------------------------------------------


def test_get_latest_returns_report(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.corrections_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _seed_corrections(paths.corrections_jsonl)
    run_calibration_report(paths)
    report = get_latest_report(paths)
    assert report["schema_version"] == "calibration_report.v1"


def test_get_latest_raises_not_found_when_absent(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with pytest.raises(CalibrationApiError) as exc_info:
        get_latest_report(paths)
    assert exc_info.value.code == "not_found"


def test_get_latest_raises_persistence_failed_on_corrupt_json(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.report_json.parent.mkdir(parents=True, exist_ok=True)
    paths.report_json.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(CalibrationApiError) as exc_info:
        get_latest_report(paths)
    assert exc_info.value.code == "persistence_failed"


def test_get_latest_raises_persistence_failed_on_wrong_schema(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.report_json.parent.mkdir(parents=True, exist_ok=True)
    paths.report_json.write_text(
        json.dumps({"schema_version": "wrong.v0"}), encoding="utf-8"
    )
    with pytest.raises(CalibrationApiError) as exc_info:
        get_latest_report(paths)
    assert exc_info.value.code == "persistence_failed"


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------


def test_calibration_api_error_to_dict_shape() -> None:
    err = CalibrationApiError(code="not_found", message="no log", details={"x": 1})
    assert err.to_dict() == {
        "error": {"code": "not_found", "message": "no log", "details": {"x": 1}}
    }
