"""Tests for :mod:`pipeline.service.hitl_api`.

Exercise the pure read/write functions directly. The router is a thin
HTTP wrapper around the same functions and the router contract is
covered transitively by these tests + a small router-shape smoke test
gated on FastAPI availability.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.service.hitl_api import (
    HITL_LOG_BASENAME,
    HitlApiError,
    HitlArtefactPaths,
    list_corrections,
    submit_correction,
)


pytestmark = pytest.mark.lightweight


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def _paths(tmp_path: Path) -> HitlArtefactPaths:
    """One run dir, no pre-existing corrections."""
    return HitlArtefactPaths.under_run_dir(tmp_path)


def _valid_payload(**overrides) -> dict:
    base = {
        "target_kind": "element_acceptance",
        "target_id": "1Pq8MeKvD2vQ8XYZabcdef",
        "predicted_value": "accept",
        "predicted_confidence": "high",
        "corrected_value": "reject",
        "reviewer_id": "alice@example.com",
        "rationale": "Wall in tolerance but missing rebar capping",
        "evidence_refs": [
            "runs/example_walkthrough/element_metrics.csv",
        ],
        "run_id": "example_walkthrough",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# under_run_dir + path layout
# ---------------------------------------------------------------------------


def test_paths_under_run_dir_uses_canonical_layout(tmp_path: Path) -> None:
    paths = HitlArtefactPaths.under_run_dir(tmp_path)
    assert paths.corrections_jsonl == (tmp_path / "reports" / HITL_LOG_BASENAME).resolve()


def test_log_basename_is_locked() -> None:
    """Lock the basename so the dashboard / Stage 11 / replay tooling
    keep finding the file at the same place."""
    assert HITL_LOG_BASENAME == "hitl_corrections.jsonl"


# ---------------------------------------------------------------------------
# submit_correction — happy paths
# ---------------------------------------------------------------------------


def test_submit_creates_parent_dir_and_appends(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    assert not paths.corrections_jsonl.exists()
    resp = submit_correction(paths, _valid_payload())
    assert resp["schema_version"] == "hitl_submit_response.v1"
    assert paths.corrections_jsonl.exists()
    assert paths.corrections_jsonl.read_text(encoding="utf-8").count("\n") == 1


def test_submit_autopopulates_record_id_and_timestamp(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    payload = _valid_payload()
    payload.pop("timestamp_utc", None)
    resp = submit_correction(paths, payload)
    record = resp["correction"]
    assert "record_id" in record and len(record["record_id"]) == 12
    assert record["timestamp_utc"].endswith("Z")


def test_submit_returns_canonical_record_dict(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    resp = submit_correction(paths, _valid_payload())
    record = resp["correction"]
    assert record["target_kind"] == "element_acceptance"
    assert record["corrected_value"] == "reject"
    assert "evidence_refs" in record
    assert isinstance(record["evidence_refs"], list)


def test_submit_appends_in_a_round_trip_with_list(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    submit_correction(paths, _valid_payload(target_id="A"))
    submit_correction(paths, _valid_payload(target_id="B"))
    listed = list_corrections(paths)
    assert listed["n_total_records"] == 2
    assert listed["n_effective"] == 2
    target_ids = {c["target_id"] for c in listed["effective"]}
    assert target_ids == {"A", "B"}


# ---------------------------------------------------------------------------
# submit_correction — invalid input
# ---------------------------------------------------------------------------


def test_submit_rejects_non_dict_payload(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with pytest.raises(HitlApiError) as exc_info:
        submit_correction(paths, "not a dict")  # type: ignore[arg-type]
    assert exc_info.value.code == "invalid_input"


def test_submit_rejects_missing_required_field(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    payload = _valid_payload()
    payload.pop("reviewer_id")
    with pytest.raises(HitlApiError) as exc_info:
        submit_correction(paths, payload)
    assert exc_info.value.code == "invalid_input"


def test_submit_rejects_unknown_target_kind(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with pytest.raises(HitlApiError) as exc_info:
        submit_correction(paths, _valid_payload(target_kind="something_random"))
    assert exc_info.value.code == "invalid_input"


def test_submit_rejects_unknown_decision_value(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with pytest.raises(HitlApiError) as exc_info:
        submit_correction(paths, _valid_payload(corrected_value="approved"))
    assert exc_info.value.code == "invalid_input"


# ---------------------------------------------------------------------------
# list_corrections — replay shape
# ---------------------------------------------------------------------------


def test_list_returns_empty_replay_when_jsonl_absent(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    assert not paths.corrections_jsonl.exists()
    listed = list_corrections(paths)
    assert listed["n_total_records"] == 0
    assert listed["n_effective"] == 0
    assert listed["n_conflicts"] == 0
    assert listed["effective"] == []
    assert listed["conflicts"] == []


def test_list_filters_by_target_kinds(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    submit_correction(paths, _valid_payload(target_kind="element_acceptance"))
    submit_correction(paths, _valid_payload(target_kind="activity_completion", target_id="A0001"))
    listed = list_corrections(paths, target_kinds=["element_acceptance"])
    assert listed["n_effective"] == 1
    assert listed["effective"][0]["target_kind"] == "element_acceptance"


def test_list_surfaces_conflicts(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    submit_correction(
        paths,
        _valid_payload(
            timestamp_utc="2026-05-01T00:00:00Z",
            corrected_value="reject",
            reviewer_id="alice",
        ),
    )
    submit_correction(
        paths,
        _valid_payload(
            timestamp_utc="2026-05-02T00:00:00Z",
            corrected_value="accept",
            reviewer_id="bob",
        ),
    )
    listed = list_corrections(paths)
    assert listed["n_total_records"] == 2
    assert listed["n_conflicts"] == 1
    assert listed["effective"][0]["corrected_value"] == "accept"


def test_list_response_schema_version_is_locked(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    listed = list_corrections(paths)
    assert listed["schema_version"] == "hitl_list_response.v1"
    assert listed["schema_version_record"] == "hitl_correction.v1"


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------


def test_hitl_api_error_to_dict_shape() -> None:
    err = HitlApiError(code="invalid_input", message="bad payload", details={"x": 1})
    d = err.to_dict()
    assert d["error"]["code"] == "invalid_input"
    assert d["error"]["message"] == "bad payload"
    assert d["error"]["details"] == {"x": 1}
