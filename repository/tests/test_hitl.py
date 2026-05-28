"""Tests for :mod:`pipeline.common.hitl`.

Cover the schema validation, the append-only on-disk semantics, the
last-write-wins replay with explicit conflict surfacing, and the bridge
into :mod:`pipeline.common.calibration`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pipeline.common import calibration
from pipeline.common.hitl import (
    CORRECTION_SCHEMA_VERSION,
    VALID_DECISION_VALUES,
    VALID_TARGET_KINDS,
    Correction,
    CorrectionStore,
    build_calibration_records,
)


pytestmark = pytest.mark.lightweight


# ---------------------------------------------------------------------------
# Correction.from_dict — schema validation
# ---------------------------------------------------------------------------


def _valid_payload(**overrides) -> dict:
    base = {
        "target_kind": "element_acceptance",
        "target_id": "0xPq8M",
        "predicted_value": "accept",
        "predicted_confidence": "high",
        "corrected_value": "reject",
        "reviewer_id": "alice",
        "timestamp_utc": "2026-05-01T08:00:00Z",
        "rationale": "wall in tolerance but missing rebar capping",
        "evidence_refs": ["runs/r1/reports/element_progress.json"],
        "run_id": "r1",
    }
    base.update(overrides)
    return base


def test_from_dict_accepts_minimal_valid_payload() -> None:
    c = Correction.from_dict(_valid_payload())
    assert c.target_kind == "element_acceptance"
    assert c.corrected_value == "reject"
    assert c.evidence_refs == ("runs/r1/reports/element_progress.json",)


def test_from_dict_rejects_missing_required_keys() -> None:
    payload = _valid_payload()
    payload.pop("reviewer_id")
    with pytest.raises(ValueError):
        Correction.from_dict(payload)


def test_from_dict_rejects_unknown_target_kind() -> None:
    with pytest.raises(ValueError):
        Correction.from_dict(_valid_payload(target_kind="something_random"))


def test_from_dict_rejects_unknown_decision_value() -> None:
    with pytest.raises(ValueError):
        Correction.from_dict(_valid_payload(corrected_value="approved"))
    with pytest.raises(ValueError):
        Correction.from_dict(_valid_payload(predicted_value="approved"))


def test_from_dict_normalises_case() -> None:
    c = Correction.from_dict(
        _valid_payload(
            target_kind="ELEMENT_Acceptance",
            predicted_value="ACCEPT",
            corrected_value="Reject",
        )
    )
    assert c.target_kind == "element_acceptance"
    assert c.predicted_value == "accept"
    assert c.corrected_value == "reject"


def test_from_dict_tolerates_extra_keys_for_forward_compat() -> None:
    payload = _valid_payload()
    payload["future_field"] = {"foo": "bar"}
    c = Correction.from_dict(payload)
    assert c.target_kind == "element_acceptance"


def test_evidence_refs_string_is_normalised_to_one_element_tuple() -> None:
    c = Correction.from_dict(_valid_payload(evidence_refs="runs/r1/reports/x.json"))
    assert c.evidence_refs == ("runs/r1/reports/x.json",)


def test_valid_value_sets_are_stable() -> None:
    """Lock the public vocabularies so downstream consumers can rely on them."""
    assert "element_acceptance" in VALID_TARGET_KINDS
    assert "activity_completion" in VALID_TARGET_KINDS
    assert "vlm_answer" in VALID_TARGET_KINDS
    assert "anchor_validation" in VALID_TARGET_KINDS
    assert "registration_quality" in VALID_TARGET_KINDS
    assert {"accept", "reject", "uncertain", "rework"} <= VALID_DECISION_VALUES


# ---------------------------------------------------------------------------
# CorrectionStore — append-only persistence
# ---------------------------------------------------------------------------


def test_store_appends_one_record_per_call(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "corrections.jsonl")
    store.append(_valid_payload())
    store.append(_valid_payload(target_id="0xYz0L", reviewer_id="bob"))
    contents = (tmp_path / "corrections.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 2


def test_store_autopopulates_record_id_and_timestamp(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "log.jsonl")
    payload = _valid_payload()
    payload.pop("timestamp_utc")
    record = store.append(payload)
    assert record.timestamp_utc.endswith("Z")
    assert len(record.record_id) == 12  # 12-char hex prefix


def test_record_id_is_deterministic(tmp_path: Path) -> None:
    """Same payload (with explicit timestamp) → same record_id every time."""
    p = _valid_payload()
    store_a = CorrectionStore(tmp_path / "a.jsonl")
    store_b = CorrectionStore(tmp_path / "b.jsonl")
    rec_a = store_a.append(p)
    rec_b = store_b.append(p)
    assert rec_a.record_id == rec_b.record_id


def test_store_iter_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    store = CorrectionStore(path)
    store.append(_valid_payload())
    # Append a corrupt line manually.
    with path.open("a", encoding="utf-8") as f:
        f.write("not valid json\n")
        f.write(json.dumps({"missing": "everything"}) + "\n")
    # Iteration should still return only the one valid record.
    records = store.all()
    assert len(records) == 1


def test_store_handles_missing_file_as_empty(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "does-not-exist.jsonl")
    assert store.all() == []
    rep = store.replay()
    assert rep.n_total_records == 0
    assert rep.effective == ()


# ---------------------------------------------------------------------------
# Replay — last-write-wins + conflict surfacing
# ---------------------------------------------------------------------------


def test_replay_last_write_wins_when_no_conflict(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "log.jsonl")
    store.append(_valid_payload(timestamp_utc="2026-05-01T00:00:00Z", corrected_value="reject"))
    store.append(_valid_payload(timestamp_utc="2026-05-02T00:00:00Z", corrected_value="reject"))
    rep = store.replay()
    assert rep.n_total_records == 2
    assert len(rep.effective) == 1
    assert rep.effective[0].timestamp_utc == "2026-05-02T00:00:00Z"
    assert rep.conflicts == ()


def test_replay_emits_conflict_record_when_two_reviewers_disagree(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "log.jsonl")
    store.append(_valid_payload(timestamp_utc="2026-05-01T00:00:00Z", corrected_value="reject", reviewer_id="alice"))
    store.append(_valid_payload(timestamp_utc="2026-05-02T00:00:00Z", corrected_value="accept", reviewer_id="bob"))
    rep = store.replay()
    assert len(rep.conflicts) == 1
    c = rep.conflicts[0]
    assert c.earlier_corrected_value == "reject"
    assert c.later_corrected_value == "accept"
    # Last writer wins.
    assert rep.effective[0].corrected_value == "accept"
    assert rep.effective[0].reviewer_id == "bob"


def test_replay_filters_by_target_kind(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "log.jsonl")
    store.append(_valid_payload(target_kind="element_acceptance", target_id="e1"))
    store.append(_valid_payload(target_kind="activity_completion", target_id="a1"))
    rep = store.replay(target_kinds=["element_acceptance"])
    assert len(rep.effective) == 1
    assert rep.effective[0].target_kind == "element_acceptance"


def test_replay_filters_by_run_id(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "log.jsonl")
    store.append(_valid_payload(run_id="r1", target_id="e1"))
    store.append(_valid_payload(run_id="r2", target_id="e2"))
    rep = store.replay(run_id="r2")
    assert len(rep.effective) == 1
    assert rep.effective[0].run_id == "r2"


def test_replay_three_writes_two_disagreements_emit_two_conflicts(tmp_path: Path) -> None:
    """When three reviewers each disagree with the previous, we emit two
    adjacent ConflictRecords (one per adjacent pair) so the full audit
    trail is visible."""
    store = CorrectionStore(tmp_path / "log.jsonl")
    store.append(_valid_payload(timestamp_utc="2026-05-01T00:00:00Z", corrected_value="reject"))
    store.append(_valid_payload(timestamp_utc="2026-05-02T00:00:00Z", corrected_value="accept"))
    store.append(_valid_payload(timestamp_utc="2026-05-03T00:00:00Z", corrected_value="reject"))
    rep = store.replay()
    assert len(rep.conflicts) == 2


# ---------------------------------------------------------------------------
# build_calibration_records — bridge to calibration module
# ---------------------------------------------------------------------------


def test_build_calibration_records_marks_correctness_correctly(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "log.jsonl")
    # Pipeline predicted accept-with-high-confidence; reviewer agreed → correct.
    store.append(_valid_payload(predicted_value="accept", corrected_value="accept", target_id="e_correct"))
    # Pipeline predicted accept-with-high-confidence; reviewer overruled → incorrect.
    store.append(_valid_payload(predicted_value="accept", corrected_value="reject", target_id="e_wrong"))
    rep = store.replay()
    cal_records = build_calibration_records(rep)
    assert len(cal_records) == 2
    by_target = {r["target_id"]: r for r in cal_records}
    assert by_target["e_correct"]["correct"] is True
    assert by_target["e_wrong"]["correct"] is False


def test_build_calibration_records_feeds_calibration_report(tmp_path: Path) -> None:
    """End-to-end: HITL log → replay → calibration report.

    This is the core value proposition of the module: HITL feedback
    becomes a *measurement* of reported confidence trustworthiness.
    """
    store = CorrectionStore(tmp_path / "log.jsonl")
    # Five 'high' confidences, all correct → low ECE.
    for i in range(5):
        store.append(
            _valid_payload(
                target_id=f"e_hi_{i}",
                predicted_confidence="high",
                predicted_value="accept",
                corrected_value="accept",
                timestamp_utc=f"2026-05-{i + 1:02d}T00:00:00Z",
            )
        )
    rep = store.replay()
    cal_records = build_calibration_records(rep)
    report = calibration.calibration_report(cal_records)
    assert report["n_samples"] == 5
    # Five 'high' (mapped to 0.85) all-correct predictions → ECE close to 0.15.
    assert report["metrics"]["expected_calibration_error"] < 0.20


def test_build_calibration_records_can_filter_by_kind(tmp_path: Path) -> None:
    store = CorrectionStore(tmp_path / "log.jsonl")
    store.append(_valid_payload(target_kind="element_acceptance", target_id="e1"))
    store.append(_valid_payload(target_kind="vlm_answer", target_id="r1::q1"))
    rep = store.replay()
    only_elem = build_calibration_records(rep, target_kinds=["element_acceptance"])
    assert len(only_elem) == 1
    assert only_elem[0]["target_kind"] == "element_acceptance"


# ---------------------------------------------------------------------------
# Schema version is stable
# ---------------------------------------------------------------------------


def test_schema_version_is_v1() -> None:
    """Lock the schema version. Bumps must be explicit."""
    assert CORRECTION_SCHEMA_VERSION == "hitl_correction.v1"
