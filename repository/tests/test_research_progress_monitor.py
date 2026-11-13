"""Conservative physical monitor contracts — synthetic fixture only, no site claims."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.stage_09_progress.element_status_export import (
    evaluate_element_truth, evaluate_expert_progress, status_for_element,
)
from pipeline.stage_11_schedule_variance.run_physical_monitor import FILENAMES, main

DATE = dt.datetime(2026, 1, 16, tzinfo=dt.timezone.utc)
HASH = "a" * 64
FIXTURE = Path(__file__).resolve().parents[1] / "configs/research_progress/round2_contract_fixture"


def approved(quantity="4", planned="10", **kwargs):
    return {
        "global_id": "0AbCdeFgHiJkLmNoPqRsTu",
        "status": "not_evidenced",
        "geometry_adapter_valid": False,
        "planned_quantity": planned,
        "executed_quantity": quantity,
        "quantity_unit": "m2",
        "quantity_verified": "True",
        "quantity_source": "approved_measurement",
        "quantity_record_id": "synthetic-record-1",
        "quantity_approval_id": "synthetic-approval-1",
        "quantity_approved_by": "synthetic-reviewer",
        "quantity_approved_at_utc": DATE.isoformat(),
        "quantity_as_of_utc": DATE.isoformat(),
        "quantity_document_sha256": HASH,
        **kwargs,
    }


def truth(global_id="0AbCdeFgHiJkLmNoPqRsTu", *, status="in_progress", fraction="0.4",
          split="test", site="fixture", session="synthetic-capture-1"):
    return {
        "global_id": global_id,
        "ground_truth_status": status,
        "ground_truth_fraction": fraction,
        "split": split, "site_id": site, "capture_session_id": session,
        "reviewer": "independent-fixture-reviewer",
        "verified_at_utc": DATE.isoformat(),
        "document_sha256": HASH,
    }


@pytest.mark.parametrize("executed,expected", [
    ("10", "built"), ("3", "in_progress"), ("0", "unknown"),
    ("", "unknown"), ("11", "unknown"),
])
def test_quantity_and_state_fail_closed(executed, expected):
    row = approved(quantity=executed)
    assert status_for_element(row, data_date=DATE, mapped=True, unique=True)["construction_status"] == expected


def test_verified_zero_needs_independent_absence_reference():
    row = approved(quantity="0", absence_verified="True", absence_evidence_ref="synthetic-site-audit-1")
    result = status_for_element(row, data_date=DATE, mapped=True, unique=True)
    assert result["construction_status"] == "not_built"
    assert result["physical_progress_estimate"] == 0.0


def test_not_evidenced_never_equates_to_not_built():
    row = approved(quantity="", geometry_evidence_state="occluded")
    result = status_for_element(row, data_date=DATE, mapped=True, unique=True)
    assert result["construction_status"] == "unknown"
    assert result["physical_progress_estimate"] is None


def test_unverified_geometry_cannot_report_observability():
    row = approved(geometry_evidence_state="occluded", observability_ratio="0",
                   observability_source="raycast_zbuffer")
    result = status_for_element(row, data_date=DATE, mapped=True, unique=True)
    assert result["observability_ratio"] is None
    assert result["occlusion_state"] == "unknown"


def test_duplicate_global_id_denies_verified_progress():
    result = status_for_element(approved(), data_date=DATE, mapped=True, unique=False)
    assert result["construction_status"] == "unknown"
    assert result["physical_progress_estimate"] is None


def test_ground_truth_metrics_only_on_independent_test_label():
    prediction = status_for_element(approved(), data_date=DATE, mapped=True, unique=True)
    score = evaluate_element_truth([prediction], [truth()], data_date=DATE)
    assert score["per_class"]["in_progress"]["f1"] == pytest.approx(1)
    assert score["fraction_mae"] == pytest.approx(0)
    assert score["fraction_mae_n"] == 1


def test_class_abstention_does_not_impute_numeric_zero():
    prediction = status_for_element(approved(quantity=""), data_date=DATE, mapped=True, unique=True)
    result = evaluate_element_truth([prediction], [truth()], data_date=DATE)
    assert result["unknown_or_missing_predictions"] == 1
    assert result["per_class"]["in_progress"]["recall"] == 0
    assert result["fraction_mae"] is None


def test_gt_leakage_is_rejected():
    t1 = truth(split="train")
    t2 = truth(global_id="1BcDefGhIjKlMnOpQrStUv", split="test")
    with pytest.raises(ValueError, match="train_test_capture_session_leakage"):
        evaluate_element_truth([], [t1, t2], data_date=DATE)


def test_expert_report_requires_independent_as_of_fields():
    report = [{
        "activity_id": "ACT-WALL-01", "expert_progress_pct": "50",
        "split": "test", "site_id": "fixture", "capture_session_id": "synthetic-capture-1",
        "reviewer": "expert", "verified_at_utc": DATE.isoformat(),
        "document_sha256": HASH,
    }]
    result = evaluate_expert_progress([{"activity_id": "ACT-WALL-01", "physical_progress_estimate": 60}], report, data_date=DATE)
    assert result["mae_percentage_points"] == 10


def _run(tmp_path, run_id="progress-monitor-fixture"):
    return main([
        "--element-metrics-csv", str(FIXTURE / "elements_stage9.csv"),
        "--schedule-csv", str(FIXTURE / "schedule.csv"),
        "--mapping-csv", str(FIXTURE / "mapping.csv"),
        "--project-root", str(FIXTURE.parents[2]),
        "--data-date-utc", DATE.isoformat(),
        "--output-root", str(tmp_path),
        "--research-run-id", run_id,
    ])


def test_full_five_file_schema_and_conservative_bounds(tmp_path):
    source = FIXTURE / "elements_stage9.csv"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    assert _run(tmp_path) == 0
    out = tmp_path / "progress-monitor-fixture"
    assert sorted(x.name for x in out.iterdir()) == sorted(FILENAMES)
    with (out / "element_status.csv").open(newline="", encoding="utf-8") as handle:
        status = list(csv.DictReader(handle))
    assert len(status) == 2
    assert {r["construction_status"] for r in status} == {"in_progress", "unknown"}
    with (out / "element_metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    assert all("global_id" in row and "construction_status" in row for row in metrics)
    with (out / "activity_progress.csv").open(newline="", encoding="utf-8") as handle:
        activity = list(csv.DictReader(handle))
    assert activity[0]["activity_id"] == "ACT-WALL-01"
    variance = json.loads((out / "schedule_variance.json").read_text(encoding="utf-8"))
    dashboard = json.loads((out / "progress_dashboard.json").read_text(encoding="utf-8"))
    assert dashboard["schema_version"] == "mycon.physical_monitor.dashboard.v1"
    assert variance["run_id"] == dashboard["run_id"]
    assert dashboard["ground_truth_metrics"]["fraction_mae"] is None
    assert dashboard["geometry_validation"]["geometry_adapter_valid"] is False
    assert variance["overall"]["physical_progress_estimate"] is None
    assert variance["overall"]["physical_progress_lower_bound_pct"] == pytest.approx(60)
    assert variance["overall"]["physical_progress_upper_bound_pct"] == pytest.approx(85)
    assert dashboard["quality_gates"]["schedule_decision"] == "blocked"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_existing_run_is_not_overwritten(tmp_path):
    assert _run(tmp_path) == 0
    target = tmp_path / "progress-monitor-fixture" / "progress_dashboard.json"
    before = target.read_bytes()
    assert _run(tmp_path) == 2
    assert target.read_bytes() == before


def test_bad_schema_fails_before_writing(tmp_path):
    invalid = tmp_path / "wrong.csv"
    invalid.write_text("bad_key,status\n1,unknown\n", encoding="utf-8")
    code = main([
        "--element-metrics-csv", str(invalid),
        "--schedule-csv", str(FIXTURE / "schedule.csv"),
        "--mapping-csv", str(FIXTURE / "mapping.csv"),
        "--data-date-utc", DATE.isoformat(),
        "--output-root", str(tmp_path),
        "--research-run-id", "progress-bad",
    ])
    assert code == 2
    assert not (tmp_path / "progress-bad").exists()
