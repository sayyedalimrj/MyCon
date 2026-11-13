"""Strict quantity/BOQ-based progress, schedule baseline and uncertainty tests."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from pipeline.common.bim_schedule_mapping import BimScheduleMapping, MappingEntry, load_mapping_csv
from pipeline.common.schedule_io import Activity, Schedule, ScheduleProvenance
from pipeline.stage_11_schedule_variance.scientific_progress import build_scientific_progress_report
from pipeline.stage_11_schedule_variance.run_schedule_variance import main

pytestmark = pytest.mark.lightweight


def _schedule(ids=("A1",)):
    activities = tuple(
        Activity(activity_id=aid, activity_name=aid,
                 planned_start=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
                 planned_finish=dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc))
        for aid in ids
    )
    return Schedule(activities=activities, provenance=ScheduleProvenance(
        source_path="", source_sha256="", source_bytes=0, schema_version="schedule.v1",
        n_rows_total=len(ids), n_rows_kept=len(ids), n_rows_skipped=0
    ), activity_index={aid:i for i,aid in enumerate(ids)})


def _mapping(entries):
    entries = tuple(MappingEntry(
        *e,
        weight_approval_id="SYNTH-W1",
        weight_approved_by="SYNTH-REVIEWER",
        weight_approved_at_utc="2026-01-10T00:00:00+00:00",
        weight_document_sha256="7"*64,
    ) for e in entries)
    groups = {}
    for e in entries:
        groups.setdefault(e.activity_id, []).append(e)
    return BimScheduleMapping(entries=entries, by_activity={k:tuple(v) for k,v in groups.items()})


def _signed(gid, done=0, total=10):
    return {"global_id":gid, "planned_quantity":total, "executed_quantity":done,
            "quantity_unit":"m3", "quantity_source":"approved_measurement",
            "quantity_verified":"true", "registration_confidence":"high",
            "quantity_record_id":"SYNTH-Q1", "quantity_approval_id":"SYNTH-Q-APP",
            "quantity_approved_by":"SYNTH-REVIEWER",
            "quantity_approved_at_utc":"2026-01-15T00:00:00+00:00",
            "quantity_as_of_utc":"2026-01-16T00:00:00+00:00",
            "quantity_document_sha256":"6"*64}


def _report(mapping, rows, ids=("A1",), **kwargs):
    return build_scientific_progress_report(
        schedule=_schedule(ids), mapping=mapping, element_rows=rows,
        data_date=dt.datetime(2026, 1, 16, tzinfo=dt.timezone.utc),
        schedule_rows=kwargs.pop("schedule_rows", []), run_id="progress-test", **kwargs
    )


def _baseline(percent=50):
    return [{"activity_id":"A1", "baseline_source":"approved_baseline",
             "baseline_id":"baseline-r1", "approved_planned_percent":percent,
             "activity_weight":100, "activity_weight_basis":"approved_cost",
             "planned_as_of_utc":"2026-01-16T00:00:00+00:00",
             "baseline_approved_at_utc":"2026-01-10T00:00:00+00:00",
             "baseline_approved_by":"SYNTH-PLANNER",
             "baseline_document_sha256":"8"*64,
             "activity_weight_approval_id":"SYNTH-APP-100",
             "activity_weight_approved_by":"SYNTH-PLANNER",
             "activity_weight_approved_at_utc":"2026-01-10T00:00:00+00:00",
             "activity_weight_document_sha256":"9"*64}]


def test_missing_observation_is_not_zero_percent():
    m = _mapping([("A1", "E1", 1, "approved_boq")])
    a = _report(m, [])["activities"][0]
    assert a["physical_progress_estimate"] is None
    assert a["physical_progress_lower_bound_pct"] == 0
    assert a["physical_progress_upper_bound_pct"] == 100
    assert a["schedule_variance_status"] == "unknown_schedule_baseline"


def test_unequal_approved_weights_and_verified_quantities():
    m = _mapping([("A1", "E1", 3, "approved_boq"), ("A1", "E2", 1, "approved_boq")])
    report = _report(m, [_signed("E1",8),_signed("E2",2)], schedule_rows=_baseline())
    a = report["activities"][0]
    assert a["physical_progress_estimate"] == pytest.approx(65.0)
    assert a["physical_progress_lower_bound_pct"] == pytest.approx(65.0)
    assert a["physical_progress_upper_bound_pct"] == pytest.approx(65.0)
    assert a["schedule_variance_percent"] == pytest.approx(15.0)
    assert a["schedule_variance_status"] == "ahead"
    assert report["overall"]["physical_progress_estimate"] == pytest.approx(65.0)
    assert report["overall"]["schedule_variance_percent"] == pytest.approx(15.0)


def test_partial_observation_gives_identification_bounds_not_wilson():
    m = _mapping([("A1", "E1", 3, "approved_boq"), ("A1", "E2", 1, "approved_boq")])
    a = _report(m, [_signed("E1",8)], schedule_rows=_baseline())[ "activities"][0]
    assert a["physical_progress_estimate"] is None
    assert a["physical_progress_lower_bound_pct"] == pytest.approx(60.0)
    assert a["physical_progress_upper_bound_pct"] == pytest.approx(85.0)
    assert a["progress_interval_type"] == "partial_identification_not_confidence_interval"
    assert a["schedule_variance_status"] == "unknown_evidence"
    assert a["schedule_variance_bounds_pct"] == pytest.approx([10.0,35.0])


def test_missing_boq_weights_stops_invented_progress():
    a = _report(_mapping([("A1","E1",1,"")]), [_signed("E1",10)],
                schedule_rows=_baseline())["activities"][0]
    assert a["physical_progress_estimate"] is None
    assert a["physical_progress_lower_bound_pct"] is None
    assert "invalid_or_incomparable_element_weights" in a["warnings"]


def test_registration_failure_does_not_erase_independent_signed_quantity():
    a = _report(_mapping([("A1","E1",1,"approved_boq")]),
                [{**_signed("E1",3), "registration_confidence":"low"}],
                schedule_rows=_baseline())["activities"][0]
    assert a["physical_progress_estimate"] == pytest.approx(30.0)
    assert a["evidence_confidence"] is None


def test_legacy_acceptance_only_does_not_create_quantity():
    a = _report(_mapping([("A1","E1",1,"approved_boq")]),
                [{"global_id":"E1","fused_decision":"acceptable","coverage":"1",
                  "status":"likely_completed"}], schedule_rows=_baseline())["activities"][0]
    assert a["physical_progress_estimate"] is None
    assert a["schedule_variance_status"] == "unknown_evidence"


def test_invalid_approved_baseline_never_uses_planner_reported_actual():
    a = _report(_mapping([("A1","E1",1,"approved_boq")]), [_signed("E1",5)])["activities"][0]
    assert a["physical_progress_estimate"] == 50
    assert a["planned_percent_complete"] is None
    assert a["provisional_linear_planned_percent"] is not None
    assert a["schedule_variance_percent"] is None
    assert a["schedule_variance_status"] == "unknown_schedule_baseline"


def test_duplicate_rows_and_mapping_fail_closed():
    m = _mapping([("A1","E1",1,"approved_boq")])
    a = _report(m, [_signed("E1",10),_signed("E1",0)], schedule_rows=_baseline())["activities"][0]
    assert a["physical_progress_estimate"] is None
    dup = _mapping([("A1","E1",1,"approved_boq"),("A1","E1",2,"approved_boq")])
    a2 = _report(dup, [_signed("E1",10)], schedule_rows=_baseline())["activities"][0]
    assert a2["physical_progress_estimate"] is None


def test_full_verified_zero_quantity_is_real_zero():
    a = _report(_mapping([("A1","E1",1,"approved_boq")]), [_signed("E1",0)],
                schedule_rows=_baseline())["activities"][0]
    assert a["physical_progress_estimate"] == 0.0
    assert a["schedule_variance_status"] == "behind"


def test_cli_output_versioned_and_legacy_keys_preserved(tmp_path: Path):
    schedule = tmp_path/"schedule.csv"
    schedule.write_text(
        "activity_id,activity_name,planned_start_iso,planned_finish_iso,approved_planned_percent,baseline_source,baseline_id,activity_weight,activity_weight_basis,planned_as_of_utc,baseline_approved_at_utc,baseline_approved_by,baseline_document_sha256,activity_weight_approval_id,activity_weight_approved_by,activity_weight_approved_at_utc,activity_weight_document_sha256\n"
        "A1,Walls,2026-01-01,2026-02-01,50,approved_baseline,B1,10,approved_cost,2026-01-16T00:00:00+00:00,2026-01-10T00:00:00+00:00,SYNTH-PLANNER," + "8"*64 + ",SYNTH-AW1,SYNTH-PLANNER,2026-01-10T00:00:00+00:00," + "9"*64 + "\n", encoding="utf-8")
    mapping = tmp_path/"mapping.csv"
    mapping.write_text("activity_id,ifc_global_id,weight,weight_basis,weight_approval_id,weight_approved_by,weight_approved_at_utc,weight_document_sha256\nA1,E1,2,approved_boq,SYNTH-BOQ1,SYNTH-REVIEWER,2026-01-10T00:00:00+00:00," + "7"*64 + "\n", encoding="utf-8")
    elements = tmp_path/"metrics.csv"
    elements.write_text(
        "global_id,status,planned_quantity,executed_quantity,quantity_unit,quantity_source,quantity_verified,registration_confidence,quantity_record_id,quantity_approval_id,quantity_approved_by,quantity_approved_at_utc,quantity_as_of_utc,quantity_document_sha256\n"
        "E1,likely_completed,10,6,m3,approved_measurement,true,high,SYNTH-Q1,SYNTH-QAPP,SYNTH-REVIEWER,2026-01-15T00:00:00+00:00,2026-01-16T00:00:00+00:00," + "6"*64 + "\n", encoding="utf-8")
    args = ["--schedule-csv",str(schedule),"--mapping-csv",str(mapping),
            "--element-metrics-csv",str(elements),
            "--activity-progress-json",str(tmp_path/"activities.json"),
            "--schedule-variance-json",str(tmp_path/"schedule_out.json"),
            "--dashboard-summary-json",str(tmp_path/"dashboard.json"),
            "--research-run-id","progress-fixture",
            "--data-date-utc","2026-01-16"]
    assert main(args) == 0
    old = json.loads((tmp_path/"schedule_out.json").read_text(encoding="utf-8"))
    assert old["schema_version"] == "schedule_variance.v1"
    assert "activities" in old
    assert old["scientific_progress_v2"]["activities"][0]["physical_progress_estimate"] == 60
    v2 = json.loads((tmp_path/"progress-fixture/scientific_progress.v2.json").read_text(encoding="utf-8"))
    assert v2["schema_version"] == "mycon.scientific_progress.v2"
    assert v2["overall"]["physical_progress_estimate"] == 60
    assert v2["provenance"]["inputs"]["mapping_csv"]["sha256"]
    assert main(args) == 1  # Must not overwrite same research-run ID.


def test_stage9_to_stage11_physical_monitor_five_file_smoke(tmp_path: Path):
    """Included in the existing integration workflow's explicit B test selection."""
    import csv
    from pipeline.stage_11_schedule_variance.run_physical_monitor import main as monitor_main
    fixture = Path(__file__).resolve().parents[1] / "configs/research_progress/round2_contract_fixture"
    arguments = [
        "--element-metrics-csv", str(fixture / "elements_stage9.csv"),
        "--schedule-csv", str(fixture / "schedule.csv"),
        "--mapping-csv", str(fixture / "mapping.csv"),
        "--project-root", str(fixture.parents[2]),
        "--data-date-utc", "2026-01-16T00:00:00Z",
        "--output-root", str(tmp_path),
        "--research-run-id", "progress-monitor-ci-synthetic",
    ]
    assert monitor_main(arguments) == 0
    root = tmp_path / "progress-monitor-ci-synthetic"
    assert {p.name for p in root.iterdir()} == {
        "element_status.csv", "element_metrics.csv", "activity_progress.csv",
        "schedule_variance.json", "progress_dashboard.json"}
    with (root / "element_status.csv").open(encoding="utf-8", newline="") as handle:
        statuses = list(csv.DictReader(handle))
    assert {row["construction_status"] for row in statuses} == {"in_progress", "unknown"}
    dashboard = json.loads((root / "progress_dashboard.json").read_text(encoding="utf-8"))
    assert dashboard["ground_truth_metrics"]["fraction_mae"] is None
    assert dashboard["overall"]["physical_progress_estimate"] is None
    assert dashboard["overall"]["physical_progress_lower_bound_pct"] == pytest.approx(60)
    assert dashboard["overall"]["physical_progress_upper_bound_pct"] == pytest.approx(85)


def test_stage8_hashed_image_references_are_only_exposed_after_geometry_gate():
    from pipeline.stage_09_progress.geometry_evidence_adapter import load_geometry_evidence_manifest
    import csv
    fixture = Path(__file__).resolve().parents[1] / "configs/research_progress/round2_contract_fixture"
    bundle = load_geometry_evidence_manifest(
        fixture / "bim_geometry_evidence.v1.json",
        project_root=fixture.parents[2],
        expected_ifc_sha256="1" * 64,
        expected_scan_sha256="2" * 64,
        expected_registration_sha256="3" * 64,
        expected_project_id="mycon-round2-synthetic",
        expected_global_ids={"0AbCdeFgHiJkLmNoPqRsTu", "1BcDefGhIjKlMnOpQrStUv"},
    )
    assert bundle.valid, bundle.reason_codes
    evidence = bundle.for_element("0AbCdeFgHiJkLmNoPqRsTu")
    assert evidence["geometry_adapter_valid"] is True
    images = json.loads(evidence["image_evidence_refs_json"])
    assert len(images) == 1 and images[0]["view_id"] == "V1"
    assert len(images[0]["image_sha256"]) == 64
    assert json.loads(evidence["geometry_support_sample_ids_json"])
    invalid = load_geometry_evidence_manifest(
        fixture / "bim_geometry_evidence.v1.json",
        project_root=fixture.parents[2],
        expected_ifc_sha256="f" * 64,
    )
    assert not invalid.valid
    assert json.loads(invalid.for_element("0AbCdeFgHiJkLmNoPqRsTu")["image_evidence_refs_json"]) == []
