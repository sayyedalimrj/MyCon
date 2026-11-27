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
    row = approved(quantity="0", absence_verified="True", absence_evidence_ref="synthetic-site-audit-1",
                   absence_document_sha256=HASH, absence_approved_by="independent-fixture-reviewer",
                   absence_approved_at_utc=DATE.isoformat())
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
    with pytest.raises(ValueError, match="train_test_site_leakage"):
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


def test_absence_audit_requires_as_of_authorization():
    audit = dict(absence_verified="True", absence_evidence_ref="site-synthetic-absence",
                 absence_document_sha256=HASH, absence_approved_by="independent-reviewer")
    missing_date = status_for_element(approved(quantity="0", **audit), data_date=DATE, mapped=True, unique=True)
    assert missing_date["construction_status"] == "unknown"
    future_date = status_for_element(approved(
        quantity="0", absence_approved_at_utc="2026-01-17T00:00:00Z", **audit
    ), data_date=DATE, mapped=True, unique=True)
    assert future_date["construction_status"] == "unknown"
    assert future_date["physical_progress_estimate"] == 0.0


def test_unmapped_ifc_quantity_remains_numeric_but_categorical_unknown():
    output = status_for_element(approved(), data_date=DATE, mapped=False, unique=True)
    assert output["construction_status"] == "unknown"
    assert output["physical_progress_estimate"] == pytest.approx(0.4)
    assert "ifc_id_not_in_approved_activity_mapping" in output["reason_codes"]


def test_dashboard_contains_civil_qa_and_quantity_reports(tmp_path):
    assert _run(tmp_path, "progress-civil-audit") == 0
    report = json.loads((tmp_path / "progress-civil-audit" / "progress_dashboard.json").read_text(encoding="utf-8"))
    assert report["approved_quantity_report"]["approved_count"] == 1
    assert len(report["observability_report"]["elements"]) == 2
    assert report["observability_report"]["metric_and_ifc_geometry_gate"] == "blocked"
    assert report["uncertain_elements_report"]["count"] == 2
    assert report["manager_summary"]["physical_progress_pct"] is None
    assert report["manager_summary"]["progress_lower_bound_pct"] == pytest.approx(60)
    assert report["manager_summary"]["progress_upper_bound_pct"] == pytest.approx(85)
    assert report["wbs_boq_weight_report"]["activities"][0]["wbs_code"] == "1.1"
    assert report["manager_summary"]["decision_quality_gate"] == "blocked"


def test_verified_stage8_occlusion_is_nonobservable_not_unbuilt(tmp_path):
    args = [
        "--element-metrics-csv", str(FIXTURE / "elements_stage9.csv"),
        "--schedule-csv", str(FIXTURE / "schedule.csv"),
        "--mapping-csv", str(FIXTURE / "mapping.csv"),
        "--project-root", str(FIXTURE.parents[2]),
        "--data-date-utc", DATE.isoformat(),
        "--output-root", str(tmp_path),
        "--research-run-id", "progress-verified-geometry-synthetic",
        "--geometry-manifest", str(FIXTURE / "bim_geometry_evidence.v1.json"),
        "--expected-ifc-sha256", "1" * 64,
        "--expected-scan-sha256", "2" * 64,
        "--expected-registration-sha256", "3" * 64,
        "--expected-project-id", "mycon-round2-synthetic",
    ]
    assert main(args) == 0
    output = tmp_path / "progress-verified-geometry-synthetic"
    with (output / "element_status.csv").open(newline="", encoding="utf-8") as fh:
        states = {r["global_id"]: r for r in csv.DictReader(fh)}
    e1 = states["0AbCdeFgHiJkLmNoPqRsTu"]
    e2 = states["1BcDefGhIjKlMnOpQrStUv"]
    assert e1["construction_status"] == "in_progress"
    assert e1["observation_status"] == "observed"
    assert len(json.loads(e1["image_evidence_refs_json"])) == 1
    assert e2["construction_status"] == "unknown"
    assert e2["observation_status"] == "not_observable"
    assert e2["occlusion_state"] == "occluded"
    dashboard = json.loads((output / "progress_dashboard.json").read_text(encoding="utf-8"))
    assert dashboard["n_not_observable"] == 1
    assert dashboard["observability_report"]["geometry_manifest_valid"] is True
    assert dashboard["manager_summary"]["physical_progress_pct"] is None


def test_unverified_nonmetric_stage8_manifest_fails_closed(tmp_path):
    args = [
        "--element-metrics-csv", str(FIXTURE / "elements_stage9.csv"),
        "--schedule-csv", str(FIXTURE / "schedule.csv"),
        "--mapping-csv", str(FIXTURE / "mapping.csv"),
        "--project-root", str(FIXTURE.parents[2]),
        "--data-date-utc", DATE.isoformat(),
        "--output-root", str(tmp_path),
        "--research-run-id", "progress-invalid-metric-fixture",
        "--geometry-manifest", str(FIXTURE.parents[0] / "comparison_ready_fixture" / "comparison_ready.json"),
        "--expected-ifc-sha256", "1" * 64,
        "--expected-scan-sha256", "2" * 64,
        "--expected-registration-sha256", "3" * 64,
        "--expected-project-id", "mycon-round2-synthetic",
    ]
    assert main(args) == 0
    dashboard = json.loads((tmp_path / "progress-invalid-metric-fixture" / "progress_dashboard.json").read_text(encoding="utf-8"))
    assert dashboard["geometry_validation"]["geometry_adapter_valid"] is False
    assert dashboard["observability_report"]["metric_and_ifc_geometry_gate"] == "blocked"
    assert dashboard["manager_summary"]["physical_progress_pct"] is None
    assert dashboard["manager_summary"]["progress_lower_bound_pct"] == pytest.approx(60)
    assert dashboard["n_not_observable"] == 0


def test_no_ifc_index_is_not_certified_identity(tmp_path):
    assert _run(tmp_path, "progress-unverified-ifc") == 0
    root = tmp_path / "progress-unverified-ifc"
    with (root / "element_status.csv").open(encoding="utf-8", newline="") as fh:
        statuses = list(csv.DictReader(fh))
    assert all(row["ifc_identity_quality_gate"] == "blocked" for row in statuses)
    assert all(row["ifc_linked_construction_status"] == "unknown" for row in statuses)
    report = json.loads((root / "progress_dashboard.json").read_text(encoding="utf-8"))
    assert report["ifc_identity_validation"]["valid"] is False
    assert report["field_release_quality_gate"] != "pass"


def test_hashed_but_fake_ifc_inventory_never_certifies_globalid(tmp_path):
    """Pure SHA matching is NOT an IFC-entity check; optional parser is mandatory."""
    from pipeline.stage_09_progress.ifc_identity_adapter import read_ifc_identity_index
    original = tmp_path / "model.ifc"
    original.write_text("NOT AN IFC FILE: SYNTHETIC TEST ONLY", encoding="utf-8")
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    manifest = tmp_path / "index.json"
    manifest.write_text(json.dumps({
        "schema_version": "mycon.ifc_identity_index.v1",
        "ifc_file": {"path": "model.ifc", "sha256": original_hash},
        "coordinate_frame": "bim_world",
        "units": "m",
        "ifc_length_scale_to_m": 1.0,
        "provenance": {"generator": "synthetic", "reviewer": "synthetic",
                       "approval_document_sha256": HASH},
        "elements": [{"global_id": "0AbCdeFgHiJkLmNoPqRsTu", "ifc_class": "IfcWall"}],
    }), encoding="utf-8")
    report = read_ifc_identity_index(
        manifest, project_root=tmp_path, expected_ifc_sha256=original_hash)
    assert report.valid is False
    assert not report.elements
    assert (any("parser" in x for x in report.reason_codes)
            or "global_id_not_found_in_ifc" in report.reason_codes)


def test_ifc_index_outside_project_is_blocked(tmp_path):
    from pipeline.stage_09_progress.ifc_identity_adapter import read_ifc_identity_index
    elsewhere = tmp_path.parent / "external_invalid_ifc_index_fixture.json"
    # No need to create or touch anything outside the isolated fixture root.
    result = read_ifc_identity_index(elsewhere, project_root=tmp_path)
    assert result.valid is False
    assert result.elements == {}


def _upstream_fixture(tmp_path, owner):
    """Use the precise producer receipt shapes; no real geometry claimed."""
    from pipeline.stage_09_progress.upstream_receipt_audit import SCHEMAS
    names = (["registration_qa.json", "feature_evidence.json", "unknown_regions.json"]
             if owner == "A" else ["scale_estimate.json", "alignment_transform.json",
             "heldout_control_errors.csv", "deviation_heatmap.json", "viewer_handoff.json"])
    folder = tmp_path / owner
    folder.mkdir()
    outputs = {}
    for name in names:
        asset = folder / name
        asset.write_text("{}")
        outputs[name] = {
            ("path_relative_to_project" if owner == "A" else "path"): asset.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(), "bytes": asset.stat().st_size,
        }
    doc = {"schema_version": SCHEMAS[owner], "outputs": outputs,
           "run_id": "geom-civil-synthetic" if owner == "A" else "metric-bim-synthetic"}
    if owner == "A":
        doc.update(registration_status="UNREGISTERED", camera_and_depth_co_registered=False,
                   metric_scale_verified=False, feature_photo_projection_allowed=False)
    else:
        doc.update(metric_validation="pending", metric_scale_verified=False,
                   registration_verified=False, can_feed_stage8=False,
                   target_frame="bim_world", target_units="m")
    path = folder / "manifest.json"
    path.write_text(json.dumps(doc))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_upstream_receipts_valid_integrity_is_not_metric_authority(tmp_path):
    from pipeline.stage_09_progress.upstream_receipt_audit import audit_upstream_receipts
    a, ah = _upstream_fixture(tmp_path, "A")
    d, dh = _upstream_fixture(tmp_path, "D")
    report = audit_upstream_receipts(project_root=tmp_path, a_manifest=a, a_sha256=ah,
                                    d_manifest=d, d_sha256=dh)
    assert report["A"]["integrity_valid"] and report["D"]["integrity_valid"]
    assert report["A"]["verified_artifact_count"] == 3
    assert report["D"]["verified_artifact_count"] == 5
    assert report["direct_geometry_gate"] == "blocked"
    assert not report["A"]["geometry_authority"] and not report["D"]["geometry_authority"]
    (d.parent / "scale_estimate.json").write_text("tampered")
    report = audit_upstream_receipts(project_root=tmp_path, d_manifest=d, d_sha256=dh)
    assert report["D"]["integrity_valid"] is False


def test_stale_geometry_csv_claims_are_cleared(tmp_path):
    from pipeline.stage_09_progress.enrich_scientific_evidence import enrich_scientific_evidence_rows
    row = approved(geometry_adapter_valid=True, geometry_quality_gate="pass",
                   geometry_evidence_state="occluded", observability_source="raycast_zbuffer",
                   observability_ratio=0, geometry_n_views=8, comparison_ready_valid=True,
                   comparison_metric_scale_verified=True, comparison_registration_verified=True,
                   comparison_ifc_quality_gate="pass", geometric_agreement_state="conforming",
                   image_evidence_refs_json='[{"view_id":"invented"}]')
    rows, _ = enrich_scientific_evidence_rows([row], manifest_path=None, project_root=tmp_path, data_date=DATE)
    item = rows[0]
    assert item["physical_progress_estimate"] == pytest.approx(.4)
    assert item["geometry_adapter_valid"] is False
    assert item["geometry_quality_gate"] == "blocked"
    assert item["comparison_ifc_quality_gate"] == "blocked"
    assert item["image_evidence_refs_json"] == "[]"
    assert item["observability_ratio"] is None


def test_a_d_cli_diagnostics_preserve_unknown_and_boq_bounds(tmp_path):
    import shutil
    fixture = tmp_path / "fixture"
    shutil.copytree(FIXTURE, fixture)
    a, ah = _upstream_fixture(tmp_path, "A")
    d, dh = _upstream_fixture(tmp_path, "D")
    args = [
        "--element-metrics-csv", str(fixture / "elements_stage9.csv"),
        "--schedule-csv", str(fixture / "schedule.csv"),
        "--mapping-csv", str(fixture / "mapping.csv"),
        "--project-root", str(tmp_path), "--data-date-utc", DATE.isoformat(),
        "--output-root", str(tmp_path / "runs"), "--research-run-id", "progress-upstream-audit",
        "--a-handoff-manifest", str(a), "--expected-a-handoff-sha256", ah,
        "--d-metric-manifest", str(d), "--expected-d-metric-sha256", dh,
    ]
    assert main(args) == 0
    out = tmp_path / "runs/progress-upstream-audit"
    report = json.loads((out / "progress_dashboard.json").read_text())
    assert report["geometry_validation"]["upstream_receipt_audit"]["A"]["integrity_valid"]
    assert report["geometry_validation"]["upstream_receipt_audit"]["D"]["integrity_valid"]
    assert report["geometry_validation"]["geometry_adapter_valid"] is False
    assert report["overall"]["physical_progress_estimate"] is None
    assert report["overall"]["physical_progress_lower_bound_pct"] == pytest.approx(60)
    assert report["overall"]["physical_progress_upper_bound_pct"] == pytest.approx(85)
    assert report["overall"]["schedule_variance_bounds_pct"] == pytest.approx([10,35])
    assert report["overall"]["schedule_variance_percent"] is None
    assert report["provenance"]["input_artifacts"]["a_handoff_manifest"]["sha256"] == ah
    assert report["provenance"]["input_artifacts"]["d_metric_manifest"]["sha256"] == dh
    with (out / "element_status.csv").open(newline="", encoding="utf-8") as handle:
        states = list(csv.DictReader(handle))
    assert {s["construction_status"] for s in states} == {"in_progress", "unknown"}
    assert all(s["observation_status"] == "unknown" for s in states)


def test_missing_mapped_element_remains_explicit_unknown(tmp_path):
    import shutil
    fixture = tmp_path / "fixture"
    shutil.copytree(FIXTURE, fixture)
    src = fixture / "elements_stage9.csv"
    with src.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    with src.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(rows[0])
    args = [
        "--element-metrics-csv", str(src), "--schedule-csv", str(fixture / "schedule.csv"),
        "--mapping-csv", str(fixture / "mapping.csv"), "--project-root", str(tmp_path),
        "--data-date-utc", DATE.isoformat(), "--output-root", str(tmp_path / "runs"),
        "--research-run-id", "progress-missing-ledger",
    ]
    assert main(args) == 0
    out = tmp_path / "runs/progress-missing-ledger"
    with (out / "element_status.csv").open(newline="", encoding="utf-8") as fh:
        records = {r["global_id"]: r for r in csv.DictReader(fh)}
    assert len(records) == 2
    missing = records[rows[1]["global_id"]]
    assert missing["construction_status"] == "unknown"
    assert missing["physical_progress_estimate"] == ""
    assert missing["source_element_record_missing"] == "True"
    report = json.loads((out / "progress_dashboard.json").read_text())
    assert report["n_missing_source_element_records"] == 1
    assert report["overall"]["physical_progress_estimate"] is None
    assert report["overall"]["physical_progress_lower_bound_pct"] == pytest.approx(60)
    assert report["overall"]["physical_progress_upper_bound_pct"] == pytest.approx(85)
