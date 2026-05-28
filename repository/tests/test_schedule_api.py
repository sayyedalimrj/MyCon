"""Tests for :mod:`pipeline.service.schedule_api`.

These tests exercise the *pure functions* of the schedule-comparison
API directly, without spinning up FastAPI. The router is a thin
HTTP wrapper around the same functions; covering the functions
covers the contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.service.schedule_api import (
    ScheduleApiError,
    ScheduleArtefactPaths,
    get_activity_detail,
    get_dashboard_summary,
    get_element_status,
    get_schedule_variance,
    list_schedule_activities,
)


pytestmark = pytest.mark.lightweight


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def _build_run(tmp_path: Path) -> ScheduleArtefactPaths:
    """Build a tiny synthetic run dir with all five artefacts.

    Layout:

        tmp_path/
          inputs/
            schedule.csv
            bim_schedule_mapping.csv
          reports/
            element_metrics.csv
            activity_progress.json
            schedule_variance.json
            dashboard_summary.json
    """
    inputs = tmp_path / "inputs"
    reports = tmp_path / "reports"
    inputs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    (inputs / "schedule.csv").write_text(
        "activity_id,activity_name,planned_start_iso,planned_finish_iso\n"
        "A0001,Foundations,2026-03-01,2026-04-01\n"
        "A0432,Floor 2 Zone B walls,2026-04-01,2026-05-01\n",
        encoding="utf-8",
    )
    (inputs / "bim_schedule_mapping.csv").write_text(
        "activity_id,ifc_global_id,weight\n"
        "A0001,Y1,1.0\n"
        "A0432,X1,1.0\n"
        "A0432,X2,1.0\n"
        "A0432,X3,1.0\n",
        encoding="utf-8",
    )
    (reports / "element_metrics.csv").write_text(
        "global_id,name,status\n"
        "Y1,Foundation,likely_completed\n"
        "X1,Wall1,likely_completed\n"
        "X2,Wall2,partially_observed\n"
        "X3,Wall3,not_evidenced\n",
        encoding="utf-8",
    )

    # Run Stage 11 to fill the three JSONs.
    from pipeline.stage_11_schedule_variance.run_schedule_variance import main as stage11_main

    rc = stage11_main(
        [
            "--schedule-csv", str(inputs / "schedule.csv"),
            "--mapping-csv", str(inputs / "bim_schedule_mapping.csv"),
            "--element-metrics-csv", str(reports / "element_metrics.csv"),
            "--activity-progress-json", str(reports / "activity_progress.json"),
            "--schedule-variance-json", str(reports / "schedule_variance.json"),
            "--dashboard-summary-json", str(reports / "dashboard_summary.json"),
            "--data-date-utc", "2026-04-16",
        ]
    )
    assert rc == 0
    return ScheduleArtefactPaths.under_run_dir(tmp_path)


# ---------------------------------------------------------------------------
# list_schedule_activities
# ---------------------------------------------------------------------------


def test_list_schedule_activities_returns_all(tmp_path: Path) -> None:
    paths = _build_run(tmp_path)
    resp = list_schedule_activities(paths)
    assert resp["schema_version"] == "schedule_activities_response.v1"
    assert resp["n_activities"] == 2
    aids = {a["activity_id"] for a in resp["activities"]}
    assert aids == {"A0001", "A0432"}


def test_list_schedule_activities_uses_data_date(tmp_path: Path) -> None:
    paths = _build_run(tmp_path)
    resp = list_schedule_activities(paths, data_date_iso="2026-04-16")
    by_id = {a["activity_id"]: a for a in resp["activities"]}
    # A0001 (Mar 1 -> Apr 1) is fully complete by April 16.
    assert by_id["A0001"]["planned_percent_complete"] == pytest.approx(100.0, abs=0.5)
    # A0432 (Apr 1 -> May 1) is half-way on April 16.
    assert by_id["A0432"]["planned_percent_complete"] == pytest.approx(50.0, abs=1.0)


def test_list_schedule_activities_invalid_data_date_raises(tmp_path: Path) -> None:
    paths = _build_run(tmp_path)
    with pytest.raises(ScheduleApiError) as exc_info:
        list_schedule_activities(paths, data_date_iso="not-a-date")
    assert exc_info.value.code == "invalid_input"


def test_list_schedule_activities_missing_csv_raises_not_found(tmp_path: Path) -> None:
    paths = ScheduleArtefactPaths.under_run_dir(tmp_path)
    with pytest.raises(ScheduleApiError) as exc_info:
        list_schedule_activities(paths)
    assert exc_info.value.code == "not_found"


# ---------------------------------------------------------------------------
# get_activity_detail
# ---------------------------------------------------------------------------


def test_get_activity_detail_returns_planned_and_actual(tmp_path: Path) -> None:
    paths = _build_run(tmp_path)
    resp = get_activity_detail(paths, "A0432", data_date_iso="2026-04-16")
    assert resp["schema_version"] == "activity_detail_response.v1"
    assert resp["activity_id"] == "A0432"
    assert resp["planned_percent_complete"] == pytest.approx(50.0, abs=1.0)
    assert resp["actual"] is not None
    assert resp["actual"]["activity_id"] == "A0432"
    # Three mapped elements (X1, X2, X3) returned in the response.
    assert {e["ifc_global_id"] for e in resp["mapped_elements"]} == {"X1", "X2", "X3"}


def test_get_activity_detail_returns_planned_only_when_variance_absent(tmp_path: Path) -> None:
    """If schedule_variance.json hasn't been generated yet, the response
    should still return planned + mapped_elements (actual=None)."""
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "schedule.csv").write_text(
        "activity_id,activity_name,planned_start_iso,planned_finish_iso\n"
        "A1,Walls,2026-04-01,2026-05-01\n",
        encoding="utf-8",
    )
    (inputs / "bim_schedule_mapping.csv").write_text(
        "activity_id,ifc_global_id\nA1,X1\n",
        encoding="utf-8",
    )
    paths = ScheduleArtefactPaths.under_run_dir(tmp_path)
    resp = get_activity_detail(paths, "A1", data_date_iso="2026-04-16")
    assert resp["actual"] is None
    assert {e["ifc_global_id"] for e in resp["mapped_elements"]} == {"X1"}


def test_get_activity_detail_unknown_activity_raises(tmp_path: Path) -> None:
    paths = _build_run(tmp_path)
    with pytest.raises(ScheduleApiError) as exc_info:
        get_activity_detail(paths, "DOES_NOT_EXIST")
    assert exc_info.value.code == "not_found"


def test_get_activity_detail_invalid_id_raises(tmp_path: Path) -> None:
    paths = _build_run(tmp_path)
    with pytest.raises(ScheduleApiError) as exc_info:
        get_activity_detail(paths, "")
    assert exc_info.value.code == "invalid_input"


# ---------------------------------------------------------------------------
# get_schedule_variance
# ---------------------------------------------------------------------------


def test_get_schedule_variance_returns_payload(tmp_path: Path) -> None:
    paths = _build_run(tmp_path)
    resp = get_schedule_variance(paths)
    assert resp["schema_version"] == "schedule_variance.v1"
    assert resp["n_activities"] == 2


def test_get_schedule_variance_missing_raises_not_found(tmp_path: Path) -> None:
    paths = ScheduleArtefactPaths.under_run_dir(tmp_path)
    with pytest.raises(ScheduleApiError) as exc_info:
        get_schedule_variance(paths)
    assert exc_info.value.code == "not_found"


def test_get_schedule_variance_bad_schema_raises_inconsistent(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "schedule_variance.json").write_text(
        json.dumps({"schema_version": "wrong.v0", "activities": []}), encoding="utf-8"
    )
    paths = ScheduleArtefactPaths.under_run_dir(tmp_path)
    with pytest.raises(ScheduleApiError) as exc_info:
        get_schedule_variance(paths)
    assert exc_info.value.code == "inconsistent"


# ---------------------------------------------------------------------------
# get_dashboard_summary
# ---------------------------------------------------------------------------


def test_get_dashboard_summary_returns_payload(tmp_path: Path) -> None:
    paths = _build_run(tmp_path)
    resp = get_dashboard_summary(paths)
    assert resp["schema_version"] == "dashboard_summary.v1"
    assert resp["kpi"]["n_activities"] == 2


def test_get_dashboard_summary_missing_raises_not_found(tmp_path: Path) -> None:
    paths = ScheduleArtefactPaths.under_run_dir(tmp_path)
    with pytest.raises(ScheduleApiError) as exc_info:
        get_dashboard_summary(paths)
    assert exc_info.value.code == "not_found"


# ---------------------------------------------------------------------------
# get_element_status
# ---------------------------------------------------------------------------


def test_get_element_status_returns_row_and_activities(tmp_path: Path) -> None:
    paths = _build_run(tmp_path)
    resp = get_element_status(paths, "X1")
    assert resp["schema_version"] == "element_status_response.v1"
    assert resp["ifc_global_id"] == "X1"
    assert resp["element_metrics_row"]["status"] == "likely_completed"
    assert any(a["activity_id"] == "A0432" for a in resp["mapped_to_activities"])


def test_get_element_status_missing_id_raises(tmp_path: Path) -> None:
    paths = _build_run(tmp_path)
    with pytest.raises(ScheduleApiError) as exc_info:
        get_element_status(paths, "DOES_NOT_EXIST")
    assert exc_info.value.code == "not_found"


def test_get_element_status_invalid_input_raises(tmp_path: Path) -> None:
    paths = _build_run(tmp_path)
    with pytest.raises(ScheduleApiError) as exc_info:
        get_element_status(paths, "")
    assert exc_info.value.code == "invalid_input"


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------


def test_schedule_api_error_to_dict_shape() -> None:
    err = ScheduleApiError(code="not_found", message="boom", details={"x": 1})
    d = err.to_dict()
    assert d["error"]["code"] == "not_found"
    assert d["error"]["message"] == "boom"
    assert d["error"]["details"] == {"x": 1}
