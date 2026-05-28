"""Schedule-comparison endpoints for the dashboard ``ScheduleCompare`` page.

These endpoints consume the artefacts produced by Stage 11 (see
:mod:`pipeline.stage_11_schedule_variance`) — ``activity_progress.json``,
``schedule_variance.json``, and ``dashboard_summary.json`` — plus the
canonical schedule CSV and the BIM <-> schedule mapping CSV they came
from. Nothing here recomputes geometry; this is a thin read layer over
the structured outputs.

The module is split from :mod:`pipeline.service.api` so it can be
imported and unit-tested without FastAPI: the **business logic** lives in
plain functions, and the FastAPI router in :func:`create_schedule_router`
is a thin wrapper around them.

Public read functions
---------------------

- :func:`list_schedule_activities`   ``GET /api/v1/schedule/activities``
- :func:`get_activity_detail`        ``GET /api/v1/schedule/activities/{activity_id}``
- :func:`get_schedule_variance`      ``GET /api/v1/schedule/variance``
- :func:`get_dashboard_summary`      ``GET /api/v1/schedule/dashboard``
- :func:`get_element_status`         ``GET /api/v1/elements/{global_id}``

Errors
------

Every public function raises :class:`ScheduleApiError` with one of:

- ``not_found``     — schedule / variance / dashboard / activity / element absent
- ``invalid_input`` — a query parameter or path part is malformed
- ``inconsistent``  — variance JSON disagrees with schedule CSV

The router maps these to HTTP 404 / 400 / 409 respectively.
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from pipeline.common.bim_schedule_mapping import (
    BimScheduleMapping,
    load_mapping_csv,
)
from pipeline.common.schedule_io import (
    Activity,
    Schedule,
    load_schedule_csv,
    parse_iso_datetime,
)


__all__ = [
    "ScheduleApiError",
    "ScheduleArtefactPaths",
    "list_schedule_activities",
    "get_activity_detail",
    "get_schedule_variance",
    "get_dashboard_summary",
    "get_element_status",
    "create_schedule_router",
]


@dataclass(frozen=True)
class ScheduleApiError(Exception):
    """Structured error type the router translates to HTTP."""

    code: str  # 'not_found' / 'invalid_input' / 'inconsistent'
    message: str
    details: Mapping[str, Any] | None = None

    def __str__(self) -> str:  # pragma: no cover - delegated to dataclass
        return f"{self.code}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": dict(self.details or {}),
            }
        }


@dataclass(frozen=True)
class ScheduleArtefactPaths:
    """Bundle of all the per-run artefact paths the schedule API reads.

    Carrying this as a single object makes the API trivially mockable in
    tests and keeps the router signature small. All paths are absolute.
    """

    schedule_csv: Path
    mapping_csv: Path
    activity_progress_json: Path
    schedule_variance_json: Path
    dashboard_summary_json: Path
    element_metrics_csv: Path | None = None  # optional; element-level reads

    @classmethod
    def under_run_dir(
        cls,
        run_dir: Path | str,
        *,
        schedule_csv: Path | str | None = None,
        mapping_csv: Path | str | None = None,
        element_metrics_csv: Path | str | None = None,
    ) -> "ScheduleArtefactPaths":
        """Construct paths assuming the canonical layout under a run dir.

        The canonical layout is documented in
        ``docs/end_to_end_finishing_plan.md`` Section 2:

            runs/<run_id>/reports/activity_progress.json
            runs/<run_id>/reports/schedule_variance.json
            runs/<run_id>/reports/dashboard_summary.json
        """
        run_dir = Path(run_dir).resolve()
        reports = run_dir / "reports"
        return cls(
            schedule_csv=Path(schedule_csv).resolve() if schedule_csv else run_dir / "inputs" / "schedule.csv",
            mapping_csv=Path(mapping_csv).resolve() if mapping_csv else run_dir / "inputs" / "bim_schedule_mapping.csv",
            activity_progress_json=reports / "activity_progress.json",
            schedule_variance_json=reports / "schedule_variance.json",
            dashboard_summary_json=reports / "dashboard_summary.json",
            element_metrics_csv=Path(element_metrics_csv).resolve() if element_metrics_csv else (reports / "element_metrics.csv" if (reports / "element_metrics.csv").exists() else None),
        )


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def _require_file(path: Path, *, kind: str) -> None:
    if not path.exists():
        raise ScheduleApiError(
            code="not_found",
            message=f"{kind} not found",
            details={"path": str(path)},
        )


def _read_json(path: Path, *, kind: str) -> dict[str, Any]:
    _require_file(path, kind=kind)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScheduleApiError(
            code="inconsistent",
            message=f"{kind} is not valid JSON",
            details={"path": str(path), "json_error": str(exc)},
        ) from exc


def _load_schedule(paths: ScheduleArtefactPaths) -> Schedule:
    _require_file(paths.schedule_csv, kind="schedule_csv")
    return load_schedule_csv(paths.schedule_csv)


def _load_mapping(paths: ScheduleArtefactPaths) -> BimScheduleMapping:
    _require_file(paths.mapping_csv, kind="mapping_csv")
    return load_mapping_csv(paths.mapping_csv)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_schedule_activities(
    paths: ScheduleArtefactPaths,
    *,
    data_date_iso: str | None = None,
) -> dict[str, Any]:
    """Return the schedule's activities + planned %% at the given date.

    Used by the ``ScheduleCompare`` page when no Stage 11 run has been
    produced yet — i.e. the dashboard can show planned %% even before
    actuals exist.
    """
    schedule = _load_schedule(paths)

    if data_date_iso:
        try:
            when = parse_iso_datetime(data_date_iso)
        except ValueError as exc:
            raise ScheduleApiError(
                code="invalid_input",
                message=f"bad data_date_iso: {exc}",
                details={"data_date_iso": data_date_iso},
            ) from exc
    else:
        when = _dt.datetime.now(_dt.timezone.utc)

    activities = []
    for a in schedule.activities:
        activities.append(
            {
                "activity_id": a.activity_id,
                "activity_name": a.activity_name,
                "wbs_code": a.wbs_code,
                "trade": a.trade,
                "location": a.location,
                "predecessors": list(a.predecessors),
                "planned_start_iso": a.planned_start.isoformat(),
                "planned_finish_iso": a.planned_finish.isoformat(),
                "planned_percent_complete": (
                    float(a.percent_complete)
                    if a.percent_complete is not None
                    else a.planned_percent_complete_at(when)
                ),
            }
        )
    return {
        "schema_version": "schedule_activities_response.v1",
        "data_date_utc": when.isoformat(),
        "n_activities": len(activities),
        "activities": activities,
        "schedule_provenance": schedule.provenance.to_dict(),
    }


def get_activity_detail(
    paths: ScheduleArtefactPaths,
    activity_id: str,
    *,
    data_date_iso: str | None = None,
) -> dict[str, Any]:
    """Return planned + actual + variance + risks + mapped elements for one activity.

    Looks up the activity in the schedule CSV and joins it with the
    activity row in ``schedule_variance.json``. If the variance JSON
    is missing the actual fields, the function returns the planned-only
    view (so the dashboard can render the row before Stage 11 runs).
    """
    if not activity_id or not isinstance(activity_id, str):
        raise ScheduleApiError(
            code="invalid_input",
            message="activity_id must be a non-empty string",
        )

    schedule = _load_schedule(paths)
    activity = schedule.get(activity_id)
    if activity is None:
        raise ScheduleApiError(
            code="not_found",
            message="activity not found in schedule",
            details={"activity_id": activity_id},
        )

    if data_date_iso:
        try:
            when = parse_iso_datetime(data_date_iso)
        except ValueError as exc:
            raise ScheduleApiError(
                code="invalid_input",
                message=f"bad data_date_iso: {exc}",
                details={"data_date_iso": data_date_iso},
            ) from exc
    else:
        when = _dt.datetime.now(_dt.timezone.utc)

    payload: dict[str, Any] = {
        "schema_version": "activity_detail_response.v1",
        "activity_id": activity.activity_id,
        "activity_name": activity.activity_name,
        "wbs_code": activity.wbs_code,
        "trade": activity.trade,
        "location": activity.location,
        "predecessors": list(activity.predecessors),
        "planned_start_iso": activity.planned_start.isoformat(),
        "planned_finish_iso": activity.planned_finish.isoformat(),
        "planned_percent_complete": (
            float(activity.percent_complete)
            if activity.percent_complete is not None
            else activity.planned_percent_complete_at(when)
        ),
        "data_date_utc": when.isoformat(),
        "actual": None,
        "mapped_elements": [],
    }

    # Pull the mapped element list when available.
    if paths.mapping_csv.exists():
        mapping = _load_mapping(paths)
        payload["mapped_elements"] = [
            {"ifc_global_id": e.ifc_global_id, "weight": e.weight}
            for e in mapping.elements_for_activity(activity_id)
        ]

    # Pull the variance row when available.
    if paths.schedule_variance_json.exists():
        variance_doc = _read_json(paths.schedule_variance_json, kind="schedule_variance_json")
        for row in variance_doc.get("activities", []):
            if isinstance(row, dict) and row.get("activity_id") == activity_id:
                payload["actual"] = row
                break
    return payload


def get_schedule_variance(paths: ScheduleArtefactPaths) -> dict[str, Any]:
    """Return the run-wide ``schedule_variance.json`` payload as-is."""
    doc = _read_json(paths.schedule_variance_json, kind="schedule_variance_json")
    expected = "schedule_variance.v1"
    if doc.get("schema_version") != expected:
        raise ScheduleApiError(
            code="inconsistent",
            message="schedule_variance.json has unexpected schema_version",
            details={"expected": expected, "actual": doc.get("schema_version")},
        )
    return doc


def get_dashboard_summary(paths: ScheduleArtefactPaths) -> dict[str, Any]:
    """Return the dashboard summary JSON as-is."""
    doc = _read_json(paths.dashboard_summary_json, kind="dashboard_summary_json")
    expected = "dashboard_summary.v1"
    if doc.get("schema_version") != expected:
        raise ScheduleApiError(
            code="inconsistent",
            message="dashboard_summary.json has unexpected schema_version",
            details={"expected": expected, "actual": doc.get("schema_version")},
        )
    return doc


def get_element_status(
    paths: ScheduleArtefactPaths,
    ifc_global_id: str,
) -> dict[str, Any]:
    """Return per-element status from Stage 9 ``element_metrics.csv``.

    Used by the dashboard when the user clicks an element on the 3-D
    BIM viewer. Returns the raw CSV row (with its column names) plus
    the activities that map to this element via the BIM<->schedule
    mapping, when available.
    """
    if not ifc_global_id or not isinstance(ifc_global_id, str):
        raise ScheduleApiError(
            code="invalid_input",
            message="ifc_global_id must be a non-empty string",
        )

    if paths.element_metrics_csv is None or not paths.element_metrics_csv.exists():
        raise ScheduleApiError(
            code="not_found",
            message="element_metrics.csv not available for this run",
            details={"ifc_global_id": ifc_global_id},
        )

    row: dict[str, Any] | None = None
    with paths.element_metrics_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            gid = (r.get("global_id") or r.get("GlobalId") or "").strip()
            if gid == ifc_global_id:
                row = dict(r)
                break

    if row is None:
        raise ScheduleApiError(
            code="not_found",
            message="element not found in element_metrics.csv",
            details={"ifc_global_id": ifc_global_id},
        )

    activities: list[dict[str, Any]] = []
    if paths.mapping_csv.exists():
        mapping = _load_mapping(paths)
        for entry in mapping.activities_for_element(ifc_global_id):
            activities.append(
                {"activity_id": entry.activity_id, "weight": entry.weight}
            )

    return {
        "schema_version": "element_status_response.v1",
        "ifc_global_id": ifc_global_id,
        "element_metrics_row": row,
        "mapped_to_activities": activities,
    }


# ---------------------------------------------------------------------------
# FastAPI router (optional — only constructed when fastapi is importable)
# ---------------------------------------------------------------------------


def create_schedule_router(paths_provider) -> Any:
    """Build a FastAPI ``APIRouter`` exposing the schedule endpoints.

    ``paths_provider`` is a callable taking ``run_id: str | None`` and
    returning the appropriate :class:`ScheduleArtefactPaths`. This
    indirection lets the router serve both 'latest run' and 'specific
    run id' shapes without baking either into this module.

    Raises :class:`RuntimeError` if FastAPI is not installed.
    """
    try:
        from fastapi import APIRouter, HTTPException, Query
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "FastAPI is not installed; create_schedule_router requires it. "
            "Install with `pip install fastapi`."
        ) from exc

    router = APIRouter(prefix="/api/v1", tags=["schedule"])

    def _resolve(run_id: str | None) -> ScheduleArtefactPaths:
        try:
            return paths_provider(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _to_http(exc: ScheduleApiError) -> "HTTPException":
        status = {"not_found": 404, "invalid_input": 400, "inconsistent": 409}.get(exc.code, 500)
        return HTTPException(status_code=status, detail=exc.to_dict())

    @router.get("/schedule/activities")
    def _activities(  # type: ignore[override]
        run_id: str | None = Query(None),
        data_date_iso: str | None = Query(None),
    ):
        try:
            return list_schedule_activities(_resolve(run_id), data_date_iso=data_date_iso)
        except ScheduleApiError as exc:
            raise _to_http(exc) from exc

    @router.get("/schedule/activities/{activity_id}")
    def _activity_detail(  # type: ignore[override]
        activity_id: str,
        run_id: str | None = Query(None),
        data_date_iso: str | None = Query(None),
    ):
        try:
            return get_activity_detail(_resolve(run_id), activity_id, data_date_iso=data_date_iso)
        except ScheduleApiError as exc:
            raise _to_http(exc) from exc

    @router.get("/schedule/variance")
    def _variance(run_id: str | None = Query(None)):  # type: ignore[override]
        try:
            return get_schedule_variance(_resolve(run_id))
        except ScheduleApiError as exc:
            raise _to_http(exc) from exc

    @router.get("/schedule/dashboard")
    def _dashboard(run_id: str | None = Query(None)):  # type: ignore[override]
        try:
            return get_dashboard_summary(_resolve(run_id))
        except ScheduleApiError as exc:
            raise _to_http(exc) from exc

    @router.get("/elements/{ifc_global_id}")
    def _element(  # type: ignore[override]
        ifc_global_id: str,
        run_id: str | None = Query(None),
    ):
        try:
            return get_element_status(_resolve(run_id), ifc_global_id)
        except ScheduleApiError as exc:
            raise _to_http(exc) from exc

    return router
