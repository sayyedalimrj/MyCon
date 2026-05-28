"""Calibration endpoints for the Schedule Compare dashboard.

Closes the loop between the HITL corrections store
(:mod:`pipeline.common.hitl`) and the calibration report
(:mod:`pipeline.common.calibration`): a one-click "Replay" action on
the dashboard re-reads the run's HITL log, re-derives the calibration
records, and writes a fresh ``calibration_report.v1`` JSON next to the
log. The dashboard's :class:`ReliabilityCard` then refetches and shows
the updated ECE / Brier / smooth-ECE.

Mirrors the design of :mod:`pipeline.service.schedule_api` and
:mod:`pipeline.service.hitl_api`: pure read/write functions plus a thin
FastAPI router so the contract is unit-testable without HTTP and
FastAPI stays an optional dependency.

Public functions
----------------

- :func:`run_calibration_report`   ``POST /api/v1/calibration/run``
- :func:`get_latest_report`        ``GET  /api/v1/calibration/report``

Errors
------

Both functions raise :class:`CalibrationApiError` with one of:

- ``not_found``           — HITL log absent (replay) or report absent (read)
- ``invalid_input``       — bad ``n_bins`` / ``strategy`` / ``target_kinds``
- ``persistence_failed``  — disk write failed (rare; HTTP 500)

The router maps these to HTTP 404 / 400 / 500 respectively.

Schema versions locked:

- ``calibration_run_response.v1`` — payload returned from the replay endpoint
  (wraps the standard ``calibration_report.v1``)
- ``calibration_report.v1`` — the report itself
  (from :func:`pipeline.common.calibration.calibration_report`)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipeline.common import calibration
from pipeline.common.hitl import CorrectionStore, build_calibration_records


__all__ = [
    "CALIBRATION_REPORT_BASENAME",
    "DEFAULT_TARGET_KINDS",
    "CalibrationApiError",
    "CalibrationArtefactPaths",
    "run_calibration_report",
    "get_latest_report",
    "create_calibration_router",
]


CALIBRATION_REPORT_BASENAME = "calibration_report.json"

# By default we calibrate the per-element acceptance decision; this is
# the dominant correction kind on the Schedule Compare dashboard. A
# caller can override via ``target_kinds=...``. We intentionally do NOT
# default to "all kinds" because mixing element-level and activity-level
# operating points produces a misleading aggregate ECE
# (cf. docs/calibration_workflow.md §9).
DEFAULT_TARGET_KINDS: tuple[str, ...] = ("element_acceptance",)


@dataclass(frozen=True)
class CalibrationApiError(Exception):
    """Structured error type the router translates to HTTP."""

    code: str
    message: str
    details: Mapping[str, Any] | None = None

    def __str__(self) -> str:  # pragma: no cover
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
class CalibrationArtefactPaths:
    """Bundle of paths the calibration API reads/writes.

    Carrying this as a single object keeps the router signature small
    and matches :class:`pipeline.service.schedule_api.ScheduleArtefactPaths`
    and :class:`pipeline.service.hitl_api.HitlArtefactPaths`.
    """

    corrections_jsonl: Path
    report_json: Path

    @classmethod
    def under_run_dir(cls, run_dir: Path | str) -> "CalibrationArtefactPaths":
        """Construct paths assuming the canonical run-dir layout.

        ``runs/<run_id>/reports/hitl_corrections.jsonl`` and
        ``runs/<run_id>/reports/calibration_report.json``.
        """
        run_dir = Path(run_dir).resolve()
        reports = run_dir / "reports"
        return cls(
            corrections_jsonl=reports / "hitl_corrections.jsonl",
            report_json=reports / CALIBRATION_REPORT_BASENAME,
        )


# ---------------------------------------------------------------------------
# Public read/write API
# ---------------------------------------------------------------------------


_VALID_STRATEGIES = frozenset({"equal_mass", "equal_width"})


def _validate_args(
    n_bins: int,
    strategy: str,
    target_kinds: Sequence[str] | None,
) -> tuple[int, str, tuple[str, ...] | None]:
    if not isinstance(n_bins, int) or n_bins < 1:
        raise CalibrationApiError(
            code="invalid_input",
            message="n_bins must be a positive integer",
            details={"n_bins": n_bins},
        )
    if strategy not in _VALID_STRATEGIES:
        raise CalibrationApiError(
            code="invalid_input",
            message=f"strategy must be one of {sorted(_VALID_STRATEGIES)}",
            details={"strategy": strategy},
        )
    kinds: tuple[str, ...] | None
    if target_kinds is None:
        kinds = None
    else:
        coerced: list[str] = []
        for k in target_kinds:
            if not isinstance(k, str) or not k.strip():
                raise CalibrationApiError(
                    code="invalid_input",
                    message="target_kinds entries must be non-empty strings",
                    details={"target_kinds": list(target_kinds)},
                )
            coerced.append(k.strip())
        kinds = tuple(coerced) if coerced else None
    return n_bins, strategy, kinds


def run_calibration_report(
    paths: CalibrationArtefactPaths,
    *,
    n_bins: int = 10,
    strategy: str = "equal_mass",
    target_kinds: Sequence[str] | None = DEFAULT_TARGET_KINDS,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Replay the HITL log and (re)produce ``calibration_report.json``.

    Returns a payload of shape::

        {
          "schema_version": "calibration_run_response.v1",
          "stored_path": "<absolute path of the report JSON>",
          "n_replayed_records": <int>,
          "n_effective_records": <int>,
          "n_conflicts": <int>,
          "report": <calibration_report.v1>,
        }

    Defaults are deliberately conservative: ``target_kinds = ("element_acceptance",)``
    so the reported ECE measures one operating point at a time. Set
    ``target_kinds=None`` to calibrate across all kinds (rarely the
    right call; see ``docs/calibration_workflow.md`` §9).

    The replay is **idempotent** with respect to the HITL log: the same
    log produces the same report.
    """
    n_bins, strategy, kinds = _validate_args(n_bins, strategy, target_kinds)

    if not paths.corrections_jsonl.exists():
        raise CalibrationApiError(
            code="not_found",
            message="HITL corrections log not found; nothing to calibrate",
            details={"corrections_jsonl": str(paths.corrections_jsonl)},
        )

    store = CorrectionStore(paths.corrections_jsonl)
    replay = store.replay(target_kinds=list(kinds) if kinds is not None else None, run_id=run_id)
    cal_records = build_calibration_records(
        replay,
        target_kinds=list(kinds) if kinds is not None else None,
    )
    report = calibration.calibration_report(
        cal_records,
        n_bins=n_bins,
        strategy=strategy,
    )
    # Self-attached provenance so the saved file is auditable.
    report["calibration_run_provenance"] = {
        "schema_version": "calibration_run_provenance.v1",
        "corrections_jsonl": str(paths.corrections_jsonl.resolve()),
        "n_replayed_records": replay.n_total_records,
        "n_effective_records": len(replay.effective),
        "n_conflicts": len(replay.conflicts),
        "target_kinds": list(kinds) if kinds is not None else None,
        "filter_run_id": run_id,
    }

    try:
        paths.report_json.parent.mkdir(parents=True, exist_ok=True)
        paths.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as exc:
        raise CalibrationApiError(
            code="persistence_failed",
            message=f"failed to write calibration report: {exc}",
            details={"report_json": str(paths.report_json)},
        ) from exc

    return {
        "schema_version": "calibration_run_response.v1",
        "stored_path": str(paths.report_json.resolve()),
        "n_replayed_records": replay.n_total_records,
        "n_effective_records": len(replay.effective),
        "n_conflicts": len(replay.conflicts),
        "report": report,
    }


def get_latest_report(paths: CalibrationArtefactPaths) -> dict[str, Any]:
    """Return the persisted ``calibration_report.json`` as a dict.

    Raises ``not_found`` if the report file does not exist (the
    dashboard treats that as the "no calibration yet" empty state and
    the route maps it to HTTP 404).
    """
    if not paths.report_json.exists():
        raise CalibrationApiError(
            code="not_found",
            message="no calibration report yet for this run",
            details={"report_json": str(paths.report_json)},
        )
    try:
        report = json.loads(paths.report_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationApiError(
            code="persistence_failed",
            message=f"calibration report on disk is not valid JSON: {exc}",
            details={"report_json": str(paths.report_json)},
        ) from exc
    expected = "calibration_report.v1"
    if report.get("schema_version") != expected:
        raise CalibrationApiError(
            code="persistence_failed",
            message="calibration report on disk has unexpected schema_version",
            details={
                "expected": expected,
                "actual": report.get("schema_version"),
                "report_json": str(paths.report_json),
            },
        )
    return report


# ---------------------------------------------------------------------------
# FastAPI router (optional)
# ---------------------------------------------------------------------------


def create_calibration_router(paths_provider) -> Any:
    """Build a FastAPI ``APIRouter`` exposing the calibration endpoints.

    ``paths_provider(run_id) -> CalibrationArtefactPaths`` decouples the
    router from any specific run-discovery policy, mirroring
    :func:`pipeline.service.hitl_api.create_hitl_router`.

    Raises :class:`RuntimeError` if FastAPI is not installed.
    """
    try:
        from fastapi import APIRouter, Body, HTTPException, Query
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "FastAPI is not installed; create_calibration_router requires it. "
            "Install with `pip install fastapi`."
        ) from exc

    router = APIRouter(prefix="/api/v1", tags=["calibration"])

    def _resolve(run_id: str | None) -> CalibrationArtefactPaths:
        try:
            return paths_provider(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _to_http(exc: CalibrationApiError) -> "HTTPException":
        status = {
            "invalid_input": 400,
            "not_found": 404,
            "persistence_failed": 500,
        }.get(exc.code, 500)
        return HTTPException(status_code=status, detail=exc.to_dict())

    @router.post("/calibration/run")
    def _run(  # type: ignore[override]
        payload: dict | None = Body(default=None),
        run_id: str | None = Query(None),
    ):
        body = dict(payload or {})
        target_kinds_raw = body.get("target_kinds", DEFAULT_TARGET_KINDS)
        if isinstance(target_kinds_raw, str):
            target_kinds: list[str] | None = [t.strip() for t in target_kinds_raw.split(",") if t.strip()]
        elif isinstance(target_kinds_raw, list):
            target_kinds = [str(t) for t in target_kinds_raw]
        elif target_kinds_raw is None:
            target_kinds = None
        else:
            raise _to_http(
                CalibrationApiError(
                    code="invalid_input",
                    message="target_kinds must be a list, string, or null",
                )
            )
        try:
            return run_calibration_report(
                _resolve(run_id),
                n_bins=int(body.get("n_bins", 10)),
                strategy=str(body.get("strategy", "equal_mass")),
                target_kinds=target_kinds,
                run_id=run_id,
            )
        except CalibrationApiError as exc:
            raise _to_http(exc) from exc

    @router.get("/calibration/report")
    def _get(run_id: str | None = Query(None)):  # type: ignore[override]
        try:
            return get_latest_report(_resolve(run_id))
        except CalibrationApiError as exc:
            raise _to_http(exc) from exc

    return router
