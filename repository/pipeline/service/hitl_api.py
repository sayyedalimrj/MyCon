"""HITL corrections endpoints for the Schedule Compare dashboard.

Allows a reviewer to submit a correction directly from the drilldown
panel, which is the workflow described in
``docs/end_to_end_finishing_plan.md`` §6 row 6 and ``docs/hitl_workflow.md``
§2.2.

The module mirrors the design of :mod:`pipeline.service.schedule_api`:
the **business logic** lives in plain Python functions, and the FastAPI
router in :func:`create_hitl_router` is a thin wrapper. This means
every endpoint is unit-testable without spinning up HTTP and FastAPI
remains an optional dependency.

Public functions
----------------

- :func:`submit_correction`   ``POST /api/v1/hitl/corrections``
- :func:`list_corrections`    ``GET  /api/v1/hitl/corrections``

Errors
------

Both functions raise :class:`HitlApiError` with one of:

- ``invalid_input``  — payload missing/invalid (caught by Pydantic-free
                        :class:`pipeline.common.hitl.Correction.from_dict`).
- ``not_found``       — when the run dir / corrections JSONL is absent
                        and the caller asked for read.
- ``persistence_failed`` — disk write failed (rare; surfaces as HTTP 500).

The router maps these to HTTP 400 / 404 / 500 respectively.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pipeline.common.hitl import (
    CORRECTION_SCHEMA_VERSION,
    Correction,
    CorrectionStore,
)


__all__ = [
    "HITL_LOG_BASENAME",
    "HitlApiError",
    "HitlArtefactPaths",
    "submit_correction",
    "list_corrections",
    "create_hitl_router",
]


HITL_LOG_BASENAME = "hitl_corrections.jsonl"


@dataclass(frozen=True)
class HitlApiError(Exception):
    """Structured error type the router translates to HTTP."""

    code: str  # 'invalid_input' / 'not_found' / 'persistence_failed'
    message: str
    details: Mapping[str, Any] | None = None

    def __str__(self) -> str:  # pragma: no cover - dataclass delegate
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
class HitlArtefactPaths:
    """Bundle of HITL artefact paths the API reads/writes.

    Carrying this as a single object makes the API trivially mockable
    in tests and keeps the router signature small.
    """

    corrections_jsonl: Path

    @classmethod
    def under_run_dir(cls, run_dir: Path | str) -> "HitlArtefactPaths":
        """Construct paths assuming the canonical layout under a run dir.

        Mirrors :meth:`pipeline.service.schedule_api.ScheduleArtefactPaths.under_run_dir`
        for shape consistency.
        """
        run_dir = Path(run_dir).resolve()
        return cls(corrections_jsonl=run_dir / "reports" / HITL_LOG_BASENAME)


# ---------------------------------------------------------------------------
# Public read/write API
# ---------------------------------------------------------------------------


def submit_correction(
    paths: HitlArtefactPaths,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate ``payload`` and append it to the run's corrections JSONL.

    Returns the canonical correction (with auto-filled ``record_id``
    and ``timestamp_utc`` if absent) as a dict. The parent directory of
    the corrections JSONL is created if it does not already exist.
    """
    if not isinstance(payload, Mapping):
        raise HitlApiError(
            code="invalid_input",
            message="payload must be a JSON object",
            details={"got_type": type(payload).__name__},
        )

    try:
        store = CorrectionStore(paths.corrections_jsonl)
    except Exception as exc:  # pragma: no cover - guard
        raise HitlApiError(
            code="persistence_failed",
            message=f"could not open corrections store: {exc}",
            details={"path": str(paths.corrections_jsonl)},
        ) from exc

    try:
        record = store.append(payload)
    except ValueError as exc:
        raise HitlApiError(
            code="invalid_input",
            message=str(exc),
            details={"payload_keys": sorted(list(payload.keys()))},
        ) from exc
    except OSError as exc:
        raise HitlApiError(
            code="persistence_failed",
            message=f"failed to write correction: {exc}",
            details={"path": str(paths.corrections_jsonl)},
        ) from exc

    return {
        "schema_version": "hitl_submit_response.v1",
        "stored_path": str(paths.corrections_jsonl.resolve()),
        "correction": record.to_dict(),
    }


def list_corrections(
    paths: HitlArtefactPaths,
    *,
    target_kinds: list[str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Return a replay of the run's corrections JSONL.

    Filtering arguments mirror :meth:`pipeline.common.hitl.CorrectionStore.replay`.
    Missing JSONL is treated as empty (an unsubmitted run is a real
    state the dashboard should render gracefully) rather than raising.
    """
    store = CorrectionStore(paths.corrections_jsonl)
    replay = store.replay(target_kinds=target_kinds, run_id=run_id)
    return {
        "schema_version": "hitl_list_response.v1",
        "stored_path": str(paths.corrections_jsonl.resolve()),
        "schema_version_record": CORRECTION_SCHEMA_VERSION,
        "n_total_records": replay.n_total_records,
        "n_effective": len(replay.effective),
        "n_conflicts": len(replay.conflicts),
        "effective": [c.to_dict() for c in replay.effective],
        "conflicts": [c.to_dict() for c in replay.conflicts],
    }


# ---------------------------------------------------------------------------
# FastAPI router (optional)
# ---------------------------------------------------------------------------


def create_hitl_router(paths_provider) -> Any:
    """Build a FastAPI ``APIRouter`` exposing the HITL endpoints.

    ``paths_provider`` is a callable taking ``run_id: str | None`` and
    returning the appropriate :class:`HitlArtefactPaths`. This indirection
    matches :func:`pipeline.service.schedule_api.create_schedule_router`
    and lets the router serve both 'latest run' and 'specific run id'
    shapes without baking either into this module.

    Raises :class:`RuntimeError` if FastAPI is not installed.
    """
    try:
        from fastapi import APIRouter, Body, HTTPException, Query
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "FastAPI is not installed; create_hitl_router requires it. "
            "Install with `pip install fastapi`."
        ) from exc

    router = APIRouter(prefix="/api/v1", tags=["hitl"])

    def _resolve(run_id: str | None) -> HitlArtefactPaths:
        try:
            return paths_provider(run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _to_http(exc: HitlApiError) -> "HTTPException":
        status = {
            "invalid_input": 400,
            "not_found": 404,
            "persistence_failed": 500,
        }.get(exc.code, 500)
        return HTTPException(status_code=status, detail=exc.to_dict())

    @router.post("/hitl/corrections")
    def _submit(  # type: ignore[override]
        payload: dict = Body(...),
        run_id: str | None = Query(None),
    ):
        try:
            return submit_correction(_resolve(run_id), payload)
        except HitlApiError as exc:
            raise _to_http(exc) from exc

    @router.get("/hitl/corrections")
    def _list(  # type: ignore[override]
        run_id: str | None = Query(None),
        target_kinds: str | None = Query(
            None,
            description="Comma-separated target_kind filter; e.g. 'element_acceptance,activity_completion'.",
        ),
    ):
        kinds = (
            [t.strip() for t in target_kinds.split(",") if t.strip()]
            if target_kinds
            else None
        )
        try:
            return list_corrections(_resolve(run_id), target_kinds=kinds, run_id=run_id)
        except HitlApiError as exc:
            raise _to_http(exc) from exc

    return router
