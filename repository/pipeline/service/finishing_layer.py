"""Convenience umbrella for the Phase 4/5 finishing-layer routers.

The Schedule Compare dashboard consumes three groups of endpoints:

- ``/api/v1/schedule/...``       — :mod:`pipeline.service.schedule_api`
- ``/api/v1/hitl/...``           — :mod:`pipeline.service.hitl_api`
- ``/api/v1/calibration/...``    — :mod:`pipeline.service.calibration_api`

Every consumer that wants to mount the finishing-layer routes against a
FastAPI app today has to import three router constructors and three
artefact-paths classes. This module exposes one helper so they can do
it in one call::

    from fastapi import FastAPI
    from pipeline.service.finishing_layer import register_finishing_layer

    app = FastAPI()
    register_finishing_layer(app, run_resolver=lambda run_id: ...)

The ``run_resolver`` callable takes ``run_id: str | None`` and returns
the run directory whose artefacts the routes should serve. The umbrella
threads the same callable into all three routers so the dashboard sees
a consistent run context.

This module is intentionally **dependency-free at import time**: it
only imports FastAPI when ``register_finishing_layer`` is actually
called. The artefact-paths classes are imported eagerly because they
are pure stdlib + stdlib only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pipeline.service.calibration_api import (
    CalibrationArtefactPaths,
    create_calibration_router,
)
from pipeline.service.hitl_api import (
    HitlArtefactPaths,
    create_hitl_router,
)
from pipeline.service.schedule_api import (
    ScheduleArtefactPaths,
    create_schedule_router,
)


__all__ = [
    "RunResolver",
    "register_finishing_layer",
    "build_finishing_layer_paths_providers",
]


# Type alias for the run-id-to-run-dir resolver. Returning None or
# raising FileNotFoundError signals "unknown run"; the FastAPI routers
# translate that to HTTP 404.
RunResolver = Callable[[str | None], Path]


def build_finishing_layer_paths_providers(
    run_resolver: RunResolver,
) -> tuple[
    Callable[[str | None], ScheduleArtefactPaths],
    Callable[[str | None], HitlArtefactPaths],
    Callable[[str | None], CalibrationArtefactPaths],
]:
    """Lift one ``run_id -> Path`` resolver into three typed paths providers.

    Each returned callable is the ``paths_provider`` that the matching
    ``create_*_router`` consumer expects. Useful when an integration
    test wants the providers without instantiating FastAPI.
    """

    def _schedule_provider(run_id: str | None) -> ScheduleArtefactPaths:
        return ScheduleArtefactPaths.under_run_dir(run_resolver(run_id))

    def _hitl_provider(run_id: str | None) -> HitlArtefactPaths:
        return HitlArtefactPaths.under_run_dir(run_resolver(run_id))

    def _calibration_provider(run_id: str | None) -> CalibrationArtefactPaths:
        return CalibrationArtefactPaths.under_run_dir(run_resolver(run_id))

    return _schedule_provider, _hitl_provider, _calibration_provider


def register_finishing_layer(
    app: Any,
    *,
    run_resolver: RunResolver,
) -> tuple[Any, Any, Any]:
    """Register the three Phase 4/5 routers on ``app``.

    Returns ``(schedule_router, hitl_router, calibration_router)`` so
    callers can introspect the registered routes (e.g. for OpenAPI
    discovery in tests). Raises :class:`RuntimeError` if FastAPI is
    not installed -- mirrors the per-router behaviour.

    Behaviour
    ---------
    - All three routers are mounted with their default ``/api/v1``
      prefix (set by the per-module ``create_*_router`` constructors).
    - ``run_resolver`` is shared across all three so a request that
      includes ``?run_id=X`` reaches the same run directory in every
      route.
    - The function is idempotent only with respect to its return
      value: calling it twice on the same ``app`` will register the
      routers twice. Callers that want idempotence should track that
      themselves.
    """
    schedule_provider, hitl_provider, calibration_provider = (
        build_finishing_layer_paths_providers(run_resolver)
    )
    schedule_router = create_schedule_router(schedule_provider)
    hitl_router = create_hitl_router(hitl_provider)
    calibration_router = create_calibration_router(calibration_provider)
    app.include_router(schedule_router)
    app.include_router(hitl_router)
    app.include_router(calibration_router)
    return schedule_router, hitl_router, calibration_router
