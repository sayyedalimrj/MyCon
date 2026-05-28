"""FastAPI routers for the service layer.

This is the only module in :mod:`pipeline.service` that imports FastAPI.
Every other module is a plain Python library that can be exercised by
``pytest`` without HTTP.

Endpoints
---------

Health and discovery
~~~~~~~~~~~~~~~~~~~~

- ``GET  /api/health`` — liveness check; also reports broker subscriber
  count and known run count.
- ``GET  /api/registry/stages`` — JSON-serialized stage registry.
- ``GET  /api/registry/vlm-backends`` — JSON-serialized VLM plugin list.
- ``GET  /api/registry/depth-providers`` — JSON-serialized depth plugin
  list.

Configs
~~~~~~~

- ``GET  /api/configs`` — list YAML files under ``configs/``.
- ``GET  /api/configs/{name}`` — load + validate; return resolved data
  and computed config hash.
- ``GET  /api/configs/{name}/schemas/{stage}`` — hydrate the typed
  schema view for one stage; returns the typed dataclass as a JSON dict.

Runs
~~~~

- ``POST /api/runs`` — submit a new run. Body: ``{"config_path": str,
  "requested_stages": [str], "force": bool, "label": str|null}``.
- ``GET  /api/runs`` — list runs (in-memory live + on-disk history merged).
- ``GET  /api/runs/{run_id}`` — full snapshot for one run.
- ``POST /api/runs/{run_id}/cancel`` — issue cancellation.
- ``GET  /api/runs/{run_id}/events`` — replay persisted events as JSON.
- ``WS   /api/runs/{run_id}/events/stream`` — live stream of events.

Artifacts
~~~~~~~~~

- ``GET  /api/runs/{run_id}/artifacts`` — discovered artifact summaries.

Design notes
------------

- All endpoints are read-only except ``POST /api/runs`` and
  ``POST /api/runs/{run_id}/cancel``.
- The router is constructed inside :func:`create_router` so an
  application factory (:func:`pipeline.service.app.create_app`) can mount
  it under a configurable prefix and pass in the broker / executor /
  history / config-discovery dependencies. No module-level state.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette import status as http_status

from pipeline.common.config import ConfigError, PipelineConfig
from pipeline.common.plugins import DEPTH_REGISTRY, VLM_REGISTRY, PluginRegistry
from pipeline.common.provenance import compute_config_hash
from pipeline.common.registry import STAGE_REGISTRY, RegistryError, StageRegistry
from pipeline.common.schema import ConfigSchemaError
from pipeline.service.artifacts import discover_run_artifacts
from pipeline.service.events import EventBroker, RunEventKind
from pipeline.service.executor import ExecutorError, RunExecutor, RunSubmission
from pipeline.service.run_history import RunHistoryStore


LOGGER = logging.getLogger(__name__)


__all__ = ["create_router", "RunSubmissionRequest"]


# ---------------------------------------------------------------------------
# Request / response models.
#
# We use Pydantic v2 BaseModel for the small set of POST bodies; the
# response side returns plain dicts produced by the foundation layer's
# ``to_dict()`` methods so we avoid double-modeling shapes that are
# already typed.
# ---------------------------------------------------------------------------


class RunSubmissionRequest(BaseModel):
    config_path: str = Field(..., description="Path to YAML config (absolute or repo-relative)")
    requested_stages: list[str] = Field(..., description="Subset of registered stage names to execute")
    force: bool = Field(False, description="Pass --force to each stage CLI")
    label: str | None = Field(None, description="Optional run label suffix; UUID8 used otherwise")


# ---------------------------------------------------------------------------
# Router factory.
# ---------------------------------------------------------------------------


def create_router(
    *,
    broker: EventBroker,
    executor: RunExecutor,
    history: RunHistoryStore,
    project_root: Path,
    configs_root: Path,
    registry: StageRegistry | None = None,
    vlm_registry: PluginRegistry | None = None,
    depth_registry: PluginRegistry | None = None,
    config_loader=None,
) -> APIRouter:
    """Build the FastAPI router. Caller owns the dependency lifetimes."""
    reg = registry or STAGE_REGISTRY
    vlm_reg = vlm_registry or VLM_REGISTRY
    depth_reg = depth_registry or DEPTH_REGISTRY
    if config_loader is None:
        from pipeline.common.config import load_config
        config_loader = load_config

    router = APIRouter(prefix="/api")

    # ---- health ------------------------------------------------------ #

    @router.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "subscriber_count": broker.subscriber_count(),
            "tracked_run_ids": list(broker.all_run_ids()),
            "history_run_count": len(history.list()),
            "stage_count": len(reg),
        }

    # ---- registry / plugin discovery -------------------------------- #

    @router.get("/registry/stages")
    def list_stages() -> list[dict[str, Any]]:
        return reg.to_dict()

    @router.get("/registry/stages/{stage_name}")
    def get_stage(stage_name: str) -> dict[str, Any]:
        try:
            descriptor = reg.get(stage_name)
        except RegistryError as exc:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=str(exc))
        return descriptor.to_dict()

    @router.get("/registry/vlm-backends")
    def list_vlm_backends() -> list[dict[str, Any]]:
        return vlm_reg.to_dict()

    @router.get("/registry/depth-providers")
    def list_depth_providers() -> list[dict[str, Any]]:
        return depth_reg.to_dict()

    # ---- configs ----------------------------------------------------- #

    @router.get("/configs")
    def list_configs() -> list[dict[str, Any]]:
        if not configs_root.exists():
            return []
        out: list[dict[str, Any]] = []
        for path in sorted(configs_root.glob("*.yaml")):
            try:
                stat = path.stat()
                out.append({
                    "name": path.stem,
                    "path": str(path),
                    "size_bytes": stat.st_size,
                    "modified_at_unix": stat.st_mtime,
                })
            except OSError:
                continue
        return out

    @router.get("/configs/{name}")
    def get_config(name: str) -> dict[str, Any]:
        path = _resolve_config_path(configs_root, name)
        if path is None:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Unknown config: {name!r}")
        try:
            cfg = config_loader(path)
        except (ConfigError, OSError) as exc:
            raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(exc))
        return {
            "name": name,
            "path": str(path),
            "config_hash": compute_config_hash(cfg),
            "data": cfg.data,
        }

    @router.get("/configs/{name}/schemas/{stage_name}")
    def get_stage_schema(name: str, stage_name: str) -> dict[str, Any]:
        path = _resolve_config_path(configs_root, name)
        if path is None:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Unknown config: {name!r}")
        try:
            descriptor = reg.get(stage_name)
        except RegistryError as exc:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=str(exc))
        try:
            cfg = config_loader(path)
        except (ConfigError, OSError) as exc:
            raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(exc))
        try:
            schema = descriptor.schema_class.from_config(cfg)
        except (ConfigSchemaError, ConfigError) as exc:
            raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(exc))
        return {
            "config_name": name,
            "stage": stage_name,
            "required_config_keys": list(descriptor.required_config_keys()),
            "schema_class": descriptor.schema_class.__name__,
            "schema": _dataclass_to_json(schema),
        }

    # ---- runs -------------------------------------------------------- #

    @router.post("/runs", status_code=http_status.HTTP_201_CREATED)
    def submit_run(body: RunSubmissionRequest) -> dict[str, Any]:
        cfg_path_raw = Path(body.config_path).expanduser()
        if not cfg_path_raw.is_absolute():
            cfg_path = (project_root / cfg_path_raw).resolve()
        else:
            cfg_path = cfg_path_raw.resolve()
        if not cfg_path.exists():
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=f"config_path does not exist: {cfg_path}",
            )

        submission = RunSubmission(
            config_path=cfg_path,
            requested_stages=tuple(body.requested_stages),
            force=body.force,
            label=body.label,
        )
        try:
            handle = executor.submit(submission)
        except (ExecutorError, ConfigError) as exc:
            raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(exc))

        snap = executor.snapshot(handle.run_id)
        return {"run_id": handle.run_id, "snapshot": snap}

    @router.get("/runs")
    def list_runs(limit: int = Query(100, ge=1, le=500)) -> list[dict[str, Any]]:
        return [rec.to_dict() for rec in history.list(limit=limit)]

    @router.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        snap = executor.snapshot(run_id)
        if snap is None:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Unknown run: {run_id!r}")
        return dict(snap)

    @router.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
        ack = executor.cancel(run_id)
        if not ack:
            # Either the run is unknown, or already finished/cancelled.
            snap = executor.snapshot(run_id)
            if snap is None:
                raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Unknown run: {run_id!r}")
            return {"cancel_requested": False, "reason": "run is not active", "status": snap.get("status")}
        return {"cancel_requested": True}

    @router.get("/runs/{run_id}/events")
    def replay_events(run_id: str) -> list[dict[str, Any]]:
        snap = executor.snapshot(run_id)
        if snap is None:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Unknown run: {run_id!r}")
        # Prefer in-memory replay (richer payload retention); fall back to disk.
        in_memory = broker.replay(run_id)
        if in_memory:
            return [ev.to_dict() for ev in in_memory]
        return history.read_events(run_id)

    @router.websocket("/runs/{run_id}/events/stream")
    async def stream_events(websocket: WebSocket, run_id: str) -> None:
        # Accept first; otherwise the client sees a close before the
        # subprotocol round-trip and treats the URL as broken.
        await websocket.accept()
        snap = executor.snapshot(run_id)
        if snap is None:
            await websocket.send_json({"error": "unknown_run", "run_id": run_id})
            await websocket.close()
            return

        loop = asyncio.get_running_loop()
        subscription = broker.subscribe(loop, run_id=run_id)

        # Replay any buffered events first so a late-joining client sees
        # the full history of the run before live events resume.
        for ev in broker.replay(run_id):
            try:
                await websocket.send_json(ev.to_dict())
            except Exception:
                broker.unsubscribe(subscription.subscription_id)
                return

        try:
            while True:
                ev = await subscription.queue.get()
                await websocket.send_json(ev.to_dict())
                # Hard close once the run reaches a terminal kind so
                # clients don't hang forever after the last event.
                if ev.kind in (
                    RunEventKind.RUN_FINISHED,
                    RunEventKind.RUN_FAILED,
                    RunEventKind.RUN_CANCELLED,
                ):
                    break
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # pragma: no cover - generic socket failure
            LOGGER.warning("stream_events: socket failure: %s", exc)
        finally:
            broker.unsubscribe(subscription.subscription_id)
            try:
                await websocket.close()
            except Exception:  # pragma: no cover
                pass

    # ---- artifacts --------------------------------------------------- #

    @router.get("/runs/{run_id}/artifacts")
    def list_artifacts(run_id: str) -> list[dict[str, Any]]:
        # Existence of the run id is verified via the snapshot; we want
        # the GUI to be able to ask about a finished run that the
        # in-memory executor has already forgotten, so the snapshot may
        # come from history. If it isn't found there either, 404.
        snap = executor.snapshot(run_id)
        if snap is None:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Unknown run: {run_id!r}")
        summaries = discover_run_artifacts(run_id, project_root=project_root, registry=reg)
        return [s.to_dict() for s in summaries]

    return router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_config_path(configs_root: Path, name: str) -> Path | None:
    """Map ``{name}`` URL segment to an actual config file under ``configs_root``.

    Accepts either ``site01`` or ``site01.yaml``. Strips any leading
    components so a malicious ``..`` cannot escape the configs directory.
    """
    safe = Path(name).name
    if not safe.endswith(".yaml") and not safe.endswith(".yml"):
        safe = f"{safe}.yaml"
    candidate = configs_root / safe
    if not candidate.exists():
        return None
    # One last sanity check: the resolved path must be inside configs_root.
    try:
        candidate.resolve().relative_to(configs_root.resolve())
    except ValueError:
        return None
    return candidate


def _dataclass_to_json(obj: Any) -> Any:
    """Recursively convert dataclasses (incl. nested) to JSON-friendly dicts."""
    if is_dataclass(obj):
        return {k: _dataclass_to_json(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): _dataclass_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
