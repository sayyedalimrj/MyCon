"""FastAPI application factory.

This module's only job is to bind the foundation pieces
(:class:`EventBroker`, :class:`RunHistoryStore`, :class:`RunExecutor`)
into a FastAPI app, mount the API router, and configure CORS for the
future GUI frontend.

Operators run the service via ``uvicorn``::

    uvicorn pipeline.service.app:make_default_app --factory --host 127.0.0.1 --port 8765

For tests, :func:`create_app` accepts an explicit broker/history/executor
trio so the test suite never spawns real subprocesses unless it wants to.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pipeline.service.api import create_router
from pipeline.service.events import EventBroker
from pipeline.service.executor import RunExecutor
from pipeline.service.run_history import RunHistoryStore


LOGGER = logging.getLogger(__name__)


__all__ = ["create_app", "make_default_app", "AppDependencies"]


# Frontend dev servers commonly run on ports 3000/5173. The default CORS
# allowlist accepts those and localhost. Operators running behind a
# reverse proxy can override via ``CORS_ALLOWED_ORIGINS`` env var when
# they wire the app into uvicorn.
_DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


class AppDependencies:
    """Container holding the long-lived service objects.

    Bundling them lets callers (tests, ``make_default_app``) construct
    one set of dependencies and pass it whole to :func:`create_app`,
    instead of repeating the wiring at every entry point.
    """

    __slots__ = ("broker", "history", "executor", "project_root", "configs_root")

    def __init__(
        self,
        *,
        broker: EventBroker,
        history: RunHistoryStore,
        executor: RunExecutor,
        project_root: Path,
        configs_root: Path,
    ) -> None:
        self.broker = broker
        self.history = history
        self.executor = executor
        self.project_root = project_root
        self.configs_root = configs_root


def create_app(
    deps: AppDependencies,
    *,
    cors_origins: Iterable[str] | None = None,
    title: str = "MyCon Pipeline Service",
) -> FastAPI:
    """Build the FastAPI app bound to ``deps``."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        LOGGER.info("MyCon service starting; project_root=%s", deps.project_root)
        yield
        LOGGER.info("MyCon service shutting down")
        deps.executor.shutdown(wait=True, cancel_running=True)

    app = FastAPI(title=title, lifespan=lifespan)

    origins = list(cors_origins) if cors_origins is not None else list(_DEFAULT_CORS_ORIGINS)
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    router = create_router(
        broker=deps.broker,
        executor=deps.executor,
        history=deps.history,
        project_root=deps.project_root,
        configs_root=deps.configs_root,
    )
    app.include_router(router)

    return app


def make_default_app() -> FastAPI:
    """Construct an app with default dependencies for ``uvicorn --factory``.

    Resolves the repository root from this file's location, places the
    service's persistence under ``runs/_service/``, and uses the canonical
    stage / VLM / depth registries.
    """
    repo_root = Path(__file__).resolve().parents[2]
    configs_root = repo_root / "configs"
    service_root = repo_root / "runs" / "_service"
    service_root.mkdir(parents=True, exist_ok=True)

    broker = EventBroker()
    history = RunHistoryStore(service_root)
    executor = RunExecutor(broker, history, repo_root=repo_root)
    deps = AppDependencies(
        broker=broker,
        history=history,
        executor=executor,
        project_root=repo_root,
        configs_root=configs_root,
    )
    return create_app(deps)
