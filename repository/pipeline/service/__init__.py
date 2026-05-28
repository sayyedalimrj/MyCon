"""HTTP / WebSocket service layer.

This subpackage is *optional*. Operators who only run stages from the CLI do
not need ``fastapi`` or ``uvicorn`` installed. The dependencies live in
``requirements-service.txt``; the module documents that fact when it cannot
be imported.

The service is intentionally a thin layer over the foundations introduced
in Phase 1 (:mod:`pipeline.common.schema`, :mod:`pipeline.common.registry`,
:mod:`pipeline.common.provenance`, :mod:`pipeline.common.plugins`). It does
not implement any new pipeline behavior. It exposes:

- The stage registry and schema views as REST endpoints, so a future GUI
  can drive form generation from JSON.
- A run-executor that invokes stages **via the canonical subprocess CLI**,
  not in-process, so Phase 1's "no behavior change" invariant carries
  through Phase 2.
- A WebSocket stream of per-stage status events for live progress display.

Imports here are lazy: importing :mod:`pipeline.service` does not import
FastAPI. Use :func:`create_app` to obtain the FastAPI application object.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from fastapi import FastAPI


def create_app(*args: Any, **kwargs: Any) -> "FastAPI":
    """Lazy entrypoint that defers FastAPI import until first call.

    Importing :mod:`pipeline.service` itself never imports FastAPI; tests
    that only need the registry / executor pieces can avoid pulling in the
    HTTP framework.
    """
    return import_module("pipeline.service.app").create_app(*args, **kwargs)


__all__ = ["create_app"]
