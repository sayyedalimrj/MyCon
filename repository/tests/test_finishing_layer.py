"""Tests for :mod:`pipeline.service.finishing_layer`.

The umbrella module exposes:

- :func:`build_finishing_layer_paths_providers`: pure-stdlib helper
  that lifts one ``run_id -> Path`` resolver into the three typed
  paths providers expected by the schedule / HITL / calibration
  routers. We test this directly because it does not require FastAPI.
- :func:`register_finishing_layer`: thin wrapper over the three
  ``create_*_router`` constructors. We do not exercise it here
  because it imports FastAPI through the per-router constructors;
  that path is covered transitively by the per-router tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.service.calibration_api import CalibrationArtefactPaths
from pipeline.service.finishing_layer import build_finishing_layer_paths_providers
from pipeline.service.hitl_api import HitlArtefactPaths
from pipeline.service.schedule_api import ScheduleArtefactPaths


pytestmark = pytest.mark.lightweight


def test_providers_resolve_to_canonical_layouts(tmp_path: Path) -> None:
    """One ``run_id`` resolver -> three canonically-shaped paths objects."""

    def _resolver(run_id):  # type: ignore[no-untyped-def]
        # Both run_id=None and run_id='X' map to the same dir under
        # tmp_path so the test stays deterministic.
        return tmp_path

    sched_p, hitl_p, cal_p = build_finishing_layer_paths_providers(_resolver)
    sched = sched_p(None)
    hitl = hitl_p(None)
    cal = cal_p(None)
    # Every path is anchored under the resolved run dir.
    assert isinstance(sched, ScheduleArtefactPaths)
    assert isinstance(hitl, HitlArtefactPaths)
    assert isinstance(cal, CalibrationArtefactPaths)
    expected_reports = (tmp_path / "reports").resolve()
    assert sched.activity_progress_json.parent == expected_reports
    assert hitl.corrections_jsonl.parent == expected_reports
    assert cal.report_json.parent == expected_reports


def test_providers_pass_run_id_through(tmp_path: Path) -> None:
    """The resolver receives the ``run_id`` argument verbatim."""
    seen_run_ids: list[str | None] = []

    def _resolver(run_id):  # type: ignore[no-untyped-def]
        seen_run_ids.append(run_id)
        return tmp_path

    sched_p, hitl_p, cal_p = build_finishing_layer_paths_providers(_resolver)
    sched_p("run-A")
    hitl_p("run-B")
    cal_p(None)
    assert seen_run_ids == ["run-A", "run-B", None]


def test_providers_propagate_resolver_exceptions(tmp_path: Path) -> None:
    """If the resolver raises, the provider must propagate the
    exception unchanged so the FastAPI router can map it to HTTP 404."""

    def _resolver(run_id):  # type: ignore[no-untyped-def]
        raise FileNotFoundError(f"no such run: {run_id}")

    sched_p, hitl_p, cal_p = build_finishing_layer_paths_providers(_resolver)
    for provider in (sched_p, hitl_p, cal_p):
        with pytest.raises(FileNotFoundError):
            provider("never-heard-of-it")


def test_register_finishing_layer_raises_runtime_error_without_fastapi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke check that the FastAPI dependency is announced loudly when
    missing. We monkey-patch the per-router constructor to raise the
    documented RuntimeError so we don't need to actually uninstall
    FastAPI in CI."""
    from pipeline.service import finishing_layer

    def _raise(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("FastAPI is not installed")

    monkeypatch.setattr(finishing_layer, "create_schedule_router", _raise)
    with pytest.raises(RuntimeError, match="FastAPI is not installed"):
        finishing_layer.register_finishing_layer(
            app=object(), run_resolver=lambda _r: tmp_path
        )
