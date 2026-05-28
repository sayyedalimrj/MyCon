"""Tests for ``pipeline.service.api`` — REST endpoints.

These tests use FastAPI's TestClient (which wraps ``httpx``) so the
async-IO is exercised end-to-end without spinning up uvicorn.

For run submission tests we use the same synthetic-stage trick as the
executor tests: monkeypatch ``StageDescriptor.cli_invocation`` so the
"stages" run real but trivial subprocesses (``echo`` / ``sleep``).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.common.config import PipelineConfig
from pipeline.common.registry import StageDescriptor, StageRegistry
from pipeline.common.schema import Stage01IngestSchema
from pipeline.service.app import AppDependencies, create_app
from pipeline.service.events import EventBroker
from pipeline.service.executor import RunExecutor
from pipeline.service.run_history import RunHistoryStore


# ---------------------------------------------------------------------------
# Test app builder
# ---------------------------------------------------------------------------


def _fake_cfg_loader(path: Path) -> PipelineConfig:
    return PipelineConfig(path=path, data={"project": {"name": "test", "random_seed": 42}})


@pytest.fixture()
def patch_cli_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipeline.common.registry import StageDescriptor as SD

    def _ci(self, config_path: str) -> tuple[str, ...]:
        argv = getattr(self, "_test_argv", None) or ("echo", self.name)
        return tuple(argv)

    monkeypatch.setattr(SD, "cli_invocation", _ci)


@pytest.fixture()
def synthetic_app(tmp_path: Path, patch_cli_invocation):
    """Build a TestClient with a tiny synthetic registry.

    Yields ``(client, deps, registry)``.
    """
    # Tiny registry: one quick stage, one slow stage, one failing stage.
    s1 = StageDescriptor(
        name="s1", order=10, title="quick", description="t",
        cli_module="builtins", callable_name="print",
        schema_class=Stage01IngestSchema,
    )
    object.__setattr__(s1, "_test_argv", ("echo", "hello-from-s1"))
    object.__setattr__(s1, "report_basename", None)  # no report to discover

    s_slow = StageDescriptor(
        name="slow", order=20, title="slow", description="t",
        cli_module="builtins", callable_name="print",
        schema_class=Stage01IngestSchema,
    )
    object.__setattr__(s_slow, "_test_argv", ("sleep", "30"))

    reg = StageRegistry()
    reg.register(s1)
    reg.register(s_slow)

    # Configs: one fake yaml that the route will load via fake loader.
    configs_root = tmp_path / "configs"
    configs_root.mkdir()
    cfg_path = configs_root / "test.yaml"
    cfg_path.write_text("project:\n  name: test\n  random_seed: 42\n")

    broker = EventBroker()
    history = RunHistoryStore(tmp_path / "service")
    executor = RunExecutor(
        broker, history, registry=reg, config_loader=_fake_cfg_loader, terminate_grace_seconds=2.0
    )
    deps = AppDependencies(
        broker=broker,
        history=history,
        executor=executor,
        project_root=tmp_path,
        configs_root=configs_root,
    )
    # Override the API router's ``registry`` arg via reaching into create_app
    # — simplest is to use create_router directly. But create_app uses
    # STAGE_REGISTRY; we want our synthetic one. Build the app manually.
    from fastapi import FastAPI
    from pipeline.service.api import create_router

    app = FastAPI()
    app.include_router(create_router(
        broker=broker, executor=executor, history=history,
        project_root=tmp_path, configs_root=configs_root,
        registry=reg, config_loader=_fake_cfg_loader,
    ))
    client = TestClient(app)
    try:
        yield client, deps, reg
    finally:
        executor.shutdown()


# ---------------------------------------------------------------------------
# Discovery endpoints
# ---------------------------------------------------------------------------


def test_health_endpoint(synthetic_app) -> None:
    client, _, reg = synthetic_app
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["stage_count"] == len(reg)


def test_list_stages_returns_registry(synthetic_app) -> None:
    client, _, reg = synthetic_app
    resp = client.get("/api/registry/stages")
    assert resp.status_code == 200
    names = {item["name"] for item in resp.json()}
    assert names == set(reg.names())


def test_get_stage_404_for_unknown(synthetic_app) -> None:
    client, _, _ = synthetic_app
    resp = client.get("/api/registry/stages/does-not-exist")
    assert resp.status_code == 404


def test_list_vlm_backends_returns_known_names(synthetic_app) -> None:
    client, _, _ = synthetic_app
    resp = client.get("/api/registry/vlm-backends")
    assert resp.status_code == 200
    names = {item["name"] for item in resp.json()}
    assert {"mock", "ollama_local", "openai_compatible_local"}.issubset(names)


def test_list_depth_providers_returns_known_names(synthetic_app) -> None:
    client, _, _ = synthetic_app
    resp = client.get("/api/registry/depth-providers")
    assert resp.status_code == 200
    names = {item["name"] for item in resp.json()}
    assert {"precomputed", "external_command"}.issubset(names)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


def test_list_configs_finds_fake_yaml(synthetic_app) -> None:
    client, _, _ = synthetic_app
    resp = client.get("/api/configs")
    assert resp.status_code == 200
    names = {item["name"] for item in resp.json()}
    assert "test" in names


def test_get_config_returns_data_and_hash(synthetic_app) -> None:
    client, _, _ = synthetic_app
    resp = client.get("/api/configs/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "test"
    assert body["data"]["project"]["name"] == "test"
    assert len(body["config_hash"]) == 64


def test_get_config_404_for_unknown(synthetic_app) -> None:
    client, _, _ = synthetic_app
    resp = client.get("/api/configs/does-not-exist")
    assert resp.status_code == 404


def test_path_traversal_in_config_name_is_rejected(synthetic_app) -> None:
    """``../etc/passwd`` must not escape configs/."""
    client, _, _ = synthetic_app
    resp = client.get("/api/configs/..%2Fetc%2Fpasswd")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def test_submit_and_get_run(synthetic_app) -> None:
    client, _, _ = synthetic_app
    resp = client.post("/api/runs", json={
        "config_path": "configs/test.yaml",
        "requested_stages": ["s1"],
        "force": False,
    })
    assert resp.status_code == 201
    run_id = resp.json()["run_id"]

    # Wait for completion via repeated GETs (test client; no WS yet).
    deadline = time.time() + 15
    while time.time() < deadline:
        snap = client.get(f"/api/runs/{run_id}").json()
        if snap["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.1)
    assert snap["status"] == "completed"


def test_submit_invalid_stage_returns_400(synthetic_app) -> None:
    client, _, _ = synthetic_app
    resp = client.post("/api/runs", json={
        "config_path": "configs/test.yaml",
        "requested_stages": ["does-not-exist"],
        "force": False,
    })
    assert resp.status_code == 400


def test_submit_missing_config_returns_400(synthetic_app) -> None:
    client, _, _ = synthetic_app
    resp = client.post("/api/runs", json={
        "config_path": "configs/no-such.yaml",
        "requested_stages": ["s1"],
        "force": False,
    })
    assert resp.status_code == 400


def test_get_unknown_run_returns_404(synthetic_app) -> None:
    client, _, _ = synthetic_app
    resp = client.get("/api/runs/does-not-exist")
    assert resp.status_code == 404


def test_cancel_running_stage(synthetic_app) -> None:
    client, _, _ = synthetic_app
    resp = client.post("/api/runs", json={
        "config_path": "configs/test.yaml",
        "requested_stages": ["slow"],
        "force": False,
    })
    assert resp.status_code == 201
    run_id = resp.json()["run_id"]

    # Wait for the subprocess to actually start before cancelling.
    deadline = time.time() + 5
    while time.time() < deadline:
        snap = client.get(f"/api/runs/{run_id}").json()
        if snap["stages"][0]["status"] == "running":
            break
        time.sleep(0.05)
    cancel = client.post(f"/api/runs/{run_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["cancel_requested"] is True

    # Wait for completion.
    deadline = time.time() + 15
    while time.time() < deadline:
        snap = client.get(f"/api/runs/{run_id}").json()
        if snap["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.1)
    assert snap["status"] == "cancelled"


def test_replay_events_after_completion(synthetic_app) -> None:
    client, _, _ = synthetic_app
    resp = client.post("/api/runs", json={
        "config_path": "configs/test.yaml",
        "requested_stages": ["s1"],
        "force": False,
    })
    run_id = resp.json()["run_id"]
    # Wait for completion.
    deadline = time.time() + 15
    while time.time() < deadline:
        snap = client.get(f"/api/runs/{run_id}").json()
        if snap["status"] == "completed":
            break
        time.sleep(0.1)

    events = client.get(f"/api/runs/{run_id}/events").json()
    kinds = [ev["kind"] for ev in events]
    assert "run.queued" in kinds
    assert "run.finished" in kinds
    # ``stage.progress`` events carry the printed line.
    progress = [ev for ev in events if ev["kind"] == "stage.progress"]
    assert any("hello-from-s1" in (ev["payload"].get("line") or "") for ev in progress)


def test_artifacts_endpoint_returns_summaries_for_known_run(synthetic_app) -> None:
    client, _, _ = synthetic_app
    # Run a stage to produce a known run_id.
    resp = client.post("/api/runs", json={
        "config_path": "configs/test.yaml",
        "requested_stages": ["s1"],
    })
    run_id = resp.json()["run_id"]
    # Wait for completion.
    deadline = time.time() + 15
    while time.time() < deadline:
        snap = client.get(f"/api/runs/{run_id}").json()
        if snap["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.1)

    art = client.get(f"/api/runs/{run_id}/artifacts")
    assert art.status_code == 200
    # Synthetic registry has no report_basename for s1, so the list is empty
    # — the contract is "no exception, just an empty list."
    assert isinstance(art.json(), list)


def test_artifacts_endpoint_404_for_unknown_run(synthetic_app) -> None:
    client, _, _ = synthetic_app
    resp = client.get("/api/runs/never-existed/artifacts")
    assert resp.status_code == 404
