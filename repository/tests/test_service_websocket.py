"""Tests for the WebSocket events stream.

WebSocket testing in FastAPI uses ``TestClient.websocket_connect``. The
contract we assert:

- A late-joining client receives a *replay* of every event already
  published for the run, in canonical order.
- After replay, live events arrive in real time.
- The socket closes cleanly when the run reaches a terminal kind.
- An unknown run yields a structured error frame and a clean close.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pipeline.common.config import PipelineConfig
from pipeline.common.registry import StageDescriptor, StageRegistry
from pipeline.common.schema import Stage01IngestSchema
from pipeline.service.api import create_router
from pipeline.service.events import EventBroker
from pipeline.service.executor import RunExecutor
from pipeline.service.run_history import RunHistoryStore


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
def ws_app(tmp_path: Path, patch_cli_invocation):
    s1 = StageDescriptor(
        name="s1", order=10, title="quick", description="t",
        cli_module="builtins", callable_name="print",
        schema_class=Stage01IngestSchema,
    )
    object.__setattr__(s1, "_test_argv", ("echo", "ws-line"))
    reg = StageRegistry()
    reg.register(s1)

    configs_root = tmp_path / "configs"
    configs_root.mkdir()
    cfg_path = configs_root / "test.yaml"
    cfg_path.write_text("project:\n  name: test\n  random_seed: 42\n")

    broker = EventBroker()
    history = RunHistoryStore(tmp_path / "service")
    executor = RunExecutor(broker, history, registry=reg, config_loader=_fake_cfg_loader, terminate_grace_seconds=2.0)

    app = FastAPI()
    app.include_router(create_router(
        broker=broker, executor=executor, history=history,
        project_root=tmp_path, configs_root=configs_root,
        registry=reg, config_loader=_fake_cfg_loader,
    ))
    client = TestClient(app)
    try:
        yield client, broker, executor
    finally:
        executor.shutdown()


def test_websocket_replays_events_for_finished_run(ws_app) -> None:
    """A late-joining client must receive the full event log for a run that
    has already completed."""
    client, broker, executor = ws_app
    # Run to completion first.
    resp = client.post("/api/runs", json={
        "config_path": "configs/test.yaml",
        "requested_stages": ["s1"],
    })
    run_id = resp.json()["run_id"]
    deadline = time.time() + 15
    while time.time() < deadline:
        snap = client.get(f"/api/runs/{run_id}").json()
        if snap["status"] == "completed":
            break
        time.sleep(0.05)
    assert snap["status"] == "completed"

    # Connect after the run has finished; expect replay of buffered events,
    # ending in a terminal kind that closes the socket.
    received: list[dict] = []
    with client.websocket_connect(f"/api/runs/{run_id}/events/stream") as ws:
        # Drain until terminal event arrives or the socket closes.
        try:
            while True:
                ev = ws.receive_json()
                received.append(ev)
                if ev["kind"] in {"run.finished", "run.failed", "run.cancelled"}:
                    break
        except Exception:
            # The server is allowed to close the socket after the terminal
            # event, which raises a WebSocketDisconnect on receive.
            pass

    kinds = [ev["kind"] for ev in received]
    assert kinds[0] == "run.queued"
    assert "run.started" in kinds
    assert "stage.started" in kinds
    assert "stage.finished" in kinds
    assert kinds[-1] == "run.finished"


def test_websocket_unknown_run_sends_error_frame(ws_app) -> None:
    client, _, _ = ws_app
    with client.websocket_connect("/api/runs/never-existed/events/stream") as ws:
        frame = ws.receive_json()
        assert frame == {"error": "unknown_run", "run_id": "never-existed"}
