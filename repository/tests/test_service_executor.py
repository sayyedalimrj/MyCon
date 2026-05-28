"""Tests for ``pipeline.service.executor``.

These tests use a synthetic registry and synthetic ``cli_invocation``
override so the real shipping stages are never spawned. That keeps the
suite fast and deterministic, and means the tests work in environments
that lack COLMAP / Open3D.

Cancellation is tested with ``sleep`` rather than a Python script because
we want to verify the SIGTERM-to-process-group path actually works on the
host kernel, not just the Python-side semantics.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from pipeline.common.config import PipelineConfig
from pipeline.common.registry import StageDescriptor, StageRegistry
from pipeline.common.schema import Stage01IngestSchema
from pipeline.service.events import EventBroker, RunEventKind
from pipeline.service.executor import ExecutorError, RunExecutor, RunSubmission
from pipeline.service.run_history import RunHistoryStore


# ---------------------------------------------------------------------------
# Shared fixtures: a tiny registry whose stages run real /bin/echo and /bin/sh
# scripts, plus a fake config loader.
# ---------------------------------------------------------------------------


def _fake_cfg(path: Path) -> PipelineConfig:
    return PipelineConfig(
        path=path,
        data={"project": {"name": "test", "random_seed": 42}},
    )


def _make_registry(*, descriptors) -> StageRegistry:
    reg = StageRegistry()
    for d in descriptors:
        reg.register(d)
    return reg


def _bin_echo_descriptor(name: str, order: int, *args: str) -> StageDescriptor:
    """Return a descriptor whose CLI is ``echo <args...>``."""
    d = StageDescriptor(
        name=name,
        order=order,
        title=name,
        description="test",
        cli_module="builtins",
        callable_name="print",
        schema_class=Stage01IngestSchema,
    )
    object.__setattr__(d, "_test_argv", ("echo", *args))
    return d


def _failing_descriptor(name: str, order: int) -> StageDescriptor:
    d = _bin_echo_descriptor(name, order)
    object.__setattr__(d, "_test_argv", ("sh", "-c", "exit 7"))
    return d


def _slow_descriptor(name: str, order: int) -> StageDescriptor:
    d = _bin_echo_descriptor(name, order)
    object.__setattr__(d, "_test_argv", ("sleep", "30"))
    return d


@pytest.fixture()
def patch_cli_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override ``cli_invocation`` to use the per-descriptor ``_test_argv``."""
    from pipeline.common.registry import StageDescriptor as SD

    def _ci(self, config_path: str) -> tuple[str, ...]:
        argv = getattr(self, "_test_argv", None) or ("echo", self.name)
        return tuple(argv)

    monkeypatch.setattr(SD, "cli_invocation", _ci)


@pytest.fixture()
def executor_setup(tmp_path: Path, patch_cli_invocation):
    """Build an executor with a fresh broker / history / temp config."""
    broker = EventBroker()
    history = RunHistoryStore(tmp_path / "service")
    cfg_path = tmp_path / "fake.yaml"
    cfg_path.write_text("project:\n  name: test\n  random_seed: 42\n")

    def _factory(registry: StageRegistry, *, terminate_grace: float = 2.0) -> RunExecutor:
        return RunExecutor(
            broker,
            history,
            registry=registry,
            config_loader=_fake_cfg,
            terminate_grace_seconds=terminate_grace,
        )

    yield broker, history, cfg_path, _factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_submit_runs_a_single_stage_to_completion(executor_setup) -> None:
    broker, history, cfg_path, factory = executor_setup
    reg = _make_registry(descriptors=[_bin_echo_descriptor("s1", 10, "hello")])
    executor = factory(reg)
    try:
        handle = executor.submit(RunSubmission(config_path=cfg_path, requested_stages=("s1",)))
        handle.future.result(timeout=15)
        snap = executor.snapshot(handle.run_id)
        assert snap is not None
        assert snap["status"] == "completed"
        assert snap["stages"][0]["status"] == "completed"
        assert snap["stages"][0]["return_code"] == 0
    finally:
        executor.shutdown()


def test_executor_runs_stages_in_topological_order(executor_setup) -> None:
    broker, history, cfg_path, factory = executor_setup
    # Two stages where s2 depends on s1; submit them in reverse order.
    s1 = _bin_echo_descriptor("s1", 10, "first")
    s2 = StageDescriptor(
        name="s2",
        order=20,
        title="s2",
        description="t",
        cli_module="builtins",
        callable_name="print",
        schema_class=Stage01IngestSchema,
        dependencies=("s1",),
    )
    object.__setattr__(s2, "_test_argv", ("echo", "second"))
    reg = _make_registry(descriptors=[s1, s2])
    executor = factory(reg)
    try:
        handle = executor.submit(RunSubmission(config_path=cfg_path, requested_stages=("s2", "s1")))
        handle.future.result(timeout=15)
        # Verify execution order via finished events
        events = [ev for ev in broker.replay(handle.run_id) if ev.kind == RunEventKind.STAGE_FINISHED]
        names = [ev.stage for ev in events]
        assert names == ["s1", "s2"]
    finally:
        executor.shutdown()


def test_failure_marks_run_failed_and_skips_subsequent_stages(executor_setup) -> None:
    broker, history, cfg_path, factory = executor_setup
    reg = _make_registry(
        descriptors=[
            _failing_descriptor("s1", 10),
            _bin_echo_descriptor("s2", 20, "should-not-run"),
        ]
    )
    executor = factory(reg)
    try:
        handle = executor.submit(RunSubmission(config_path=cfg_path, requested_stages=("s1", "s2")))
        handle.future.result(timeout=15)
        snap = executor.snapshot(handle.run_id)
        assert snap["status"] == "failed"
        statuses = {s["name"]: s["status"] for s in snap["stages"]}
        assert statuses["s1"] == "failed"
        assert statuses["s2"] == "skipped"
    finally:
        executor.shutdown()


def test_submit_validates_unknown_stage(executor_setup) -> None:
    broker, history, cfg_path, factory = executor_setup
    reg = _make_registry(descriptors=[_bin_echo_descriptor("s1", 10)])
    executor = factory(reg)
    try:
        with pytest.raises(ExecutorError):
            executor.submit(RunSubmission(config_path=cfg_path, requested_stages=("does-not-exist",)))
    finally:
        executor.shutdown()


def test_submit_validates_empty_stage_list(executor_setup) -> None:
    broker, history, cfg_path, factory = executor_setup
    reg = _make_registry(descriptors=[_bin_echo_descriptor("s1", 10)])
    executor = factory(reg)
    try:
        with pytest.raises(ExecutorError):
            executor.submit(RunSubmission(config_path=cfg_path, requested_stages=()))
    finally:
        executor.shutdown()


def test_cancel_running_stage_marks_run_cancelled(executor_setup) -> None:
    broker, history, cfg_path, factory = executor_setup
    reg = _make_registry(descriptors=[_slow_descriptor("slow", 10)])
    executor = factory(reg, terminate_grace=2.0)
    try:
        handle = executor.submit(RunSubmission(config_path=cfg_path, requested_stages=("slow",)))
        # Wait until the subprocess is actually running before cancelling, so
        # we exercise the SIGTERM path rather than the queued-cancel path.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            stage = handle.stages[0]
            if stage.process is not None and stage.process.poll() is None:
                break
            time.sleep(0.05)
        assert handle.stages[0].process is not None, "subprocess did not start"
        ack = executor.cancel(handle.run_id)
        assert ack is True
        handle.future.result(timeout=10)
        snap = executor.snapshot(handle.run_id)
        assert snap["status"] == "cancelled"
        assert snap["stages"][0]["status"] == "cancelled"
        # SIGTERM yields negative rc on POSIX, equal to -SIGTERM (-15).
        assert snap["stages"][0]["return_code"] is not None
        assert snap["stages"][0]["return_code"] < 0
    finally:
        executor.shutdown()


def test_cancel_unknown_run_returns_false(executor_setup) -> None:
    broker, history, cfg_path, factory = executor_setup
    reg = _make_registry(descriptors=[_bin_echo_descriptor("s1", 10)])
    executor = factory(reg)
    try:
        assert executor.cancel("never-existed") is False
    finally:
        executor.shutdown()


def test_snapshot_falls_back_to_history_after_executor_forgets(executor_setup) -> None:
    """A finished run must remain queryable even after the in-memory handle
    is gone — the history store is the persistent source of truth."""
    broker, history, cfg_path, factory = executor_setup
    reg = _make_registry(descriptors=[_bin_echo_descriptor("s1", 10)])
    executor = factory(reg)
    try:
        handle = executor.submit(RunSubmission(config_path=cfg_path, requested_stages=("s1",)))
        handle.future.result(timeout=15)
        run_id = handle.run_id
    finally:
        executor.shutdown()

    # Build a fresh executor against the same history; the run should still
    # be visible (even if the in-memory handle is no longer there).
    executor2 = RunExecutor(
        broker,
        history,
        registry=reg,
        config_loader=_fake_cfg,
        terminate_grace_seconds=2.0,
    )
    try:
        snap = executor2.snapshot(run_id)
        assert snap is not None
        assert snap["status"] == "completed"
    finally:
        executor2.shutdown()


def test_events_are_published_in_canonical_order(executor_setup) -> None:
    broker, history, cfg_path, factory = executor_setup
    reg = _make_registry(descriptors=[_bin_echo_descriptor("s1", 10, "the-line")])
    executor = factory(reg)
    try:
        handle = executor.submit(RunSubmission(config_path=cfg_path, requested_stages=("s1",)))
        handle.future.result(timeout=15)
        kinds = [ev.kind.value for ev in broker.replay(handle.run_id)]
        # Expected order: run.queued, stage.queued, run.started, stage.started,
        # one or more stage.progress, stage.finished, run.finished.
        assert kinds[0] == "run.queued"
        assert kinds[1] == "stage.queued"
        assert kinds[2] == "run.started"
        assert kinds[3] == "stage.started"
        assert kinds[-1] == "run.finished"
        assert kinds[-2] == "stage.finished"
        # At least one progress event, payload contains the printed line.
        progress = [ev for ev in broker.replay(handle.run_id) if ev.kind == RunEventKind.STAGE_PROGRESS]
        assert any("the-line" in ev.payload.get("line", "") for ev in progress)
    finally:
        executor.shutdown()
