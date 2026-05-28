"""Subprocess-based run executor for the service layer.

This module owns the *execution* of a run: it walks a subset of the stage
registry in topological order, spawns each stage as a subprocess via the
canonical ``python3 -m <cli_module> --config <path>`` invocation, streams
stdout/stderr line by line into the event broker, and updates the run
history store.

Why subprocess and not in-process
---------------------------------

The shipping ``run_*`` functions wrap COLMAP / Open3D / IfcOpenShell. Some
of those native libraries can segfault, exhaust GPU memory, or block on
stdin in pathological inputs; running them in-process inside the API
worker would risk taking the whole HTTP service down with one bad PLY
file. Subprocess isolation gives us:

- Stage segfaults that do not kill the API.
- A reliable cancellation handle (SIGTERM the child).
- The exact same invocation flow that the existing operator-facing CLI
  uses, so Phase 1's "no behavior change" guarantee carries through Phase
  2 byte-identically.

The cost is ~50 ms of process-startup overhead per stage. For a real run
where the cheapest stage takes seconds and the heaviest takes hours, that
cost is irrelevant.

What this module does NOT do
----------------------------

- **Pause / resume of running stages** is not implemented and is
  documented as such on the public API. Pausing a long-running native
  Open3D call in mid-iteration without crashing it is a real research
  problem; it does not belong in Phase 2. Cancellation works via SIGTERM,
  which is the operationally correct primitive for "stop now."
- **Concurrent runs** are not supported by default (``max_workers=1``).
  Setting ``max_workers > 1`` is allowed but the operator is responsible
  for understanding that COLMAP / GPU / disk are global resources.
- **Stage parallelism within a run** is not implemented. Stages run
  sequentially according to the topological order restricted to the
  requested subset. Phase 4 may revisit this.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from pipeline.common.config import PipelineConfig
from pipeline.common.provenance import compute_config_hash
from pipeline.common.registry import (
    STAGE_REGISTRY,
    RegistryError,
    StageDescriptor,
    StageRegistry,
)
from pipeline.service.events import EventBroker, RunEvent, RunEventKind
from pipeline.service.run_history import RunHistoryStore, RunRecord


LOGGER = logging.getLogger(__name__)


__all__ = [
    "RunSubmission",
    "RunHandle",
    "RunExecutor",
    "ExecutorError",
]


_DEFAULT_TERMINATE_GRACE_SECONDS: float = 8.0
_PROCESS_POLL_INTERVAL_SECONDS: float = 0.5


class ExecutorError(RuntimeError):
    """Raised when a run cannot be submitted or processed by the executor."""


@dataclass(frozen=True)
class RunSubmission:
    """A request to execute a subset of the registry against a config.

    Attributes
    ----------
    config_path : Path
        Absolute path to the YAML config the stages will be invoked with.
        Must already exist on disk; the executor does not create it.
    requested_stages : tuple[str, ...]
        Stage names to run, in any order. The executor sorts them by the
        registry's topological order. Duplicates are silently deduped.
    force : bool
        Whether to pass ``--force`` to each stage CLI. Most stages respect
        this; some ignore it. The executor passes it unconditionally.
    label : str | None
        Optional human-readable run label. When provided, used as the
        run_id prefix; otherwise a fresh UUID4 is used.
    """

    config_path: Path
    requested_stages: tuple[str, ...]
    force: bool = False
    label: str | None = None


@dataclass
class StageRuntime:
    """Per-stage runtime state. Mutable; owned by the run thread."""

    descriptor: StageDescriptor
    status: str = "queued"
    """One of ``queued``, ``running``, ``completed``, ``failed``, ``cancelled``, ``skipped``."""

    started_at_unix: float | None = None
    finished_at_unix: float | None = None
    return_code: int | None = None
    process: subprocess.Popen[str] | None = None


@dataclass
class RunHandle:
    """Handle returned by :meth:`RunExecutor.submit`.

    The handle is the supervisor's view of one run: it carries the
    cancellation flag, the future the worker is running on, and the
    in-memory per-stage state. It is not directly returned to API
    clients; the API serves a more redacted view from
    :meth:`RunExecutor.snapshot`.
    """

    run_id: str
    submission: RunSubmission
    record: RunRecord
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Future | None = None
    stages: list[StageRuntime] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class RunExecutor:
    """Sequential subprocess executor with cancellation and event streaming.

    Parameters
    ----------
    broker : EventBroker
        Receives every lifecycle event the executor publishes.
    history : RunHistoryStore
        Persists run metadata and event logs.
    registry : StageRegistry, optional
        Registry to resolve stage descriptors against. Defaults to the
        canonical :data:`pipeline.common.registry.STAGE_REGISTRY`.
    config_loader : callable, optional
        Function taking a path and returning a :class:`PipelineConfig`.
        Defaults to :func:`pipeline.common.config.load_config`. Tests
        override this to avoid the YAML round-trip on synthetic configs.
    max_workers : int
        Thread pool size. Default 1 (runs are not concurrent within a
        single executor; see module docstring for the rationale).
    terminate_grace_seconds : float
        Seconds to wait after SIGTERM before escalating to SIGKILL.
    repo_root : Path, optional
        Working directory passed to subprocesses. Defaults to the directory
        containing the loaded config file, which matches how the existing
        CLI is typically invoked.
    """

    def __init__(
        self,
        broker: EventBroker,
        history: RunHistoryStore,
        *,
        registry: StageRegistry | None = None,
        config_loader: Callable[[Path], PipelineConfig] | None = None,
        max_workers: int = 1,
        terminate_grace_seconds: float = _DEFAULT_TERMINATE_GRACE_SECONDS,
        repo_root: Path | None = None,
    ) -> None:
        self._broker = broker
        self._history = history
        self._registry = registry or STAGE_REGISTRY
        self._config_loader = config_loader or _default_config_loader
        self._terminate_grace = float(terminate_grace_seconds)
        self._repo_root = repo_root

        self._handles: dict[str, RunHandle] = {}
        self._handles_lock = threading.Lock()

        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="run-executor")
        self._closed = False

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def submit(self, submission: RunSubmission) -> RunHandle:
        """Validate and enqueue a run; return immediately with a handle."""
        if self._closed:
            raise ExecutorError("RunExecutor is closed")

        # Validate the requested stage set up-front so a malformed request
        # surfaces synchronously to the caller, not as an event later.
        descriptors = self._resolve_requested_stages(submission.requested_stages)

        # Validate the config can be loaded; capture project name + config
        # hash for the run record.
        cfg = self._config_loader(submission.config_path)
        project_name = str(cfg.require("project.name"))
        config_hash = compute_config_hash(cfg)

        run_id = self._mint_run_id(submission.label)
        record = RunRecord(
            run_id=run_id,
            project_name=project_name,
            config_path=str(submission.config_path),
            config_hash=config_hash,
            requested_stages=[d.name for d in descriptors],
            stage_statuses={d.name: "queued" for d in descriptors},
            status="queued",
        )
        self._history.save(record)

        handle = RunHandle(
            run_id=run_id,
            submission=submission,
            record=record,
            stages=[StageRuntime(descriptor=d) for d in descriptors],
        )
        with self._handles_lock:
            self._handles[run_id] = handle

        self._publish(run_id, RunEventKind.RUN_QUEUED, payload={
            "config_path": str(submission.config_path),
            "config_hash": config_hash,
            "project_name": project_name,
            "requested_stages": list(record.requested_stages),
            "force": submission.force,
        })
        for stage_name in record.requested_stages:
            self._publish(run_id, RunEventKind.STAGE_QUEUED, stage=stage_name)

        handle.future = self._pool.submit(self._execute_run, handle)
        return handle

    def cancel(self, run_id: str) -> bool:
        """Signal a run to stop. Returns True if a cancel was actually issued.

        The call returns once the cancel flag is set and any active
        subprocess has been signalled; it does **not** block until the
        subprocess actually exits. Callers that need to observe completion
        should subscribe to events.
        """
        with self._handles_lock:
            handle = self._handles.get(run_id)
        if handle is None:
            return False
        if handle.cancel_event.is_set():
            return False
        handle.cancel_event.set()
        # If a stage is currently in subprocess, signal it.
        with handle.lock:
            for stage_rt in handle.stages:
                proc = stage_rt.process
                if proc is not None and proc.poll() is None:
                    self._terminate_subprocess(proc)
        return True

    def snapshot(self, run_id: str) -> Mapping[str, Any] | None:
        """Return a JSON-friendly snapshot of one run's current state.

        Reads from the in-memory handle if the run is live; falls back to
        the on-disk history record otherwise.
        """
        with self._handles_lock:
            handle = self._handles.get(run_id)
        if handle is not None:
            with handle.lock:
                return {
                    "run_id": handle.run_id,
                    "submission": {
                        "config_path": str(handle.submission.config_path),
                        "requested_stages": list(handle.submission.requested_stages),
                        "force": handle.submission.force,
                    },
                    "status": handle.record.status,
                    "stages": [
                        {
                            "name": s.descriptor.name,
                            "status": s.status,
                            "started_at_unix": s.started_at_unix,
                            "finished_at_unix": s.finished_at_unix,
                            "return_code": s.return_code,
                        }
                        for s in handle.stages
                    ],
                    "config_hash": handle.record.config_hash,
                    "project_name": handle.record.project_name,
                    "submitted_at_unix": handle.record.submitted_at_unix,
                    "started_at_unix": handle.record.started_at_unix,
                    "finished_at_unix": handle.record.finished_at_unix,
                    "cancel_requested": handle.cancel_event.is_set(),
                }
        record = self._history.get(run_id)
        if record is None:
            return None
        return {
            "run_id": record.run_id,
            "submission": {
                "config_path": record.config_path,
                "requested_stages": list(record.requested_stages),
                "force": False,  # not preserved across restart; cosmetic
            },
            "status": record.status,
            "stages": [
                {"name": name, "status": status, "started_at_unix": None, "finished_at_unix": None, "return_code": None}
                for name, status in record.stage_statuses.items()
            ],
            "config_hash": record.config_hash,
            "project_name": record.project_name,
            "submitted_at_unix": record.submitted_at_unix,
            "started_at_unix": record.started_at_unix,
            "finished_at_unix": record.finished_at_unix,
            "cancel_requested": False,
        }

    def shutdown(self, *, wait: bool = True, cancel_running: bool = True) -> None:
        """Stop accepting new runs and (optionally) cancel everything in flight."""
        self._closed = True
        if cancel_running:
            with self._handles_lock:
                run_ids = list(self._handles.keys())
            for rid in run_ids:
                self.cancel(rid)
        self._pool.shutdown(wait=wait, cancel_futures=True)

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _resolve_requested_stages(self, names: Iterable[str]) -> tuple[StageDescriptor, ...]:
        seen: dict[str, StageDescriptor] = {}
        for name in names:
            if name in seen:
                continue
            try:
                descriptor = self._registry.get(name)
            except RegistryError as exc:
                raise ExecutorError(str(exc)) from exc
            seen[name] = descriptor
        if not seen:
            raise ExecutorError("requested_stages must be non-empty")
        # Topological order over the canonical registry; we filter to the
        # requested subset but preserve dependency order so e.g. stage_03
        # always runs before stage_05 if both are requested.
        topo = self._registry.topological_order()
        return tuple(d for d in topo if d.name in seen)

    def _mint_run_id(self, label: str | None) -> str:
        # Deterministic format: <YYYYmmdd_HHMMSS>_<label-or-uuid8>
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        suffix = (label or uuid.uuid4().hex[:8]).strip().replace(" ", "_")
        return f"{ts}_{suffix}"

    def _publish(
        self,
        run_id: str,
        kind: RunEventKind,
        *,
        stage: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        ev = RunEvent.make(run_id, kind, stage=stage, payload=payload)
        self._broker.publish(ev)
        # Persist to disk so a finished run's events survive an API
        # restart. Not under the broker lock to avoid blocking publish.
        try:
            self._history.append_events(run_id, [ev])
        except OSError as exc:  # pragma: no cover - disk-full edge
            LOGGER.warning("Run %s: failed to persist event %s: %s", run_id, kind.value, exc)

    def _execute_run(self, handle: RunHandle) -> None:
        run_id = handle.run_id
        record = handle.record

        if handle.cancel_event.is_set():
            self._finalize_run(handle, status="cancelled")
            return

        record.started_at_unix = time.time()
        record.status = "running"
        self._history.save(record)
        self._publish(run_id, RunEventKind.RUN_STARTED, payload={"started_at_unix": record.started_at_unix})

        any_failed = False
        for stage_rt in handle.stages:
            if handle.cancel_event.is_set():
                self._mark_remaining_cancelled(handle)
                self._finalize_run(handle, status="cancelled")
                return
            self._execute_stage(handle, stage_rt)
            if stage_rt.status == "failed":
                any_failed = True
                # On failure, mark every subsequent stage skipped — we do
                # not run dependents of a failed stage.
                self._mark_remaining_skipped(handle, after=stage_rt)
                break
            if stage_rt.status == "cancelled":
                self._mark_remaining_cancelled(handle)
                self._finalize_run(handle, status="cancelled")
                return

        self._finalize_run(handle, status="failed" if any_failed else "completed")

    def _execute_stage(self, handle: RunHandle, stage_rt: StageRuntime) -> None:
        run_id = handle.run_id
        descriptor = stage_rt.descriptor
        argv = list(descriptor.cli_invocation(str(handle.submission.config_path)))
        if handle.submission.force:
            argv.append("--force")

        with handle.lock:
            stage_rt.status = "running"
            stage_rt.started_at_unix = time.time()
        handle.record.stage_statuses[descriptor.name] = "running"
        self._history.save(handle.record)
        self._publish(
            run_id,
            RunEventKind.STAGE_STARTED,
            stage=descriptor.name,
            payload={"argv": argv, "started_at_unix": stage_rt.started_at_unix},
        )

        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(self._repo_root) if self._repo_root else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                # New process group so SIGTERM can target the whole tree if
                # the stage spawns helpers (COLMAP does on Linux).
                start_new_session=True,
            )
        except OSError as exc:
            with handle.lock:
                stage_rt.status = "failed"
                stage_rt.finished_at_unix = time.time()
                stage_rt.return_code = -1
            handle.record.stage_statuses[descriptor.name] = "failed"
            self._publish(
                run_id,
                RunEventKind.STAGE_FAILED,
                stage=descriptor.name,
                payload={"error": f"failed to spawn subprocess: {exc}"},
            )
            return

        with handle.lock:
            stage_rt.process = proc

        # Stream stdout/stderr line by line in two threads.
        stdout_thread = threading.Thread(
            target=self._pump_stream,
            args=(handle, descriptor.name, proc.stdout, "stdout"),
            name=f"{descriptor.name}.stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._pump_stream,
            args=(handle, descriptor.name, proc.stderr, "stderr"),
            name=f"{descriptor.name}.stderr",
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        # Poll the process while watching the cancel flag. wait() with a
        # short timeout lets us interleave the two without a busy loop.
        cancelled = False
        while True:
            try:
                rc = proc.wait(timeout=_PROCESS_POLL_INTERVAL_SECONDS)
                # The process may have exited because cancel() was called
                # from another thread and signalled the subprocess directly,
                # in which case we still want to label this as cancelled
                # (not failed) even though we did not enter the timeout
                # branch ourselves.
                if handle.cancel_event.is_set():
                    cancelled = True
                break
            except subprocess.TimeoutExpired:
                if handle.cancel_event.is_set():
                    cancelled = True
                    self._terminate_subprocess(proc)
                    # After signalling, give the grace period a chance.
                    try:
                        rc = proc.wait(timeout=self._terminate_grace)
                    except subprocess.TimeoutExpired:
                        # Escalate.
                        self._kill_subprocess(proc)
                        rc = proc.wait()
                    break

        # Drain reader threads.
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)

        with handle.lock:
            stage_rt.process = None
            stage_rt.return_code = rc
            stage_rt.finished_at_unix = time.time()
            if cancelled:
                stage_rt.status = "cancelled"
            elif rc == 0:
                stage_rt.status = "completed"
            else:
                stage_rt.status = "failed"

        handle.record.stage_statuses[descriptor.name] = stage_rt.status
        self._history.save(handle.record)

        if cancelled:
            self._publish(run_id, RunEventKind.STAGE_CANCELLED, stage=descriptor.name, payload={"return_code": rc})
        elif rc == 0:
            self._publish(run_id, RunEventKind.STAGE_FINISHED, stage=descriptor.name, payload={
                "return_code": rc,
                "elapsed_sec": (stage_rt.finished_at_unix or 0) - (stage_rt.started_at_unix or 0),
            })
        else:
            self._publish(run_id, RunEventKind.STAGE_FAILED, stage=descriptor.name, payload={
                "return_code": rc,
                "elapsed_sec": (stage_rt.finished_at_unix or 0) - (stage_rt.started_at_unix or 0),
            })

    def _pump_stream(self, handle: RunHandle, stage_name: str, stream: Any, label: str) -> None:
        """Forward each newline-terminated chunk from ``stream`` as a STAGE_PROGRESS event."""
        if stream is None:  # pragma: no cover - subprocess always provides streams
            return
        try:
            for line in stream:
                if line is None:
                    continue
                stripped = line.rstrip("\n")
                if not stripped:
                    continue
                self._publish(
                    handle.run_id,
                    RunEventKind.STAGE_PROGRESS,
                    stage=stage_name,
                    payload={"stream": label, "line": stripped},
                )
        finally:
            try:
                stream.close()
            except Exception:  # pragma: no cover
                pass

    def _terminate_subprocess(self, proc: subprocess.Popen[str]) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    def _kill_subprocess(self, proc: subprocess.Popen[str]) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def _mark_remaining_skipped(self, handle: RunHandle, *, after: StageRuntime) -> None:
        passed = False
        for stage_rt in handle.stages:
            if stage_rt is after:
                passed = True
                continue
            if not passed:
                continue
            with handle.lock:
                stage_rt.status = "skipped"
                stage_rt.finished_at_unix = time.time()
            handle.record.stage_statuses[stage_rt.descriptor.name] = "skipped"

    def _mark_remaining_cancelled(self, handle: RunHandle) -> None:
        for stage_rt in handle.stages:
            if stage_rt.status not in {"completed", "failed", "cancelled"}:
                with handle.lock:
                    stage_rt.status = "cancelled"
                    stage_rt.finished_at_unix = time.time()
                handle.record.stage_statuses[stage_rt.descriptor.name] = "cancelled"

    def _finalize_run(self, handle: RunHandle, *, status: str) -> None:
        handle.record.status = status
        handle.record.finished_at_unix = time.time()
        self._history.save(handle.record)
        kind_map = {
            "completed": RunEventKind.RUN_FINISHED,
            "failed": RunEventKind.RUN_FAILED,
            "cancelled": RunEventKind.RUN_CANCELLED,
        }
        kind = kind_map.get(status, RunEventKind.RUN_FAILED)
        self._publish(handle.run_id, kind, payload={
            "status": status,
            "finished_at_unix": handle.record.finished_at_unix,
        })


def _default_config_loader(path: Path) -> PipelineConfig:
    """Module-level default loader; allows tests to inject a stub without an import cycle."""
    from pipeline.common.config import load_config  # local import keeps top-of-module clean

    return load_config(path)
