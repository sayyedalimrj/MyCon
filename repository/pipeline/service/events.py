"""In-memory event broker for run lifecycle events.

This module is the contract every other piece of the service layer sits on
top of:

- :mod:`pipeline.service.executor` *publishes* events (queued, started,
  stdout chunk, finished, cancelled, failed).
- :mod:`pipeline.service.api` *subscribes* WebSocket clients to those
  events.
- :mod:`pipeline.service.run_history` *persists* finished events so a GUI
  loaded against a completed run can replay history.

The broker is intentionally *in-memory*. Distributed durability is not a
Phase 2 goal; the assumption is one API process per machine, which matches
how a research workstation actually runs. If that assumption ever changes,
the broker interface is small enough to swap for a Redis pub/sub adapter
without touching call sites.

Design properties
-----------------

- **Thread-safe.** Multiple stages may publish concurrently; the API
  process may subscribe concurrently. All mutation happens under a single
  ``threading.Lock``.
- **Async-friendly.** Subscribers receive events through an
  :class:`asyncio.Queue` per subscription. ``publish`` is callable from any
  thread; the broker uses ``loop.call_soon_threadsafe`` to deliver.
- **Bounded backpressure.** Each subscription queue is bounded
  (``maxsize=256`` by default). When a subscriber falls behind, oldest
  events are dropped first and a single
  :class:`RunEvent` of kind ``backpressure_drop`` is enqueued so the
  client knows it lost data — silent loss is unacceptable.
- **Replay.** ``replay(run_id)`` returns the in-memory log of events for
  one run. The persistence layer
  (:mod:`pipeline.service.run_history`) reads this log on shutdown for
  finished runs, and serves history-only replays from disk.

The broker holds *no* knowledge of FastAPI / WebSocket frame formats; the
API layer is responsible for serializing :class:`RunEvent` to JSON.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Mapping

LOGGER = logging.getLogger(__name__)


__all__ = [
    "RunEventKind",
    "RunEvent",
    "Subscription",
    "EventBroker",
]


# Default per-subscription queue capacity. 256 events ≈ ~1 minute of typical
# stage output at the rates we generate; a slow GUI client can fall back
# without losing the entire stream.
_DEFAULT_QUEUE_MAXSIZE: int = 256


class RunEventKind(str, Enum):
    """All event kinds the executor may publish."""

    RUN_QUEUED = "run.queued"
    """A run has been accepted and queued for execution."""

    RUN_STARTED = "run.started"
    """Execution has begun; the run is now occupying the executor."""

    RUN_FINISHED = "run.finished"
    """Run completed successfully; ``payload['exit_status'] == 'completed'``."""

    RUN_FAILED = "run.failed"
    """Run terminated with at least one stage failure."""

    RUN_CANCELLED = "run.cancelled"
    """Run was cancelled by an operator before all stages completed."""

    STAGE_QUEUED = "stage.queued"
    """A stage was queued within a run (i.e. its dependencies were satisfied)."""

    STAGE_STARTED = "stage.started"
    """A stage's subprocess was successfully spawned."""

    STAGE_PROGRESS = "stage.progress"
    """A line of stdout / stderr from the stage subprocess.

    Payload contains ``{"stream": "stdout"|"stderr", "line": "..."}``."""

    STAGE_FINISHED = "stage.finished"
    """Stage subprocess exited with returncode 0."""

    STAGE_FAILED = "stage.failed"
    """Stage subprocess exited with non-zero returncode."""

    STAGE_CANCELLED = "stage.cancelled"
    """Stage subprocess was killed before completion."""

    BACKPRESSURE_DROP = "broker.backpressure_drop"
    """Sentinel inserted into a subscription when its queue overflows."""


@dataclass(frozen=True)
class RunEvent:
    """One event in a run's lifecycle.

    Attributes
    ----------
    event_id : str
        Server-assigned UUID. Useful for clients reconnecting mid-stream
        to deduplicate.
    run_id : str
        Identifier of the run that produced this event.
    stage : str | None
        Stage name when the event is stage-scoped; ``None`` for run-scoped
        events (queued / started / finished / failed / cancelled).
    kind : RunEventKind
        Event kind; clients should switch on this.
    timestamp_unix : float
        ``time.time()`` at event creation.
    payload : Mapping[str, Any]
        Event-kind-specific data. The schema for each kind is documented
        on :class:`RunEventKind`.
    """

    event_id: str
    run_id: str
    stage: str | None
    kind: RunEventKind
    timestamp_unix: float
    payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def make(
        cls,
        run_id: str,
        kind: RunEventKind,
        *,
        stage: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> "RunEvent":
        return cls(
            event_id=str(uuid.uuid4()),
            run_id=run_id,
            stage=stage,
            kind=kind,
            timestamp_unix=time.time(),
            payload=dict(payload or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view; the API layer uses this for WebSocket frames."""
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "stage": self.stage,
            "kind": self.kind.value,
            "timestamp_unix": self.timestamp_unix,
            "payload": dict(self.payload),
        }


@dataclass
class Subscription:
    """A live subscription to a run's events.

    Created by :meth:`EventBroker.subscribe`. Holds an asyncio queue the
    client awaits on, plus the run filter and the event loop the
    subscription belongs to.

    Attributes
    ----------
    subscription_id : str
        UUID assigned by the broker; used for unsubscribe.
    run_id : str | None
        When non-None, only events for this run are delivered. When None,
        the subscription receives every event (used by the future GUI's
        global activity ticker).
    queue : asyncio.Queue[RunEvent]
        The bounded queue events are pushed onto.
    loop : asyncio.AbstractEventLoop
        The loop that owns ``queue``; the broker calls back into it via
        ``call_soon_threadsafe``.
    """

    subscription_id: str
    run_id: str | None
    queue: "asyncio.Queue[RunEvent]"
    loop: asyncio.AbstractEventLoop

    def __aiter__(self) -> "Subscription":
        return self

    async def __anext__(self) -> RunEvent:
        return await self.queue.get()


class EventBroker:
    """Thread-safe event publisher with replay buffer.

    Lifecycle: typically one broker per :class:`pipeline.service.app`
    application. The broker survives across runs; its per-run replay
    buffers grow until ``forget(run_id)`` is called.
    """

    def __init__(self, *, max_buffered_events_per_run: int = 10_000) -> None:
        self._lock = threading.Lock()
        self._subscriptions: dict[str, Subscription] = {}
        # Replay buffers, keyed by run_id. Capped per run to bound memory.
        self._buffers: dict[str, list[RunEvent]] = {}
        self._buffer_cap = int(max_buffered_events_per_run)

    # ---- subscription management ----

    def subscribe(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        run_id: str | None = None,
        queue_maxsize: int = _DEFAULT_QUEUE_MAXSIZE,
    ) -> Subscription:
        """Create a new subscription bound to ``loop``.

        The caller is responsible for awaiting on ``subscription.queue.get()``
        (or iterating the subscription directly via ``async for``) and for
        calling :meth:`unsubscribe` when done. The broker will not garbage-
        collect dropped subscriptions on its own.
        """
        sub = Subscription(
            subscription_id=str(uuid.uuid4()),
            run_id=run_id,
            queue=asyncio.Queue(maxsize=queue_maxsize),
            loop=loop,
        )
        with self._lock:
            self._subscriptions[sub.subscription_id] = sub
        return sub

    def unsubscribe(self, subscription_id: str) -> None:
        with self._lock:
            self._subscriptions.pop(subscription_id, None)

    def subscriber_count(self, *, run_id: str | None = None) -> int:
        """Diagnostic helper used by tests and the ``/api/health`` endpoint."""
        with self._lock:
            if run_id is None:
                return len(self._subscriptions)
            return sum(1 for s in self._subscriptions.values() if s.run_id == run_id)

    # ---- publication ----

    def publish(self, event: RunEvent) -> None:
        """Buffer the event and deliver to every matching subscription.

        Safe to call from any thread. Delivery into asyncio queues is
        scheduled via ``loop.call_soon_threadsafe`` so subscriptions are
        never blocked by publication.
        """
        with self._lock:
            buf = self._buffers.setdefault(event.run_id, [])
            buf.append(event)
            if len(buf) > self._buffer_cap:
                # Drop the oldest. We don't insert a sentinel here because
                # this trim is per-run replay only; live subscribers get
                # their own backpressure-drop sentinel via _enqueue.
                del buf[: len(buf) - self._buffer_cap]
            targets = [s for s in self._subscriptions.values() if s.run_id in (None, event.run_id)]
        for sub in targets:
            self._enqueue(sub, event)

    def _enqueue(self, sub: Subscription, event: RunEvent) -> None:
        """Best-effort delivery into one subscription's queue.

        Uses :meth:`asyncio.AbstractEventLoop.call_soon_threadsafe` so a
        cross-thread publish does not require the publishing thread to
        know anything about asyncio.
        """
        loop = sub.loop
        if loop.is_closed():
            return

        def _put() -> None:
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drain the oldest event and replace with a backpressure
                # marker so the client learns that data was lost.
                try:
                    sub.queue.get_nowait()
                    sub.queue.task_done()
                except (asyncio.QueueEmpty, ValueError):
                    pass
                drop = RunEvent.make(
                    event.run_id,
                    RunEventKind.BACKPRESSURE_DROP,
                    stage=event.stage,
                    payload={"dropped_event_id": event.event_id, "subscription_id": sub.subscription_id},
                )
                try:
                    sub.queue.put_nowait(drop)
                except asyncio.QueueFull:  # pragma: no cover - extreme overload
                    LOGGER.warning("Subscription %s queue full even after drop; event lost", sub.subscription_id)

        try:
            loop.call_soon_threadsafe(_put)
        except RuntimeError:
            # Loop already shut down; treat as a closed subscription.
            return

    # ---- replay ----

    def replay(self, run_id: str) -> tuple[RunEvent, ...]:
        """Return the in-memory event log for ``run_id`` in publish order."""
        with self._lock:
            return tuple(self._buffers.get(run_id, ()))

    def forget(self, run_id: str) -> None:
        """Discard the replay buffer for one run.

        Called by the run-history layer once events have been persisted to
        disk and the run is finished.
        """
        with self._lock:
            self._buffers.pop(run_id, None)

    def all_run_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._buffers.keys())

    # ---- iteration helper for tests ----

    def iter_buffered(self, run_id: str) -> Iterator[RunEvent]:
        """Synchronous iteration over the replay buffer; useful in tests."""
        with self._lock:
            yield from list(self._buffers.get(run_id, []))
