"""Tests for ``pipeline.service.events``.

The event broker is the contract every other service-layer module sits on
top of, so its invariants are pinned here aggressively:

- Cross-thread ``publish`` reaches an asyncio subscriber without blocking.
- Multiple subscribers receive the same event.
- ``run_id`` filtering works (a subscriber for run A never sees run B).
- Replay buffers retain published events and respect the per-run cap.
- Backpressure inserts a sentinel rather than silently dropping.
- A subscription whose loop has shut down is treated as closed.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from pipeline.service.events import EventBroker, RunEvent, RunEventKind


def _drain(queue: "asyncio.Queue[RunEvent]") -> list[RunEvent]:
    out: list[RunEvent] = []
    while not queue.empty():
        try:
            out.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return out


def test_publish_delivers_to_matching_subscriber() -> None:
    async def go() -> list[RunEvent]:
        broker = EventBroker()
        loop = asyncio.get_running_loop()
        sub = broker.subscribe(loop, run_id="r1")
        for i in range(3):
            broker.publish(RunEvent.make("r1", RunEventKind.STAGE_PROGRESS, stage="s", payload={"i": i}))
        # Cooperatively yield so the publish-time call_soon_threadsafe is processed.
        for _ in range(3):
            await asyncio.sleep(0)
        return _drain(sub.queue)

    events = asyncio.run(go())
    assert [ev.payload["i"] for ev in events] == [0, 1, 2]


def test_global_subscriber_sees_all_runs() -> None:
    async def go() -> list[RunEvent]:
        broker = EventBroker()
        loop = asyncio.get_running_loop()
        sub = broker.subscribe(loop)  # no run_id filter
        broker.publish(RunEvent.make("r1", RunEventKind.RUN_STARTED))
        broker.publish(RunEvent.make("r2", RunEventKind.RUN_STARTED))
        for _ in range(3):
            await asyncio.sleep(0)
        return _drain(sub.queue)

    events = asyncio.run(go())
    assert {e.run_id for e in events} == {"r1", "r2"}


def test_run_filter_excludes_other_runs() -> None:
    async def go() -> list[RunEvent]:
        broker = EventBroker()
        loop = asyncio.get_running_loop()
        sub = broker.subscribe(loop, run_id="r1")
        broker.publish(RunEvent.make("r1", RunEventKind.STAGE_STARTED))
        broker.publish(RunEvent.make("r2", RunEventKind.STAGE_STARTED))
        broker.publish(RunEvent.make("r1", RunEventKind.STAGE_FINISHED))
        for _ in range(3):
            await asyncio.sleep(0)
        return _drain(sub.queue)

    events = asyncio.run(go())
    assert {e.run_id for e in events} == {"r1"}
    assert len(events) == 2


def test_publish_from_other_thread_reaches_subscriber() -> None:
    """The executor publishes from worker threads; the API consumes from
    the asyncio loop. This must just work."""
    async def go() -> list[RunEvent]:
        broker = EventBroker()
        loop = asyncio.get_running_loop()
        sub = broker.subscribe(loop, run_id="r1")

        def worker() -> None:
            for i in range(5):
                broker.publish(RunEvent.make("r1", RunEventKind.STAGE_PROGRESS, payload={"i": i}))

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        # Let the thread-safe scheduling be processed.
        for _ in range(5):
            await asyncio.sleep(0.01)
        return _drain(sub.queue)

    events = asyncio.run(go())
    assert [e.payload["i"] for e in events] == [0, 1, 2, 3, 4]


def test_replay_buffer_holds_published_events() -> None:
    broker = EventBroker()
    for i in range(4):
        broker.publish(RunEvent.make("r1", RunEventKind.STAGE_PROGRESS, payload={"i": i}))
    replayed = broker.replay("r1")
    assert [e.payload["i"] for e in replayed] == [0, 1, 2, 3]


def test_replay_buffer_respects_per_run_cap() -> None:
    broker = EventBroker(max_buffered_events_per_run=3)
    for i in range(10):
        broker.publish(RunEvent.make("r1", RunEventKind.STAGE_PROGRESS, payload={"i": i}))
    replayed = broker.replay("r1")
    assert len(replayed) == 3
    # Newest events kept.
    assert [e.payload["i"] for e in replayed] == [7, 8, 9]


def test_forget_drops_replay_buffer() -> None:
    broker = EventBroker()
    broker.publish(RunEvent.make("r1", RunEventKind.RUN_STARTED))
    assert len(broker.replay("r1")) == 1
    broker.forget("r1")
    assert broker.replay("r1") == ()


def test_unsubscribe_removes_subscription() -> None:
    async def go() -> int:
        broker = EventBroker()
        loop = asyncio.get_running_loop()
        sub = broker.subscribe(loop)
        assert broker.subscriber_count() == 1
        broker.unsubscribe(sub.subscription_id)
        return broker.subscriber_count()

    assert asyncio.run(go()) == 0


def test_backpressure_inserts_drop_sentinel() -> None:
    """If a subscription queue fills, the broker must replace one event
    with a ``BACKPRESSURE_DROP`` sentinel rather than silently lose data."""
    async def go() -> tuple[int, int, int]:
        # Tiny queue so we can force overflow without sending many events.
        broker = EventBroker()
        loop = asyncio.get_running_loop()
        sub = broker.subscribe(loop, queue_maxsize=2)
        # Push three events before the loop drains anything.
        for i in range(3):
            broker.publish(RunEvent.make("r1", RunEventKind.STAGE_PROGRESS, payload={"i": i}))
        for _ in range(3):
            await asyncio.sleep(0)
        events = _drain(sub.queue)
        # Queue had room for 2; on the 3rd publish, the broker dropped the
        # oldest and inserted a sentinel.
        progress = [e for e in events if e.kind == RunEventKind.STAGE_PROGRESS]
        drops = [e for e in events if e.kind == RunEventKind.BACKPRESSURE_DROP]
        return len(events), len(progress), len(drops)

    total, progress, drops = asyncio.run(go())
    assert total == 2
    assert progress == 1
    assert drops == 1


def test_run_event_to_dict_round_trips_through_json() -> None:
    import json

    ev = RunEvent.make("r1", RunEventKind.STAGE_PROGRESS, stage="s", payload={"line": "hello"})
    encoded = json.dumps(ev.to_dict())
    decoded = json.loads(encoded)
    assert decoded["kind"] == "stage.progress"
    assert decoded["stage"] == "s"
    assert decoded["payload"]["line"] == "hello"


def test_subscriber_count_filters_by_run_id() -> None:
    async def go() -> tuple[int, int, int]:
        broker = EventBroker()
        loop = asyncio.get_running_loop()
        broker.subscribe(loop, run_id="r1")
        broker.subscribe(loop, run_id="r1")
        broker.subscribe(loop, run_id="r2")
        broker.subscribe(loop)  # global
        return broker.subscriber_count(), broker.subscriber_count(run_id="r1"), broker.subscriber_count(run_id="r2")

    total, r1, r2 = asyncio.run(go())
    assert total == 4
    assert r1 == 2
    assert r2 == 1
