"""Tests for ``pipeline.service.run_history``.

Pinned contracts:

- A new store reads back any prior history.
- Records are ordered newest-first.
- Event logs append correctly and survive a process restart.
- A corrupted single line in an event log does not lose the whole run.
- Atomic JSON write does not leave a half-written ``run_history.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pipeline.service.events import RunEvent, RunEventKind
from pipeline.service.run_history import RunHistoryStore, RunRecord


def _make_record(run_id: str, *, status: str = "queued", submitted_offset: float = 0.0) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        project_name="site01",
        config_path="/abs/path/site01.yaml",
        config_hash=run_id + "_hash",
        requested_stages=["stage_09_progress"],
        submitted_at_unix=time.time() + submitted_offset,
        status=status,
    )


def test_save_and_get(tmp_path: Path) -> None:
    store = RunHistoryStore(tmp_path)
    store.save(_make_record("run-001"))
    rec = store.get("run-001")
    assert rec is not None
    assert rec.project_name == "site01"


def test_get_unknown_returns_none(tmp_path: Path) -> None:
    store = RunHistoryStore(tmp_path)
    assert store.get("nonexistent") is None


def test_list_is_newest_first(tmp_path: Path) -> None:
    store = RunHistoryStore(tmp_path)
    store.save(_make_record("old", submitted_offset=-100.0))
    store.save(_make_record("new", submitted_offset=0.0))
    listed = [r.run_id for r in store.list()]
    assert listed == ["new", "old"]


def test_list_respects_limit(tmp_path: Path) -> None:
    store = RunHistoryStore(tmp_path)
    for i in range(5):
        store.save(_make_record(f"r-{i}", submitted_offset=float(i)))
    listed = store.list(limit=2)
    assert len(listed) == 2
    # Newest first, so r-4, r-3.
    assert [r.run_id for r in listed] == ["r-4", "r-3"]


def test_persistence_across_restart(tmp_path: Path) -> None:
    store_a = RunHistoryStore(tmp_path)
    store_a.save(_make_record("run-001", status="completed"))
    store_a.save(_make_record("run-002", status="failed"))

    # Fresh store on the same directory
    store_b = RunHistoryStore(tmp_path)
    rec_a = store_b.get("run-001")
    rec_b = store_b.get("run-002")
    assert rec_a is not None and rec_a.status == "completed"
    assert rec_b is not None and rec_b.status == "failed"


def test_event_log_round_trip(tmp_path: Path) -> None:
    store = RunHistoryStore(tmp_path)
    events = [
        RunEvent.make("r1", RunEventKind.RUN_STARTED),
        RunEvent.make("r1", RunEventKind.STAGE_STARTED, stage="s", payload={"argv": ["echo"]}),
        RunEvent.make("r1", RunEventKind.STAGE_FINISHED, stage="s", payload={"return_code": 0}),
    ]
    n = store.append_events("r1", events)
    assert n == 3
    reread = store.read_events("r1")
    assert len(reread) == 3
    assert [e["kind"] for e in reread] == ["run.started", "stage.started", "stage.finished"]


def test_event_log_tolerates_one_corrupt_line(tmp_path: Path) -> None:
    store = RunHistoryStore(tmp_path)
    store.append_events("r1", [RunEvent.make("r1", RunEventKind.RUN_STARTED)])
    # Inject a corrupt line in the middle of the log.
    log_path = tmp_path / "events" / "r1.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write("not-json\n")
    store.append_events("r1", [RunEvent.make("r1", RunEventKind.RUN_FINISHED)])
    reread = store.read_events("r1")
    # The corrupt line is dropped; the other two are preserved.
    assert len(reread) == 2


def test_read_events_returns_empty_for_unknown_run(tmp_path: Path) -> None:
    store = RunHistoryStore(tmp_path)
    assert store.read_events("never-existed") == []


def test_delete_removes_record_and_event_log(tmp_path: Path) -> None:
    store = RunHistoryStore(tmp_path)
    store.save(_make_record("r-x"))
    store.append_events("r-x", [RunEvent.make("r-x", RunEventKind.RUN_STARTED)])
    assert store.delete("r-x") is True
    assert store.get("r-x") is None
    assert store.read_events("r-x") == []
    assert store.delete("r-x") is False  # second delete is a no-op


def test_history_file_is_well_formed_json_after_save(tmp_path: Path) -> None:
    store = RunHistoryStore(tmp_path)
    for i in range(3):
        store.save(_make_record(f"run-{i}"))
    raw = (tmp_path / "run_history.json").read_text(encoding="utf-8")
    decoded = json.loads(raw)
    assert isinstance(decoded, list)
    assert len(decoded) == 3
    for entry in decoded:
        assert "run_id" in entry
        assert "config_hash" in entry


def test_record_to_dict_round_trip() -> None:
    rec = _make_record("rt", status="completed")
    rec.notes.append("everything-fine")
    rec.stage_statuses["stage_09_progress"] = "completed"
    decoded = RunRecord.from_dict(rec.to_dict())
    assert decoded.run_id == rec.run_id
    assert decoded.notes == ["everything-fine"]
    assert decoded.stage_statuses == rec.stage_statuses
