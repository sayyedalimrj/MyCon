"""Run-history persistence.

The service-layer scope says runs must outlive the API process: an operator
who restarts ``uvicorn`` should still see every previously-executed run in
the GUI's run list. This module persists run metadata and event logs to
disk under ``runs/_service/`` so that requirement holds without a database.

On-disk layout
--------------

::

    runs/_service/
      run_history.json       # one JSON-array of RunRecord summaries
      events/
        <run_id>.jsonl       # newline-delimited RunEvent.to_dict() per run

The ``run_history.json`` file is rewritten atomically (``.tmp`` →
``os.replace``) on every update. Per-run event logs are append-only.

Why a flat JSON file
--------------------

Every operator workstation already has hundreds of YAML configs and gigabytes
of dense MVS output sitting on disk; adding a SQLite database for ~50 runs
would be over-engineering. JSON-on-disk is human-inspectable, trivially
backed up, and the resulting code is small enough to read in one screen.

Public API
----------

- :class:`RunRecord` — structured per-run metadata.
- :class:`RunHistoryStore` — file-backed store; thread-safe.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from pipeline.common.paths import atomic_output_path

from pipeline.service.events import RunEvent, RunEventKind


__all__ = [
    "RunRecord",
    "RunHistoryStore",
]


@dataclass
class RunRecord:
    """Persistent metadata for a run.

    The record carries enough information for the future GUI's run list to
    render a useful overview: who, what, when, with which config, and the
    result. Per-stage details live in the event log; this is the per-run
    summary.
    """

    run_id: str
    project_name: str
    config_path: str
    config_hash: str
    requested_stages: list[str]
    submitted_at_unix: float = field(default_factory=time.time)
    started_at_unix: float | None = None
    finished_at_unix: float | None = None
    status: str = "queued"
    """One of: ``queued``, ``running``, ``completed``, ``failed``,
    ``cancelled``."""

    stage_statuses: dict[str, str] = field(default_factory=dict)
    """Last known per-stage status keyed by stage name."""

    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunRecord":
        return cls(
            run_id=str(data["run_id"]),
            project_name=str(data["project_name"]),
            config_path=str(data["config_path"]),
            config_hash=str(data["config_hash"]),
            requested_stages=list(data.get("requested_stages") or []),
            submitted_at_unix=float(data.get("submitted_at_unix") or time.time()),
            started_at_unix=(float(data["started_at_unix"]) if data.get("started_at_unix") is not None else None),
            finished_at_unix=(float(data["finished_at_unix"]) if data.get("finished_at_unix") is not None else None),
            status=str(data.get("status") or "queued"),
            stage_statuses=dict(data.get("stage_statuses") or {}),
            notes=list(data.get("notes") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "project_name": self.project_name,
            "config_path": self.config_path,
            "config_hash": self.config_hash,
            "requested_stages": list(self.requested_stages),
            "submitted_at_unix": self.submitted_at_unix,
            "started_at_unix": self.started_at_unix,
            "finished_at_unix": self.finished_at_unix,
            "status": self.status,
            "stage_statuses": dict(self.stage_statuses),
            "notes": list(self.notes),
        }


class RunHistoryStore:
    """Thread-safe, file-backed run-history persistence.

    The store holds the in-memory canonical copy of ``run_history.json``;
    every mutation updates both the in-memory copy and disk under a
    single :class:`threading.Lock`. Concurrent reads from the API layer
    are served from memory.
    """

    HISTORY_FILENAME: str = "run_history.json"
    EVENTS_SUBDIR: str = "events"

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = Path(base_dir)
        self._history_path = self._base_dir / self.HISTORY_FILENAME
        self._events_dir = self._base_dir / self.EVENTS_SUBDIR
        self._lock = threading.Lock()
        self._records: dict[str, RunRecord] = {}
        self._load_from_disk()

    # ---- record CRUD ----

    def save(self, record: RunRecord) -> None:
        """Insert or update a record and flush to disk."""
        with self._lock:
            self._records[record.run_id] = record
            self._flush_locked()

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._records.get(run_id)

    def list(self, *, limit: int | None = None) -> list[RunRecord]:
        """Return records sorted by submitted_at_unix descending (newest first)."""
        with self._lock:
            records = sorted(self._records.values(), key=lambda r: r.submitted_at_unix, reverse=True)
        if limit is not None:
            return records[:limit]
        return records

    def delete(self, run_id: str) -> bool:
        with self._lock:
            existed = self._records.pop(run_id, None) is not None
            self._flush_locked()
        events_path = self._events_dir / f"{run_id}.jsonl"
        try:
            events_path.unlink(missing_ok=True)
        except OSError:
            pass
        return existed

    # ---- event log persistence ----

    def append_events(self, run_id: str, events: Iterable[RunEvent]) -> int:
        """Append events for one run to its events file. Returns the number written."""
        self._events_dir.mkdir(parents=True, exist_ok=True)
        path = self._events_dir / f"{run_id}.jsonl"
        count = 0
        with path.open("a", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev.to_dict()) + "\n")
                count += 1
        return count

    def read_events(self, run_id: str) -> list[dict[str, Any]]:
        """Read the persisted event log for one run.

        Returns an empty list if the run has no events file (either because
        the run never produced any, or because it has been deleted).
        """
        path = self._events_dir / f"{run_id}.jsonl"
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # Tolerate a single corrupted line without losing the
                    # rest of the log; persistence corruption is rare and
                    # the GUI is more useful with partial history than no
                    # history.
                    continue
        return out

    # ---- internals ----

    def _load_from_disk(self) -> None:
        if not self._history_path.exists():
            return
        try:
            data = json.loads(self._history_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, list):
            return
        with self._lock:
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                try:
                    rec = RunRecord.from_dict(entry)
                except (KeyError, TypeError, ValueError):
                    continue
                self._records[rec.run_id] = rec

    def _flush_locked(self) -> None:
        """Persist the in-memory records atomically. Caller holds ``self._lock``."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        ordered = sorted(self._records.values(), key=lambda r: r.submitted_at_unix, reverse=True)
        payload = [rec.to_dict() for rec in ordered]
        with atomic_output_path(self._history_path) as tmp:
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
