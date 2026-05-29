"""Portable checkpoint / resume manager for the MyCon Colab pipeline.

The pipeline is a chain of independent subprocess stages whose outputs all
land under a single ``project.root`` directory that, on Colab, lives on
Google Drive. A run can be interrupted at any moment (Colab disconnects,
runtime resets, the user closes the tab, the GPU OOMs). To make long,
unattended runs survivable we persist a single JSON *run-state manifest*
next to the per-run reports and update it atomically after every state
transition.

Design goals
------------
* **Resilient** — every write is atomic (``tmp`` + ``os.replace``) and a
  corrupt/half-written manifest never crashes the resume path.
* **Portable** — the manifest stores only *relative* output paths and a
  content fingerprint of the effective config. A second machine (or a
  different Drive account that received a copy of the project folder) can
  load the same manifest and continue exactly where the first left off,
  because "is this stage already done?" is answered by *inspecting the
  artifacts on disk*, not by trusting an absolute path from another host.
* **Self-validating** — a stage is only treated as resumable-complete when
  it both finished with ``rc == 0`` *and* its declared output globs still
  resolve to real files. This protects against a manifest that says "ok"
  but whose artifacts were partially synced / lost.

This module is intentionally dependency-free (stdlib only) so it imports
and runs on a bare Colab kernel before any ``pip install`` has happened,
and so it is trivially unit-testable on a laptop.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import getpass
import hashlib
import json
import os
import platform
import socket
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

SCHEMA_VERSION = 2
STATE_FILENAME = "run_state.json"

# Stage lifecycle states.
PENDING = "pending"
RUNNING = "running"
OK = "ok"
FAILED = "failed"
SKIPPED = "skipped"

_TERMINAL_OK = {OK, SKIPPED}


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_username() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - exotic environments
        return os.environ.get("USER") or "unknown"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class StageState:
    """Persisted lifecycle record for a single stage."""

    key: str
    status: str = PENDING
    attempts: int = 0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_sec: float = 0.0
    return_code: Optional[int] = None
    error: Optional[str] = None
    # Fingerprints of declared outputs at the moment the stage finished ok.
    outputs: list[dict[str, Any]] = field(default_factory=list)
    # Free-form notes (e.g. "skipped: insufficient anchors").
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageState":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class RunState:
    """Top-level persisted manifest for one pipeline run."""

    run_id: str
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    config_relpath: str = "configs/active.yaml"
    config_fingerprint: str = ""
    repo_commit: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, StageState] = field(default_factory=dict)
    # Append-only audit of session attachments (resume history).
    sessions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config_relpath": self.config_relpath,
            "config_fingerprint": self.config_fingerprint,
            "repo_commit": self.repo_commit,
            "environment": self.environment,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "sessions": self.sessions,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunState":
        stages_raw = data.get("stages") or {}
        stages = {
            key: StageState.from_dict(val)
            for key, val in stages_raw.items()
            if isinstance(val, Mapping)
        }
        return cls(
            run_id=str(data.get("run_id", "")),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            created_at=str(data.get("created_at") or _now_iso()),
            updated_at=str(data.get("updated_at") or _now_iso()),
            config_relpath=str(data.get("config_relpath") or "configs/active.yaml"),
            config_fingerprint=str(data.get("config_fingerprint") or ""),
            repo_commit=str(data.get("repo_commit") or ""),
            environment=dict(data.get("environment") or {}),
            stages=stages,
            sessions=list(data.get("sessions") or []),
        )


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------


def fingerprint_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks. Empty string on error."""
    h = hashlib.sha256()
    try:
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def resolve_output_globs(
    project_root: Path, patterns: Iterable[str]
) -> list[Path]:
    """Resolve declared output globs (relative to project_root) to real files."""
    project_root = Path(project_root)
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for match in sorted(project_root.glob(pattern)):
            if match.is_file() and match not in seen:
                seen.add(match)
                found.append(match)
    return found


def fingerprint_outputs(
    project_root: Path,
    patterns: Iterable[str],
    *,
    hash_max_bytes: int = 64 * 1024 * 1024,
) -> list[dict[str, Any]]:
    """Build a lightweight fingerprint list for a stage's declared outputs.

    Files larger than ``hash_max_bytes`` are fingerprinted by (size, mtime)
    only to avoid re-hashing multi-GB point clouds on every checkpoint.
    """
    project_root = Path(project_root)
    out: list[dict[str, Any]] = []
    for path in resolve_output_globs(project_root, patterns):
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = path.relative_to(project_root).as_posix()
        entry: dict[str, Any] = {
            "path": rel,
            "bytes": int(stat.st_size),
            "mtime": round(stat.st_mtime, 3),
        }
        if stat.st_size <= hash_max_bytes:
            entry["sha256"] = fingerprint_file(path)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Load/update/persist a :class:`RunState` manifest atomically."""

    def __init__(
        self,
        *,
        project_root: Path,
        run_id: str,
        state_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        repo_commit: Optional[str] = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.run_id = run_id
        if state_path is not None:
            self.state_path = Path(state_path)
        else:
            self.state_path = (
                self.project_root / "runs" / run_id / "reports" / STATE_FILENAME
            )
        self._lock = threading.Lock()
        self.state = self._load_or_init(config_path=config_path, repo_commit=repo_commit)

    # ----- load / init -----

    def _load_or_init(
        self, *, config_path: Optional[Path], repo_commit: Optional[str]
    ) -> RunState:
        state: Optional[RunState] = None
        if self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
                state = RunState.from_dict(raw)
            except (OSError, ValueError, json.JSONDecodeError):
                # Corrupt manifest: keep a backup, start fresh.
                self._quarantine_corrupt()
                state = None
        if state is None:
            state = RunState(run_id=self.run_id)

        # Refresh volatile metadata for this attach/session.
        if config_path is not None and Path(config_path).exists():
            try:
                cfg_text = Path(config_path).read_text(encoding="utf-8")
                state.config_fingerprint = fingerprint_text(cfg_text)
            except OSError:
                pass
        if repo_commit:
            state.repo_commit = repo_commit
        state.environment = self._detect_environment()
        state.sessions.append(
            {
                "attached_at": _now_iso(),
                "host": state.environment.get("host", ""),
                "session_id": state.environment.get("colab_session", ""),
            }
        )
        # Trim session history so the manifest cannot grow unbounded.
        if len(state.sessions) > 50:
            state.sessions = state.sessions[-50:]
        return state

    def _quarantine_corrupt(self) -> None:
        try:
            backup = self.state_path.with_suffix(
                self.state_path.suffix + f".corrupt-{_dt.datetime.now():%Y%m%d_%H%M%S}"
            )
            self.state_path.replace(backup)
        except OSError:
            pass

    @staticmethod
    def _detect_environment() -> dict[str, Any]:
        env: dict[str, Any] = {
            "host": socket.gethostname(),
            "user": _safe_username(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "colab_session": os.environ.get("COLAB_RELEASE_TAG", "")
            or os.environ.get("HOSTNAME", ""),
        }
        try:  # GPU name without importing torch.
            import subprocess

            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            env["gpu"] = out.strip().splitlines()[0] if out.strip() else "none"
        except Exception:
            env["gpu"] = "unknown"
        return env

    # ----- persistence -----

    def save(self) -> None:
        with self._lock:
            self.state.updated_at = _now_iso()
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self.state.to_dict(), indent=2, sort_keys=False)
            tmp_fd, tmp_name = tempfile.mkstemp(
                dir=str(self.state_path.parent), prefix=".run_state.", suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                    fh.write(payload)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_name, self.state_path)
            except OSError:
                # Never let a checkpoint write crash the pipeline.
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    # ----- stage transitions -----

    def _stage(self, key: str) -> StageState:
        st = self.state.stages.get(key)
        if st is None:
            st = StageState(key=key)
            self.state.stages[key] = st
        return st

    def get(self, key: str) -> StageState:
        return self._stage(key)

    def mark_running(self, key: str) -> StageState:
        st = self._stage(key)
        st.status = RUNNING
        st.attempts += 1
        st.started_at = _now_iso()
        st.finished_at = None
        st.error = None
        self.save()
        return st

    def mark_ok(
        self,
        key: str,
        *,
        duration_sec: float,
        return_code: int = 0,
        output_patterns: Optional[Iterable[str]] = None,
        note: str = "",
    ) -> StageState:
        st = self._stage(key)
        st.status = OK
        st.finished_at = _now_iso()
        st.duration_sec = float(duration_sec)
        st.return_code = int(return_code)
        st.error = None
        st.note = note
        if output_patterns is not None:
            st.outputs = fingerprint_outputs(self.project_root, output_patterns)
        self.save()
        return st

    def mark_failed(
        self,
        key: str,
        *,
        duration_sec: float,
        return_code: int,
        error: str = "",
    ) -> StageState:
        st = self._stage(key)
        st.status = FAILED
        st.finished_at = _now_iso()
        st.duration_sec = float(duration_sec)
        st.return_code = int(return_code)
        st.error = error[:2000] if error else None
        self.save()
        return st

    def mark_skipped(self, key: str, *, note: str = "") -> StageState:
        st = self._stage(key)
        st.status = SKIPPED
        st.finished_at = _now_iso()
        st.note = note
        self.save()
        return st

    # ----- resume logic -----

    def is_complete(
        self,
        key: str,
        *,
        output_patterns: Optional[Iterable[str]] = None,
        verify_outputs: bool = True,
    ) -> bool:
        """Return True if ``key`` can be safely skipped on resume.

        A stage counts as complete when its recorded status is terminal-ok
        AND (when ``verify_outputs`` and the stage declares output globs) at
        least one declared output still resolves to a real file on disk.
        """
        st = self.state.stages.get(key)
        if st is None or st.status not in _TERMINAL_OK:
            return False
        if st.status == SKIPPED:
            # Skipped stages have no artifacts to verify.
            return True
        if not verify_outputs or not output_patterns:
            return True
        patterns = list(output_patterns)
        if not patterns:
            return True
        return bool(resolve_output_globs(self.project_root, patterns))

    def plan_resume(
        self,
        requested_keys: Iterable[str],
        *,
        outputs_for: Optional["OutputResolver"] = None,
        resume: bool = True,
    ) -> "ResumePlan":
        """Split requested stages into (to_skip_complete, to_run)."""
        to_run: list[str] = []
        to_skip: list[str] = []
        for key in requested_keys:
            patterns = outputs_for(key) if outputs_for is not None else None
            if resume and self.is_complete(key, output_patterns=patterns):
                to_skip.append(key)
            else:
                to_run.append(key)
        return ResumePlan(to_run=to_run, to_skip=to_skip)

    # ----- reporting -----

    def status_map(self) -> dict[str, str]:
        return {k: v.status for k, v in self.state.stages.items()}

    def summary_rows(self) -> list[list[str]]:
        rows: list[list[str]] = []
        for key, st in self.state.stages.items():
            rows.append(
                [
                    key,
                    st.status,
                    str(st.attempts),
                    f"{st.duration_sec:.1f}s" if st.duration_sec else "-",
                    st.note or (st.error or ""),
                ]
            )
        return rows


# A callable that maps a stage key to its declared output glob patterns.
from typing import Callable  # noqa: E402  (kept here for readability)

OutputResolver = Callable[[str], Optional[list[str]]]


@dataclass
class ResumePlan:
    to_run: list[str]
    to_skip: list[str]

    def describe(self) -> str:
        lines = [f"resume plan: {len(self.to_run)} to run, {len(self.to_skip)} already complete"]
        if self.to_skip:
            lines.append("  skip (complete): " + ", ".join(self.to_skip))
        if self.to_run:
            lines.append("  run            : " + ", ".join(self.to_run))
        return "\n".join(lines)
