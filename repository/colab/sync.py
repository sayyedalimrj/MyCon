"""Resilient Google Drive sync utilities for long Colab sessions.

On Colab, Google Drive is a FUSE mount. It is mostly transparent but it
is *not* a normal POSIX filesystem:

* Writes can fail transiently (``OSError``/``IOError``) when the FUSE layer
  is busy or the session is briefly throttled.
* After a runtime hiccup the mount can silently go stale — paths under
  ``/content/drive/MyDrive`` still exist as objects but reads/writes error.
* Writing many small files (e.g. a Hugging Face cache or COLMAP database)
  directly onto the FUSE mount is slow.

This module provides a small, dependency-free toolbox to make persistence
robust:

* :func:`atomic_write_text` / :func:`atomic_write_bytes` — temp-file +
  ``os.replace`` so a reader never sees a half-written file, with retries.
* :func:`copy_file` — retrying ``shutil.copy2`` with a temp staging name.
* :func:`mirror_tree` — rsync-style "copy new/changed files" between two
  directories (used to push a fast local scratch cache onto Drive, and to
  pull a cache back from Drive after a reconnect).
* :class:`DriveSyncManager` — health checks + a background daemon that
  periodically mirrors registered local scratch dirs onto Drive so heavy
  caches survive a disconnect even though they are written locally for
  speed.

All functions degrade gracefully when Drive is not mounted (local fallback)
so the same code path works on a laptop and in CI.
"""

from __future__ import annotations

import errno
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

try:  # LogBuffer is optional; sync must work before colab.* is importable.
    from colab.log_capture import LogBuffer
except Exception:  # pragma: no cover
    LogBuffer = None  # type: ignore


def _log(log: Optional["LogBuffer"], message: str) -> None:
    if log is not None:
        try:
            log.append(message)
            return
        except Exception:  # pragma: no cover
            pass
    print(message, flush=True)


# Transient errors we retry on. ENOSPC / EROFS / EDQUOT are not transient.
_FATAL_ERRNOS = {errno.ENOSPC, errno.EROFS, errno.EDQUOT}


def _is_transient(exc: OSError) -> bool:
    return exc.errno not in _FATAL_ERRNOS


def retry(
    func: Callable[[], object],
    *,
    attempts: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    log: Optional["LogBuffer"] = None,
    what: str = "io",
):
    """Run ``func`` with exponential backoff on transient ``OSError``."""
    last: Optional[BaseException] = None
    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except OSError as exc:
            last = exc
            if not _is_transient(exc) or attempt == attempts:
                raise
            _log(
                log,
                f"[sync] transient {what} error (attempt {attempt}/{attempts}): {exc}; "
                f"retrying in {delay:.1f}s",
            )
            time.sleep(delay)
            delay = min(max_delay, delay * 2)
    if last is not None:  # pragma: no cover - defensive
        raise last
    return None


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def atomic_write_bytes(
    path: Path | str,
    data: bytes,
    *,
    attempts: int = 5,
    fsync: bool = True,
    log: Optional["LogBuffer"] = None,
) -> Path:
    path = Path(path)

    def _do() -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            if fsync:
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
        os.replace(tmp, path)
        return path

    return retry(_do, attempts=attempts, log=log, what="write")  # type: ignore[return-value]


def atomic_write_text(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
    attempts: int = 5,
    log: Optional["LogBuffer"] = None,
) -> Path:
    return atomic_write_bytes(path, text.encode(encoding), attempts=attempts, log=log)


def copy_file(
    src: Path | str,
    dst: Path | str,
    *,
    attempts: int = 5,
    log: Optional["LogBuffer"] = None,
) -> Path:
    """Copy a file to ``dst`` atomically (stage to tmp, then replace)."""
    src = Path(src)
    dst = Path(dst)

    def _do() -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.copytmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return dst

    return retry(_do, attempts=attempts, log=log, what="copy")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tree mirroring
# ---------------------------------------------------------------------------


def _needs_copy(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return True
    try:
        s = src.stat()
        d = dst.stat()
    except OSError:
        return True
    if s.st_size != d.st_size:
        return True
    # 2s tolerance: FAT/Drive timestamps have coarse resolution.
    return s.st_mtime > d.st_mtime + 2


@dataclass
class MirrorStats:
    copied: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_copied: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "copied": self.copied,
            "skipped": self.skipped,
            "failed": self.failed,
            "bytes_copied": self.bytes_copied,
        }


def mirror_tree(
    src_dir: Path | str,
    dst_dir: Path | str,
    *,
    log: Optional["LogBuffer"] = None,
    attempts: int = 4,
) -> MirrorStats:
    """Copy new/changed files from ``src_dir`` to ``dst_dir`` (one direction).

    Comparison is by (size, mtime). Existing destination files that are
    already up to date are skipped, so repeated calls are cheap. Failures on
    individual files are counted but never abort the whole mirror.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    stats = MirrorStats()
    if not src_dir.exists():
        return stats
    for root, _dirs, files in os.walk(src_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(src_dir)
        for name in files:
            src = root_path / name
            if name.startswith(".") and name.endswith((".tmp", ".copytmp")):
                continue
            dst = dst_dir / rel_root / name
            if not _needs_copy(src, dst):
                stats.skipped += 1
                continue
            try:
                copy_file(src, dst, attempts=attempts, log=None)
                stats.copied += 1
                try:
                    stats.bytes_copied += src.stat().st_size
                except OSError:
                    pass
            except OSError as exc:
                stats.failed += 1
                _log(log, f"[sync] mirror failed for {src}: {exc}")
    if log is not None and (stats.copied or stats.failed):
        _log(
            log,
            f"[sync] mirror {src_dir} -> {dst_dir}: "
            f"copied={stats.copied} skipped={stats.skipped} failed={stats.failed} "
            f"({stats.bytes_copied / 1024 / 1024:.1f} MB)",
        )
    return stats


# ---------------------------------------------------------------------------
# Health + background daemon
# ---------------------------------------------------------------------------


def verify_writable(directory: Path | str) -> bool:
    """Return True if we can create+delete a probe file inside ``directory``."""
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".sync_probe_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


@dataclass
class DriveSyncManager:
    """Background mirror of fast local scratch dirs onto Drive.

    Register pairs of ``(local_scratch, drive_target)``. The pipeline writes
    heavy caches (HF cache, model cache, COLMAP DB) to the local scratch for
    speed; the manager flushes them onto Drive every ``interval`` seconds and
    once more on :meth:`stop`, so a disconnect loses at most ``interval``
    seconds of cache churn.
    """

    drive_mount: Path
    log: Optional["LogBuffer"] = None
    interval: float = 120.0
    pairs: list[tuple[Path, Path]] = field(default_factory=list)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _remount: Optional[Callable[[], bool]] = field(default=None, init=False)

    def set_remount_callback(self, fn: Callable[[], bool]) -> None:
        self._remount = fn

    def register(self, local_scratch: Path | str, drive_target: Path | str) -> None:
        self.pairs.append((Path(local_scratch), Path(drive_target)))

    def is_mounted(self) -> bool:
        return (Path(self.drive_mount) / "MyDrive").exists()

    def ensure_healthy(self) -> bool:
        """Verify the mount is present & writable; try to remount if not."""
        mydrive = Path(self.drive_mount) / "MyDrive"
        if mydrive.exists() and verify_writable(mydrive):
            return True
        _log(self.log, "[sync] Drive mount looks unhealthy; attempting remount")
        if self._remount is not None:
            try:
                if self._remount():
                    return (Path(self.drive_mount) / "MyDrive").exists()
            except Exception as exc:  # pragma: no cover
                _log(self.log, f"[sync] remount callback failed: {exc}")
        return mydrive.exists()

    def flush(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        if not self.pairs:
            return out
        self.ensure_healthy()
        for local_scratch, drive_target in self.pairs:
            stats = mirror_tree(local_scratch, drive_target, log=self.log)
            out[str(local_scratch)] = stats.as_dict()
        return out

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.flush()
            except Exception as exc:  # pragma: no cover - daemon must not die
                _log(self.log, f"[sync] periodic flush error: {exc}")

    def start_periodic(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="drive-sync", daemon=True
        )
        self._thread.start()
        _log(self.log, f"[sync] background Drive sync started (every {self.interval:.0f}s)")

    def stop(self, *, final_flush: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if final_flush:
            self.flush()
        _log(self.log, "[sync] background Drive sync stopped")
