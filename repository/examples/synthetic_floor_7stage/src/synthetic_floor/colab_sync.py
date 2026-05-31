"""Google Drive integration for the synthetic_floor GPU pipeline on Colab.

This makes the example's outputs survive Colab disconnects / runtime resets
and lets a run be resumed from another device or Drive account:

* :func:`maybe_mount_drive` mounts Google Drive (idempotent; safe off-Colab).
* :func:`default_drive_root` returns the canonical on-Drive folder for a run.
* :class:`DriveMirror` keeps a local working ``output/`` tree mirrored onto
  Drive — it ``pull()``s any prior outputs back before a resume, ``push()``es
  after every stage, and runs a background daemon so partial progress within a
  long stage also reaches Drive.

Whenever the repository's resilient ``colab`` package is importable (it is in
the Colab notebook, which clones the repo) we reuse its atomic-write + retry +
remount-recovery sync layer. Otherwise we fall back to a small self-contained
``shutil`` mirror so the example also works standalone.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

DEFAULT_DRIVE_MOUNT = Path("/content/drive")
DEFAULT_DRIVE_BASE = "MyDrive/MyCon_Colab/synthetic_floor_7stage"


def _repo_root() -> Path:
    # .../examples/synthetic_floor_7stage/src/synthetic_floor/colab_sync.py
    return Path(__file__).resolve().parents[4]


def _import_repo_colab():
    """Import the repo-level resilient colab.sync / colab.drive if available."""
    try:
        root = _repo_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from colab import drive as colab_drive  # type: ignore
        from colab import sync as colab_sync  # type: ignore

        return colab_sync, colab_drive
    except Exception:
        return None, None


_COLAB_SYNC, _COLAB_DRIVE = _import_repo_colab()


def _log(log: Optional[Callable[[str], None]], msg: str) -> None:
    if log is not None:
        try:
            log(msg)
            return
        except Exception:
            pass
    print(msg, flush=True)


# ---------------------------------------------------------------------
# Mount
# ---------------------------------------------------------------------


def drive_mounted(mount: Path = DEFAULT_DRIVE_MOUNT) -> bool:
    return (Path(mount) / "MyDrive").exists()


def maybe_mount_drive(
    *, mount: Path = DEFAULT_DRIVE_MOUNT, log: Optional[Callable[[str], None]] = None
) -> bool:
    """Mount Google Drive when running on Colab. Returns True if available."""
    if drive_mounted(mount):
        _log(log, f"[drive] already mounted at {mount}")
        return True
    if _COLAB_DRIVE is not None:
        try:
            ok = _COLAB_DRIVE.mount_drive(mount_point=Path(mount))
            if ok:
                return True
        except Exception as exc:  # pragma: no cover
            _log(log, f"[drive] repo mount helper failed: {exc}")
    try:
        from google.colab import drive as gdrive  # type: ignore

        gdrive.mount(str(mount), force_remount=False)
        return drive_mounted(mount)
    except Exception as exc:  # pragma: no cover - not on Colab
        _log(log, f"[drive] not on Colab / mount unavailable ({exc})")
        return False


def remount_drive(
    *, mount: Path = DEFAULT_DRIVE_MOUNT, log: Optional[Callable[[str], None]] = None
) -> bool:
    if _COLAB_DRIVE is not None:
        try:
            return bool(_COLAB_DRIVE.remount_drive(mount_point=Path(mount)))
        except Exception:
            pass
    try:
        from google.colab import drive as gdrive  # type: ignore

        gdrive.mount(str(mount), force_remount=True)
        return drive_mounted(mount)
    except Exception:
        return False


def default_drive_root(
    run_id: str,
    *,
    mount: Path = DEFAULT_DRIVE_MOUNT,
    base: str = DEFAULT_DRIVE_BASE,
) -> Path:
    safe = "".join(c for c in str(run_id) if c.isalnum() or c in ("-", "_", ".")) or "run"
    return Path(mount) / base / safe


# ---------------------------------------------------------------------
# Mirror
# ---------------------------------------------------------------------


def _file_sha1(path: Path, *, block: int = 1024 * 1024) -> str:
    import hashlib
    h = hashlib.sha1()
    try:
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(block), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _needs_copy(src: Path, dst: Path, *, use_hash: bool = False) -> bool:
    """Delta check: copy only when content actually differs.

    Fast path: missing dst, or different size -> copy. Same size + close mtime
    -> skip. Same size but mtime drift (the common cross-system case after a
    Drive download resets timestamps) -> verify by SHA-1 and copy only if the
    content truly differs, so resuming on a new machine transfers nothing it
    already has.
    """
    if not dst.exists():
        return True
    try:
        s, d = src.stat(), dst.stat()
    except OSError:
        return True
    if s.st_size != d.st_size:
        return True
    if abs(s.st_mtime - d.st_mtime) <= 2:
        return False
    if not use_hash:
        return s.st_mtime > d.st_mtime + 2
    # Size matches but timestamps drifted -> compare content hashes.
    return _file_sha1(src) != _file_sha1(dst)


def _builtin_mirror(src_dir: Path, dst_dir: Path, log, *, use_hash: bool = False) -> dict:
    copied = skipped = failed = hashed = 0
    bytes_copied = 0
    src_dir = Path(src_dir)
    if not src_dir.exists():
        return {"copied": 0, "skipped": 0, "failed": 0, "bytes_copied": 0}
    for root, _dirs, files in os.walk(src_dir):
        rel = Path(root).relative_to(src_dir)
        for name in files:
            if name.startswith(".") and name.endswith(".tmp"):
                continue
            s = Path(root) / name
            d = Path(dst_dir) / rel / name
            try:
                need = _needs_copy(s, d, use_hash=use_hash)
            except OSError:
                need = True
            if use_hash and d.exists():
                hashed += 1
            if not need:
                skipped += 1
                continue
            try:
                d.parent.mkdir(parents=True, exist_ok=True)
                tmp = d.with_name(f".{d.name}.{os.getpid()}.tmp")
                shutil.copy2(s, tmp)
                os.replace(tmp, d)
                copied += 1
                try:
                    bytes_copied += s.stat().st_size
                except OSError:
                    pass
            except OSError as exc:
                failed += 1
                _log(log, f"[drive] mirror failed for {s}: {exc}")
    if copied or failed:
        _log(log, f"[drive] delta-mirror {src_dir} -> {dst_dir}: "
                  f"copied={copied} skipped={skipped} failed={failed} "
                  f"({bytes_copied / 1024 / 1024:.1f} MB)")
    return {"copied": copied, "skipped": skipped, "failed": failed,
            "bytes_copied": bytes_copied}


def mirror_tree(src_dir: Path, dst_dir: Path, *, log=None, use_hash: bool = False) -> dict:
    """Incrementally copy new/modified files from ``src_dir`` to ``dst_dir``.

    ``use_hash=True`` performs a content-hash delta (only genuinely changed
    files transfer, even when timestamps drift on a new system). ``use_hash=False``
    uses the fast size+mtime delta (good for same-machine periodic sync).
    """
    if use_hash:
        # Hash-verified delta is implemented locally (the repo mirror is mtime-based).
        return _builtin_mirror(Path(src_dir), Path(dst_dir), log, use_hash=True)
    if _COLAB_SYNC is not None:
        try:
            stats = _COLAB_SYNC.mirror_tree(Path(src_dir), Path(dst_dir))
            return stats.as_dict() if hasattr(stats, "as_dict") else dict(stats)
        except Exception as exc:  # pragma: no cover
            _log(log, f"[drive] repo mirror failed ({exc}); using builtin")
    return _builtin_mirror(Path(src_dir), Path(dst_dir), log, use_hash=False)


class DriveMirror:
    """Keep a local working tree mirrored to a Drive folder, both directions."""

    def __init__(
        self,
        local_root: Path,
        drive_root: Path,
        *,
        log: Optional[Callable[[str], None]] = None,
        interval: float = 120.0,
        mount: Path = DEFAULT_DRIVE_MOUNT,
        use_hash: bool = True,
    ) -> None:
        self.local_root = Path(local_root)
        self.drive_root = Path(drive_root)
        self.log = log
        self.interval = interval
        self.mount = Path(mount)
        self.use_hash = use_hash
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ----- health -----

    def ensure_healthy(self) -> bool:
        if (self.mount / "MyDrive").exists():
            return True
        _log(self.log, "[drive] mount unhealthy; attempting remount")
        return remount_drive(mount=self.mount, log=self.log)

    # ----- one-shot transfers -----

    def pull(self) -> dict:
        """Restore prior outputs from Drive into the local tree (hash-verified delta)."""
        self.ensure_healthy()
        self.local_root.mkdir(parents=True, exist_ok=True)
        # Cross-system pulls: content-hash delta so identical files are skipped
        # even though their timestamps were reset by the Drive download.
        return mirror_tree(self.drive_root, self.local_root, log=self.log, use_hash=self.use_hash)

    def push(self) -> dict:
        """Flush the local tree onto Drive (fast size+mtime delta)."""
        self.ensure_healthy()
        self.drive_root.mkdir(parents=True, exist_ok=True)
        return mirror_tree(self.local_root, self.drive_root, log=self.log, use_hash=False)

    # ----- background daemon -----

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.push()
            except Exception as exc:  # pragma: no cover - daemon must not die
                _log(self.log, f"[drive] periodic push error: {exc}")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="syntheticfloor-drive-sync", daemon=True)
        self._thread.start()
        _log(self.log, f"[drive] background sync started (every {self.interval:.0f}s) -> {self.drive_root}")

    def stop(self, *, final_push: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if final_push:
            self.push()
        _log(self.log, "[drive] background sync stopped")
