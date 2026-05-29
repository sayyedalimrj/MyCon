"""Lightweight tests for the resilient Drive sync layer (stdlib-only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from colab import sync


def test_atomic_write_text_creates_parents_and_content(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c.txt"
    sync.atomic_write_text(target, "hello world")
    assert target.read_text(encoding="utf-8") == "hello world"
    # No stray temp files left behind.
    assert not list(target.parent.glob(".c.txt*"))


def test_atomic_write_is_all_or_nothing_on_repeated_writes(tmp_path: Path) -> None:
    target = tmp_path / "x.bin"
    sync.atomic_write_bytes(target, b"first")
    sync.atomic_write_bytes(target, b"second-longer")
    assert target.read_bytes() == b"second-longer"


def test_copy_file_atomic(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dst = tmp_path / "nested" / "dst.bin"
    sync.copy_file(src, dst)
    assert dst.read_bytes() == b"payload"


def test_mirror_tree_copies_then_skips_unchanged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "a.txt").write_text("a", encoding="utf-8")
    (src / "b.txt").write_text("b", encoding="utf-8")

    first = sync.mirror_tree(src, dst)
    assert first.copied == 2
    assert (dst / "sub" / "a.txt").read_text(encoding="utf-8") == "a"

    second = sync.mirror_tree(src, dst)
    assert second.copied == 0
    assert second.skipped == 2


def test_mirror_tree_detects_changed_size(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    f = src / "f.txt"
    f.write_text("short", encoding="utf-8")
    sync.mirror_tree(src, dst)
    f.write_text("a much longer content than before", encoding="utf-8")
    stats = sync.mirror_tree(src, dst)
    assert stats.copied == 1
    assert (dst / "f.txt").read_text(encoding="utf-8") == "a much longer content than before"


def test_mirror_tree_missing_source_is_noop(tmp_path: Path) -> None:
    stats = sync.mirror_tree(tmp_path / "nope", tmp_path / "dst")
    assert stats.copied == 0 and stats.skipped == 0


def test_verify_writable(tmp_path: Path) -> None:
    assert sync.verify_writable(tmp_path) is True
    assert sync.verify_writable("/proc/this/should/not/be/writable") is False


def test_retry_reraises_fatal_enospc(monkeypatch) -> None:
    import errno

    calls = {"n": 0}

    def _boom():
        calls["n"] += 1
        raise OSError(errno.ENOSPC, "no space")

    with pytest.raises(OSError):
        sync.retry(_boom, attempts=4, base_delay=0.0)
    # ENOSPC is fatal -> must not retry.
    assert calls["n"] == 1


def test_retry_succeeds_after_transient(monkeypatch) -> None:
    import errno

    state = {"n": 0}

    def _flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise OSError(errno.EIO, "io")
        return "ok"

    monkeypatch.setattr(sync.time, "sleep", lambda *_a, **_k: None)
    assert sync.retry(_flaky, attempts=5, base_delay=0.0) == "ok"
    assert state["n"] == 3


def test_drive_sync_manager_flush_mirrors_pairs(tmp_path: Path) -> None:
    # Simulate a mounted Drive with a MyDrive directory.
    mount = tmp_path / "drive"
    (mount / "MyDrive").mkdir(parents=True)
    local = tmp_path / "local"
    (local).mkdir()
    (local / "model.bin").write_bytes(b"weights")
    drive_target = mount / "MyDrive" / "cache"

    mgr = sync.DriveSyncManager(drive_mount=mount, interval=999)
    mgr.register(local, drive_target)
    out = mgr.flush()
    assert (drive_target / "model.bin").read_bytes() == b"weights"
    assert out[str(local)]["copied"] == 1


def test_drive_sync_manager_unhealthy_triggers_remount_callback(tmp_path: Path) -> None:
    mount = tmp_path / "drive"  # MyDrive does NOT exist -> unhealthy
    called = {"n": 0}

    def _remount() -> bool:
        called["n"] += 1
        (mount / "MyDrive").mkdir(parents=True, exist_ok=True)
        return True

    mgr = sync.DriveSyncManager(drive_mount=mount, interval=999)
    mgr.set_remount_callback(_remount)
    assert mgr.ensure_healthy() is True
    assert called["n"] == 1
