from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

from scripts.export_server_handoff_zip import (
    FORBIDDEN_PREFIXES,
    REQUIRED_HANDOFF_FILES,
    collect_handoff_files,
    write_zip,
)


def test_required_handoff_files_exist_in_working_tree() -> None:
    for rel in REQUIRED_HANDOFF_FILES:
        assert Path(rel).exists(), rel


def test_required_handoff_files_are_tracked_by_git() -> None:
    out = subprocess.check_output(
        ["git", "-c", f"safe.directory={Path.cwd()}", "ls-files"],
        text=True,
    )
    tracked = set(out.splitlines())

    missing = [rel for rel in REQUIRED_HANDOFF_FILES if rel not in tracked]
    assert missing == []


def test_collect_handoff_files_includes_required_and_excludes_runtime_dirs() -> None:
    files = collect_handoff_files(Path.cwd())

    for rel in REQUIRED_HANDOFF_FILES:
        assert rel in files

    bad = [rel for rel in files if rel.startswith(FORBIDDEN_PREFIXES)]
    assert bad == []


def test_export_zip_contains_required_files(tmp_path: Path) -> None:
    out = tmp_path / "handoff.zip"
    files = write_zip(Path.cwd(), out)

    assert out.exists()
    assert files

    with zipfile.ZipFile(out, "r") as zf:
        names = set(zf.namelist())

    for rel in REQUIRED_HANDOFF_FILES:
        assert rel in names

    bad = [rel for rel in names if rel.startswith(FORBIDDEN_PREFIXES)]
    assert bad == []
