from __future__ import annotations

import zipfile
from pathlib import Path

from scripts.export_server_handoff_zip import REQUIRED_HANDOFF_FILES, write_zip
from scripts.verify_server_handoff_zip import verify_server_handoff_zip


def test_verify_official_exported_zip_passes(tmp_path: Path) -> None:
    out = tmp_path / "handoff.zip"
    write_zip(Path.cwd(), out)

    result = verify_server_handoff_zip(out)

    assert result.passed is True
    assert result.status == "ok"
    assert result.file_count > 0
    assert result.size_bytes > 0
    assert len(result.sha256) == 64
    assert result.missing_required == []
    assert result.forbidden_entries == []


def test_verify_zip_fails_when_required_file_missing(tmp_path: Path) -> None:
    out = tmp_path / "bad.zip"

    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("README.md", "demo")

    result = verify_server_handoff_zip(out)

    assert result.passed is False
    assert result.status == "failed"
    assert "requirements-core.txt" in result.missing_required


def test_verify_zip_fails_when_runtime_data_included(tmp_path: Path) -> None:
    out = tmp_path / "bad_runtime.zip"

    with zipfile.ZipFile(out, "w") as zf:
        for required in REQUIRED_HANDOFF_FILES:
            zf.writestr(required, "x")

        zf.writestr("data/raw/site01/video.mp4", "fake")

    result = verify_server_handoff_zip(out)

    assert result.passed is False
    assert result.status == "failed"
    assert "data/raw/site01/video.mp4" in result.forbidden_entries
