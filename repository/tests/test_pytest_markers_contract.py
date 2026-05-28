from __future__ import annotations

from pathlib import Path


def test_pytest_ini_defines_expected_markers() -> None:
    text = Path("pytest.ini").read_text(encoding="utf-8")
    for marker in ["lightweight", "geometry", "server", "vlm", "colmap", "slow"]:
        assert f"{marker}:" in text
