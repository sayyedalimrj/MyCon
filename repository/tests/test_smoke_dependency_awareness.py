from __future__ import annotations

from pathlib import Path


SMOKE_SCRIPTS = [
    Path("scripts/smoke_test_stage_07.py"),
    Path("scripts/smoke_test_stage_07_5_vlm_qa.py"),
    Path("scripts/smoke_test_stage_08.py"),
    Path("scripts/smoke_test_stage_09.py"),
]


def test_geometry_smoke_scripts_are_dependency_aware() -> None:
    for path in SMOKE_SCRIPTS:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        if "open3d" in text or "ifcopenshell" in text:
            assert "SMOKE_SKIP_MISSING_DEPENDENCY" in text
            assert "raise SystemExit(0)" in text
