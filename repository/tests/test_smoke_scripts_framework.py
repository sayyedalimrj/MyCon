
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


SMOKE_CASES = [
    ("stage_04_5_cams_gs", "scripts/smoke_test_stage_04_5_cams_gs.py", "STAGE_04_5_SMOKE_OK"),
    ("stage_07_cleanup", "scripts/smoke_test_stage_07.py", "STAGE_07_SMOKE_OK"),
    ("stage_07_5_vlm_qa", "scripts/smoke_test_stage_07_5_vlm_qa.py", "STAGE_07_5_SMOKE_OK"),
    ("stage_07_6_viewer_export", "scripts/smoke_test_stage_07_6_viewer_export.py", "STAGE_07_6_SMOKE_OK"),
    ("stage_07_7_cams_gs_evidence", "scripts/smoke_test_stage_07_7_cams_gs_evidence.py", "STAGE_07_7_SMOKE_OK"),
    ("stage_08_bim_eval", "scripts/smoke_test_stage_08.py", "STAGE_08_SMOKE_OK"),
    ("stage_09_progress", "scripts/smoke_test_stage_09.py", "STAGE_09_SMOKE_OK"),
    ("stage_10_copilot", "scripts/smoke_test_stage_10.py", "STAGE_10_SMOKE_OK"),
    ("stage_11_schedule_variance", "scripts/smoke_test_stage_11.py", "STAGE_11_SMOKE_OK"),
]


@pytest.mark.parametrize(("name", "script_path", "expected_marker"), SMOKE_CASES)
def test_lightweight_smoke_script(name: str, script_path: str, expected_marker: str) -> None:
    script = ROOT / script_path
    assert script.exists(), f"Missing smoke script for {name}: {script}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert expected_marker in result.stdout, result.stdout
