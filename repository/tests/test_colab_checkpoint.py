"""Lightweight tests for the Colab checkpoint/resume manager.

These deliberately avoid any heavy dependency (torch/cv2/gradio) so they run
in the default laptop-safe suite. The checkpoint module is stdlib-only.
"""

from __future__ import annotations

import json
from pathlib import Path

from colab import checkpoint as cp


def _touch(path: Path, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_mark_running_then_ok_persists_atomically(tmp_path: Path) -> None:
    mgr = cp.CheckpointManager(project_root=tmp_path, run_id="run1")
    mgr.mark_running("stage_01_ingest")
    _touch(tmp_path / "data" / "normalized" / "v.mp4")
    st = mgr.mark_ok(
        "stage_01_ingest",
        duration_sec=2.5,
        output_patterns=["data/normalized/*.mp4"],
    )
    assert st.status == cp.OK
    assert st.attempts == 1
    assert st.outputs and st.outputs[0]["path"] == "data/normalized/v.mp4"

    # The on-disk manifest is valid JSON and round-trips.
    raw = json.loads(mgr.state_path.read_text(encoding="utf-8"))
    assert raw["run_id"] == "run1"
    assert raw["stages"]["stage_01_ingest"]["status"] == "ok"


def test_is_complete_requires_outputs_to_exist(tmp_path: Path) -> None:
    mgr = cp.CheckpointManager(project_root=tmp_path, run_id="run1")
    _touch(tmp_path / "data" / "normalized" / "v.mp4")
    mgr.mark_ok("stage_01_ingest", duration_sec=1.0, output_patterns=["data/normalized/*.mp4"])
    assert mgr.is_complete("stage_01_ingest", output_patterns=["data/normalized/*.mp4"])

    # Remove the artifact -> no longer considered complete (lost on Drive).
    (tmp_path / "data" / "normalized" / "v.mp4").unlink()
    assert not mgr.is_complete("stage_01_ingest", output_patterns=["data/normalized/*.mp4"])


def test_skipped_stage_is_complete_without_outputs(tmp_path: Path) -> None:
    mgr = cp.CheckpointManager(project_root=tmp_path, run_id="run1")
    mgr.mark_skipped("stage_08_bim_registration", note="no_bim")
    assert mgr.is_complete(
        "stage_08_bim_registration", output_patterns=["runs/**/reports/registration_report.json"]
    )


def test_failed_stage_is_not_complete(tmp_path: Path) -> None:
    mgr = cp.CheckpointManager(project_root=tmp_path, run_id="run1")
    mgr.mark_failed("stage_05_dense", duration_sec=3.0, return_code=1, error="oom")
    assert not mgr.is_complete("stage_05_dense", output_patterns=["data/dense/**/fused.ply"])


def test_resume_is_portable_across_manager_instances(tmp_path: Path) -> None:
    # First "session" completes a stage.
    mgr1 = cp.CheckpointManager(project_root=tmp_path, run_id="run1")
    _touch(tmp_path / "data" / "frames" / "key" / "m_manifest.csv")
    mgr1.mark_ok("stage_02_keyframes", duration_sec=1.0, output_patterns=["data/frames/key/*manifest*.csv"])

    # A new "session" (e.g. after a disconnect / on another device) loads the
    # same on-disk state and plans a resume.
    mgr2 = cp.CheckpointManager(project_root=tmp_path, run_id="run1")
    plan = mgr2.plan_resume(
        ["stage_02_keyframes", "stage_03_colmap"],
        outputs_for=lambda k: {
            "stage_02_keyframes": ["data/frames/key/*manifest*.csv"],
            "stage_03_colmap": ["data/sparse/**/*.bin"],
        }.get(k),
    )
    assert plan.to_skip == ["stage_02_keyframes"]
    assert plan.to_run == ["stage_03_colmap"]


def test_corrupt_manifest_is_quarantined_not_fatal(tmp_path: Path) -> None:
    state_path = tmp_path / "runs" / "run1" / "reports" / "run_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{not valid json", encoding="utf-8")

    # Constructing a manager over a corrupt manifest must not raise; it should
    # quarantine the bad file and start fresh.
    mgr = cp.CheckpointManager(project_root=tmp_path, run_id="run1", state_path=state_path)
    assert mgr.state.run_id == "run1"
    quarantined = list(state_path.parent.glob("run_state.json.corrupt-*"))
    assert quarantined, "corrupt manifest should be quarantined"


def test_attempts_increment_on_repeated_running(tmp_path: Path) -> None:
    mgr = cp.CheckpointManager(project_root=tmp_path, run_id="run1")
    mgr.mark_running("stage_05_dense")
    mgr.mark_running("stage_05_dense")
    assert mgr.get("stage_05_dense").attempts == 2


def test_config_fingerprint_recorded(tmp_path: Path) -> None:
    cfg = tmp_path / "active.yaml"
    cfg.write_text("project:\n  run_id: run1\n", encoding="utf-8")
    mgr = cp.CheckpointManager(project_root=tmp_path, run_id="run1", config_path=cfg)
    assert mgr.state.config_fingerprint == cp.fingerprint_text(cfg.read_text(encoding="utf-8"))
