"""Checks that the delivered Google Colab notebook matches its generator and is safe by default.

These tests do not pretend to run GPU/COLMAP or connect the user's Drive.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from colab._build_notebook import NOTEBOOK
from colab.stage_runner import run_stage, run_stages

pytestmark = pytest.mark.lightweight
ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "MyCon_Colab_Pipeline.ipynb"


def _cell_text(idx: int) -> str:
    return "".join(NOTEBOOK["cells"][idx]["source"])


def test_notebook_is_exact_builder_output_and_has_no_execution_outputs():
    actual = json.loads(NB.read_text(encoding="utf-8"))
    assert actual == NOTEBOOK, "Run colab/_build_notebook.py and commit the generated notebook"
    assert actual["nbformat"] == 4
    assert len(actual["cells"]) >= 20
    for cell in actual["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_all_python_cells_compile_without_installing_dependencies():
    for i, c in enumerate(NOTEBOOK["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"notebook_cell_{i}", "exec")


def test_notebook_targets_validated_integration_branch_and_checks_clone_state():
    clone = _cell_text(4)
    assert "research/mycon-integration-20261112" in clone
    assert "BRANCH   = 'main'" not in clone
    assert "'--ff-only','origin',BRANCH" in clone
    assert "status','--porcelain" in clone
    assert "subprocess.call(" not in clone


def test_no_automatic_force_public_share_or_long_gpu_run():
    assert "PUBLIC_SHARE = False" in _cell_text(10)
    assert "RUN_FULL_PIPELINE = False" in _cell_text(15)
    assert "force=False" in _cell_text(15)
    assert "RUN_ID = 'mycon_site_01'" in _cell_text(8)
    assert "if not PROJECT.on_drive:" in _cell_text(8)
    assert inspect.signature(run_stage).parameters["force"].default is False
    assert inspect.signature(run_stages).parameters["force"].default is False


def test_vlm_configuration_keeps_prior_user_inputs_and_mock_stage_separate():
    vlm = _cell_text(13)
    assert "video_path=VIDEO_PATH" in vlm
    assert "ifc_path=IFC_PATH" in vlm
    assert "schedule_path=SCHEDULE_PATH" in vlm
    assert "user_overrides_yaml=USER_OVERRIDES" in vlm
    assert "config_overrides" in vlm
    assert "Stage 7.5 remains deterministic/mock-only" in vlm


def test_comparison_ready_is_explicit_opt_in_with_no_metric_claim():
    assert "comparison_ready_enabled: true" in _cell_text(12)
    assert "unverified physical scale" in _cell_text(20)
    assert "Cannot safely resume" in inspect.getsource(run_stages)


def test_missing_checkpoint_does_not_silently_replay_heavy_pipeline(monkeypatch, tmp_path):
    import colab.stage_runner as sr
    from colab.log_capture import LogBuffer

    def broken_checkpoint(**kwargs):
        raise ValueError("corrupt metadata")

    monkeypatch.setattr(sr, "CheckpointManager", broken_checkpoint)
    with pytest.raises(RuntimeError, match="Cannot safely resume"):
        sr.run_stages(
            spec_keys=["stage_01_ingest"],
            config_path=tmp_path / "cfg.yaml",
            repo_root=ROOT,
            logs_dir=tmp_path / "logs",
            reports_dir=tmp_path / "reports",
            log=LogBuffer(max_lines=50),
            project_root=tmp_path, run_id="safe-run",
            resume=True,
        )
