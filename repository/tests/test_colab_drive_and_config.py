"""Lightweight tests for colab.drive (local fallback) and colab.config_manager.

No google.colab, no GPU, no heavy deps — exercises the laptop/CI code paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from colab import config_manager as cm
from colab import drive


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# drive
# ---------------------------------------------------------------------------


def test_sanitize_run_id() -> None:
    assert drive.sanitize_run_id("  Site A/2026  ") == "SiteA2026"
    assert drive.sanitize_run_id("a-b_c.1") == "a-b_c.1"
    with pytest.raises(ValueError):
        drive.sanitize_run_id("   ")
    with pytest.raises(ValueError):
        drive.sanitize_run_id("///")


def test_setup_project_tree_local_fallback(tmp_path: Path) -> None:
    paths = drive.setup_project_tree(
        run_id="run_x",
        drive_mount=tmp_path / "no_mount",
        fallback_root=tmp_path / "fallback",
        local_scratch=tmp_path / "scratch",
        auto_mount=False,
    )
    assert paths.on_drive is False
    assert paths.project_root.exists()
    assert paths.run_state_path.parent.exists()
    assert paths.model_cache_dir.exists()
    assert paths.hf_cache_dir.exists()
    assert paths.local_hf_cache_dir.exists()
    # The pipeline data tree is materialised.
    for sub in ("data/normalized", "data/frames/key", "data/dense", "exports/viewer"):
        assert (paths.project_root / sub).is_dir()


def test_drive_health_false_when_not_mounted(tmp_path: Path) -> None:
    assert drive.is_mounted(tmp_path) is False
    assert drive.drive_health(tmp_path) is False


def test_stage_upload_to_drive_copies(tmp_path: Path) -> None:
    src = tmp_path / "video.mp4"
    src.write_bytes(b"data")
    dest_dir = tmp_path / "uploads"
    out = drive.stage_upload_to_drive(src=src, dest_dir=dest_dir)
    assert out.read_bytes() == b"data"
    assert out.parent == dest_dir


# ---------------------------------------------------------------------------
# config_manager profiles
# ---------------------------------------------------------------------------


def test_profiles_exist_and_resolve() -> None:
    assert set(cm.PROFILES) == {"colab_safe", "colab_gpu", "production"}
    assert cm.DEFAULT_PROFILE in cm.PROFILES
    safe = cm.profile_overrides("colab_safe")
    assert safe["copilot.vlm.provider"] == "mock"
    with pytest.raises(ValueError):
        cm.profile_overrides("does_not_exist")


def test_build_effective_config_applies_profile(tmp_path: Path) -> None:
    data = cm.build_effective_config(
        repo_root=REPO_ROOT,
        project_root=tmp_path / "proj",
        run_id="run1",
        profile="colab_safe",
    )
    assert data["project"]["run_id"] == "run1"
    assert str(data["project"]["root"]).endswith("proj")
    assert data["dense"]["max_image_size"] == 1024
    assert data["copilot"]["vlm"]["provider"] == "mock"


def test_override_dict_beats_profile_and_user_yaml_beats_all(tmp_path: Path) -> None:
    data = cm.build_effective_config(
        repo_root=REPO_ROOT,
        project_root=tmp_path / "proj",
        run_id="run1",
        profile="colab_gpu",
        override_dict={
            "copilot.vlm.provider": "ollama_local",
            "dense.max_image_size": 1200,
        },
        user_overrides_yaml="dense:\n  max_image_size: 640\n",
    )
    # override_dict switches the VLM provider...
    assert data["copilot"]["vlm"]["provider"] == "ollama_local"
    # ...but the user YAML has the final word on a shared key.
    assert data["dense"]["max_image_size"] == 640


def test_apply_safe_overrides_backward_compat(tmp_path: Path) -> None:
    # profile=None + apply_safe_overrides=True -> colab_safe (legacy behaviour).
    data = cm.build_effective_config(
        repo_root=REPO_ROOT,
        project_root=tmp_path / "proj",
        run_id="run1",
        apply_safe_overrides=True,
    )
    assert data["copilot"]["vlm"]["provider"] == "mock"


def test_effective_config_validates_against_pipeline_loader(tmp_path: Path) -> None:
    data = cm.build_effective_config(
        repo_root=REPO_ROOT,
        project_root=tmp_path / "proj",
        run_id="run1",
        profile="production",
    )
    out = tmp_path / "active.yaml"
    cm.write_effective_config(data=data, out_path=out)
    ok, detail = cm.validate_effective_config(config_path=out, repo_root=REPO_ROOT)
    assert ok, detail
