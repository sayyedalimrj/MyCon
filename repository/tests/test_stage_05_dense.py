from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.stage_05_dense.dense_stats import build_dense_stats, evaluate_quality_gate, parse_ply_vertex_count
from pipeline.stage_05_dense.io_utils import DenseWorkspaceSafetyError, clean_dense_workspace
from pipeline.stage_05_dense.model_io import require_image_dir, validate_sparse_model


def test_parse_ply_vertex_count(tmp_path: Path) -> None:
    ply = tmp_path / "fused.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 123\nproperty float x\nproperty float y\nproperty float z\nend_header\n",
        encoding="utf-8",
    )
    assert parse_ply_vertex_count(ply) == 123


def test_validate_sparse_model_contract(tmp_path: Path) -> None:
    model = tmp_path / "0"
    model.mkdir()
    for name in ["cameras.bin", "images.bin", "points3D.bin"]:
        (model / name).write_bytes(b"x")
    result = validate_sparse_model(model)
    assert result["valid_binary_contract"] is True


def test_require_image_dir_counts_supported_images(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"x")
    (images / "b.png").write_bytes(b"x")
    (images / "ignore.txt").write_text("x", encoding="utf-8")
    assert len(require_image_dir(images, 2)) == 2


def test_dense_quality_gate_passes_for_reasonable_density() -> None:
    cfg = {"dense": {"quality_min_fused_points": 10, "quality_min_fused_points_per_image": 20.0}}
    stats = {
        "input_image_count": 4,
        "fused_vertex_count": 100,
        "points_per_input_image": 25.0,
        "depth_map_ratio": 0.5,
        "fused_ply_exists": True,
    }
    gate = evaluate_quality_gate(cfg, stats)
    assert gate["passed"] is True
    assert gate["failures"] == []


def test_dense_quality_gate_fails_without_vertices() -> None:
    cfg = {"dense": {"quality_min_fused_points": 10, "quality_min_fused_points_per_image": 1.0}}
    stats = {"input_image_count": 4, "fused_vertex_count": None, "points_per_input_image": None, "depth_map_ratio": 1.0, "fused_ply_exists": True}
    gate = evaluate_quality_gate(cfg, stats)
    assert gate["passed"] is False
    assert "fused_vertex_count_unavailable" in gate["failures"]


def test_dense_stats_points_per_image_and_depth_ratio(tmp_path: Path) -> None:
    workspace = tmp_path / "dense"
    (workspace / "stereo" / "depth_maps").mkdir(parents=True)
    (workspace / "images").mkdir(parents=True)
    for idx in range(3):
        (workspace / "stereo" / "depth_maps" / f"img{idx}.geometric.bin").write_bytes(b"d")
    fused = workspace / "fused.ply"
    fused.write_text("ply\nformat ascii 1.0\nelement vertex 90\nend_header\n", encoding="utf-8")
    stats = build_dense_stats(workspace, fused, input_image_count=3)
    assert stats["points_per_input_image"] == 30.0
    assert stats["depth_map_ratio"] == 1.0


def test_dense_workspace_safety_rejects_sfm_dir(tmp_path: Path) -> None:
    root = tmp_path / "project"
    bad = root / "data" / "sfm"
    bad.mkdir(parents=True)
    with pytest.raises(DenseWorkspaceSafetyError):
        clean_dense_workspace(bad, root, force=True)


def test_dense_workspace_safety_allows_data_dense(tmp_path: Path) -> None:
    root = tmp_path / "project"
    workspace = root / "data" / "dense" / "site01"
    clean_dense_workspace(workspace, root, force=True)
    assert (workspace / ".dense_workspace_lock").exists()


def test_config_overlay_runtime_override() -> None:
    from pipeline.stage_05_dense.config_access import ConfigOverlay, cfg_get

    base = {"dense": {"max_image_size": 1600}}
    overlay = ConfigOverlay(base, {"dense.max_image_size": 2200})
    assert cfg_get(overlay, "dense.max_image_size", 0) == 2200
    assert cfg_get(overlay, "dense.patch_window_radius", 5) == 5


def test_gpu_profile_memory_classes() -> None:
    from pipeline.stage_05_dense.gpu_preflight import _profile_overrides_for_memory

    high, high_name = _profile_overrides_for_memory(24576)
    low, low_name = _profile_overrides_for_memory(6144)
    assert high_name == "24GB-class"
    assert high["dense.patch_match_max_image_size"] >= 2200
    assert high["dense.num_patch_match_src_images"] == 20
    assert low_name == "6-8GB-class"
    assert low["dense.patch_match_max_image_size"] <= 1200


def test_gpu_profile_caps_large_image_sets() -> None:
    from pipeline.stage_05_dense.gpu_preflight import _profile_overrides_for_memory

    high, profile_name = _profile_overrides_for_memory(24576, input_image_count=127)
    very_large, very_large_name = _profile_overrides_for_memory(24576, input_image_count=250)
    assert profile_name == "24GB-class+large-image-set-cap"
    assert high["dense.patch_match_max_image_size"] <= 1500
    assert high["dense.num_patch_match_src_images"] <= 14
    assert very_large_name == "24GB-class+very-large-image-set-cap"
    assert very_large["dense.patch_match_max_image_size"] <= 1400
    assert very_large["dense.num_patch_match_src_images"] <= 12
