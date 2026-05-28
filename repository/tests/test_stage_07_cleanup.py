from __future__ import annotations

import pytest
import json
from pathlib import Path

import numpy as np
o3d = pytest.importorskip("open3d")

from pipeline.stage_07_cleanup.config_access import input_candidates
from pipeline.stage_07_cleanup.input_selection import select_input_cloud
from pipeline.stage_07_cleanup.plane_extraction import extract_planes
from pipeline.stage_07_cleanup.point_cloud_cleanup import clean_point_cloud
from pipeline.stage_07_cleanup.run_cleanup import run_cleanup
from pipeline.stage_07_cleanup.semantic_context import summarise_yolo_context, summarise_vlm_context


def _write_plane_cloud(path: Path) -> None:
    rng = np.random.default_rng(7)
    x, y = np.meshgrid(np.linspace(-1.5, 1.5, 35), np.linspace(-1.5, 1.5, 35))
    floor = np.column_stack([x.ravel(), y.ravel(), rng.normal(0, 0.005, x.size)])
    y2, z2 = np.meshgrid(np.linspace(-1.5, 1.5, 35), np.linspace(0, 2, 35))
    wall = np.column_stack([rng.normal(0, 0.005, y2.size), y2.ravel(), z2.ravel()])
    pts = np.vstack([floor, wall])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([[0.5, 0.5, 0.5]], (pts.shape[0], 1)))
    path.parent.mkdir(parents=True, exist_ok=True)
    assert o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def _cfg(tmp_path: Path, input_path: Path) -> dict:
    return {
        "project": {"root": str(tmp_path), "name": "site01", "run_id": "test"},
        "paths": {
            "clean_dir": "data/clean/site01",
            "clean_downsampled_cloud": "data/clean/site01/downsampled_cloud.ply",
            "clean_cloud": "data/clean/site01/cleaned_cloud.ply",
            "clean_mesh": "data/clean/site01/mesh.ply",
            "clean_planes_json": "data/clean/site01/planes.json",
            "clean_plane_clouds_dir": "data/clean/site01/plane_clouds",
            "cleanup_report_json": "runs/test/reports/cleanup_summary.json",
            "fused_ply": str(input_path.relative_to(tmp_path)),
        },
        "cleanup": {
            "input_candidates": [str(input_path.relative_to(tmp_path))],
            "voxel_size_m": 0.03,
            "statistical_enabled": False,
            "radius_enabled": False,
            "estimate_normals": True,
            "normal_radius_m": 0.10,
            "normal_max_nn": 30,
            "orient_normals_consistent_tangent_plane_k": 0,
            "plane_extraction_enabled": True,
            "plane_voxel_size_m": 0.04,
            "max_planes": 4,
            "plane_distance_threshold_m": 0.05,
            "plane_num_iterations": 400,
            "min_plane_points": 50,
            "min_remaining_points": 50,
            "mesh_enabled": True,
            "mesh_method": "poisson",
            "mesh_min_vertices": 50000,
            "write_binary_ply": True,
            "quality_min_points": 100,
            "quality_min_planes": 1,
            "fail_on_quality_gate": False,
            "yolo_detections_jsonl": "data/semantics/site01/yolo_detections.jsonl",
            "yolo_summary_json": "data/semantics/site01/yolo_summary.json",
            "yolo_transient_classes": ["person", "truck"],
            "vlm_scene_report_json": "data/semantics/site01/vlm_scene_report.json",
            "vlm_summary_json": "data/semantics/site01/vlm_summary.json",
        },
    }


def test_input_candidates_resolve_relative_paths(tmp_path: Path) -> None:
    cloud = tmp_path / "data" / "dense" / "site01" / "fused.ply"
    cfg = _cfg(tmp_path, cloud)
    candidates = input_candidates(cfg)
    assert candidates[0] == cloud


def test_select_input_cloud(tmp_path: Path) -> None:
    cloud = tmp_path / "data" / "dense" / "site01" / "fused.ply"
    _write_plane_cloud(cloud)
    selected = select_input_cloud(_cfg(tmp_path, cloud))
    assert selected.path == cloud
    assert selected.source == "dense_fused"


def test_clean_point_cloud_and_planes(tmp_path: Path) -> None:
    cloud = tmp_path / "data" / "dense" / "site01" / "fused.ply"
    _write_plane_cloud(cloud)
    cfg = _cfg(tmp_path, cloud)
    import logging

    cleaned, result = clean_point_cloud(
        cfg,
        cloud,
        tmp_path / "data" / "clean" / "site01" / "downsampled_cloud.ply",
        tmp_path / "data" / "clean" / "site01" / "cleaned_cloud.ply",
        logging.getLogger("test"),
    )
    assert result.cleaned_count > 100
    planes = extract_planes(cfg, cleaned, tmp_path / "planes", tmp_path / "planes.json")
    assert len(planes) >= 1
    assert (tmp_path / "planes.json").exists()


def test_semantic_context_summaries(tmp_path: Path) -> None:
    cfg = {"project": {"root": str(tmp_path)}, "cleanup": {"yolo_transient_classes": ["person"]}}
    yolo = tmp_path / "detections.jsonl"
    yolo.write_text(json.dumps({"detections": [{"class": "person"}, {"class": "wall"}]}) + "\n", encoding="utf-8")
    summary = summarise_yolo_context(cfg, yolo, tmp_path / "yolo_summary.json")
    assert summary["status"] == "ok"
    assert summary["transient_frame_count"] == 1
    vlm = tmp_path / "vlm.json"
    vlm.write_text(json.dumps({"summary": "site scene", "cleanup_hints": ["remove sky"]}), encoding="utf-8")
    vsummary = summarise_vlm_context(cfg, vlm, tmp_path / "vlm_summary.json")
    assert vsummary["status"] == "ok"
    assert vsummary["summary"] == "site scene"


def test_run_cleanup_smoke(tmp_path: Path) -> None:
    cloud = tmp_path / "data" / "dense" / "site01" / "fused.ply"
    _write_plane_cloud(cloud)
    cfg = _cfg(tmp_path, cloud)
    report = run_cleanup(cfg, force=True, log_level="ERROR")
    assert report["cleanup"]["cleaned_count"] > 100
    assert report["planes"]["count"] >= 1
    assert Path(report["outputs"]["cleaned_cloud"]).exists()
    assert Path(report["outputs"]["planes_json"]).exists()


def test_voxel_downsample_precedes_last_resort_cap(tmp_path: Path) -> None:
    # Many duplicate points in a tiny area should collapse through voxelization before the max cap matters.
    cloud = tmp_path / "data" / "dense" / "site01" / "fused.ply"
    pts = np.repeat(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64), 100, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([[0.5, 0.5, 0.5]], (pts.shape[0], 1)))
    cloud.parent.mkdir(parents=True, exist_ok=True)
    assert o3d.io.write_point_cloud(str(cloud), pcd, write_ascii=False)
    cfg = _cfg(tmp_path, cloud)
    cfg["cleanup"].update({"voxel_size_m": 0.5, "max_processing_points": 10, "dynamic_voxel_enabled": True})
    import logging

    _cleaned, result = clean_point_cloud(
        cfg,
        cloud,
        tmp_path / "data" / "clean" / "site01" / "downsampled_cloud.ply",
        tmp_path / "data" / "clean" / "site01" / "cleaned_cloud.ply",
        logging.getLogger("test"),
    )
    assert result.downsampled_count <= 10
    assert not any(str(w).startswith("post_voxel_random_subsample_applied") for w in result.warnings)


def test_semantic_color_filter_is_conditional_and_capped(tmp_path: Path) -> None:
    cloud = tmp_path / "data" / "dense" / "site01" / "fused.ply"
    gray = np.tile([[0.5, 0.5, 0.5]], (100, 1))
    orange = np.tile([[1.0, 0.45, 0.0]], (8, 1))
    pts = np.column_stack([np.linspace(0, 1, 108), np.zeros(108), np.zeros(108)])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.vstack([gray, orange]))
    cloud.parent.mkdir(parents=True, exist_ok=True)
    assert o3d.io.write_point_cloud(str(cloud), pcd, write_ascii=False)
    cfg = _cfg(tmp_path, cloud)
    cfg["cleanup"].update(
        {
            "semantic_color_filter_enabled": True,
            "semantic_color_filter_activation": "if_yolo_transients",
            "semantic_color_filter_max_removed_ratio": 0.12,
            "voxel_size_m": 0.0,
            "statistical_enabled": False,
            "radius_enabled": False,
        }
    )
    import logging

    _cleaned, result = clean_point_cloud(
        cfg,
        cloud,
        tmp_path / "data" / "clean" / "site01" / "downsampled_cloud.ply",
        tmp_path / "data" / "clean" / "site01" / "cleaned_cloud.ply",
        logging.getLogger("test"),
        semantic_context={"yolo": {"transient_frame_count": 2}},
    )
    assert result.cleaned_count < result.input_count
    assert any(str(w).startswith("semantic_color_filter_removed_ratio") for w in result.warnings)


def test_normal_guided_plane_extraction_labels_floor_and_wall(tmp_path: Path) -> None:
    cloud = tmp_path / "data" / "dense" / "site01" / "fused.ply"
    _write_plane_cloud(cloud)
    cfg = _cfg(tmp_path, cloud)
    cfg["cleanup"].update(
        {
            "normal_guided_plane_extraction": True,
            "normal_guided_horizontal_dot_min": 0.75,
            "normal_guided_wall_abs_dot_max": 0.45,
            "plane_voxel_size_m": 0.02,
            "min_plane_points": 50,
            "min_remaining_points": 20,
        }
    )
    pcd = o3d.io.read_point_cloud(str(cloud))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    planes = extract_planes(cfg, pcd, tmp_path / "planes", tmp_path / "planes.json")
    labels = {p.label for p in planes}
    assert labels & {"floor_or_ceiling", "ceiling_or_floor"}
    assert "wall" in labels
    assert all(p.extraction_mode.startswith("normal_guided") for p in planes[:2])

pytestmark = pytest.mark.geometry
