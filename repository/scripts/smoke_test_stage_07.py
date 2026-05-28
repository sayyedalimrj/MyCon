#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
try:
    import open3d as o3d
except ModuleNotFoundError:
    print("SMOKE_SKIP_MISSING_DEPENDENCY script=smoke_test_stage_07.py dependency=open3d")
    raise SystemExit(0)
import yaml

from pipeline.common.config import load_config
from pipeline.stage_07_cleanup.run_cleanup import run_cleanup


def _make_cloud(path: Path) -> None:
    rng = np.random.default_rng(42)
    floor_x, floor_y = np.meshgrid(np.linspace(-2, 2, 45), np.linspace(-2, 2, 45))
    floor = np.column_stack([floor_x.ravel(), floor_y.ravel(), rng.normal(0, 0.006, floor_x.size)])
    wall_y, wall_z = np.meshgrid(np.linspace(-2, 2, 45), np.linspace(0, 2.8, 45))
    wall = np.column_stack([rng.normal(0, 0.006, wall_y.size), wall_y.ravel(), wall_z.ravel()])
    column_theta = np.linspace(0, 2 * np.pi, 120)
    column_z = np.linspace(0, 2.5, 30)
    ct, cz = np.meshgrid(column_theta, column_z)
    column = np.column_stack([1.1 + 0.15 * np.cos(ct.ravel()), -0.8 + 0.15 * np.sin(ct.ravel()), cz.ravel()])
    outliers = rng.uniform(-5, 5, size=(120, 3))
    points = np.vstack([floor, wall, column, outliers]).astype(np.float64)
    colors = np.clip((points - points.min(axis=0)) / np.maximum(points.ptp(axis=0), 1e-6), 0.0, 1.0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), pcd, write_ascii=False):
        raise RuntimeError(f"Failed to write smoke cloud: {path}")


def _base_config(root: Path) -> dict:
    template = PROJECT_ROOT / "configs" / "site01.yaml"
    if template.exists():
        cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
    else:
        cfg = {
            "project": {"name": "site01", "run_id": "smoke_stage_07", "root": str(root), "random_seed": 42},
            "inputs": {"video": "data/raw/site01.mp4", "ifc": "data/bim/design/model.ifc", "schedule": "data/bim/design/schedule.csv"},
            "paths": {},
            "video": {"normalize_fps": 30, "sample_fps_for_quality": 2},
            "video_quality": {"min_blur_laplacian": 80, "max_duplicate_similarity": 0.96, "adaptive_blur_update_min_ratio": 0.55, "quality_weights": {"sharpness": 0.25, "exposure": 0.2, "motion": 0.15, "novelty": 0.2, "feature_density": 0.2}},
            "keyframes": {},
            "colmap": {},
            "refinement": {},
            "dense": {},
            "da3": {},
            "cleanup": {},
            "bim": {"units": "meters", "icp_max_corr_distance_m": 0.08},
            "progress": {"coverage_threshold": 0.65, "deviation_threshold_m": 0.05},
        }
    cfg.setdefault("project", {})["name"] = "site01"
    cfg["project"]["run_id"] = "smoke_stage_07"
    cfg["project"]["root"] = str(root)
    cfg["project"]["random_seed"] = 42
    return cfg


def _write_config(root: Path, input_cloud: Path) -> Path:
    cfg = _base_config(root)
    cfg.setdefault("paths", {}).update(
        {
            "clean_dir": "data/clean/site01",
            "clean_downsampled_cloud": "data/clean/site01/downsampled_cloud.ply",
            "clean_cloud": "data/clean/site01/cleaned_cloud.ply",
            "clean_mesh": "data/clean/site01/mesh.ply",
            "clean_planes_json": "data/clean/site01/planes.json",
            "clean_plane_clouds_dir": "data/clean/site01/plane_clouds",
            "cleanup_report_json": "runs/smoke_stage_07/reports/cleanup_summary.json",
            "fused_ply": str(input_cloud.relative_to(root)),
            "da3_assisted_ply": "data/da3/site01/missing.ply",
        }
    )
    cfg["cleanup"] = {
        "enabled": True,
        "input_candidates": [str(input_cloud.relative_to(root))],
        "output_dir": "data/clean/site01",
        "report_json": "runs/smoke_stage_07/reports/cleanup_summary.json",
        "log_path": "runs/smoke_stage_07/logs/stage_07_cleanup.log",
        "max_processing_points": 100000,
        "voxel_size_m": 0.04,
        "statistical_enabled": True,
        "statistical_nb_neighbors": 20,
        "statistical_std_ratio": 2.5,
        "radius_enabled": False,
        "estimate_normals": True,
        "normal_radius_m": 0.12,
        "normal_max_nn": 30,
        "orient_normals_consistent_tangent_plane_k": 0,
        "plane_extraction_enabled": True,
        "plane_voxel_size_m": 0.05,
        "max_planes": 5,
        "plane_distance_threshold_m": 0.06,
        "plane_ransac_n": 3,
        "plane_num_iterations": 500,
        "min_plane_points": 80,
        "min_plane_ratio": 0.01,
        "min_remaining_points": 80,
        "min_remaining_ratio": 0.05,
        "mesh_enabled": True,
        "mesh_method": "poisson",
        "mesh_min_vertices": 50000,
        "fail_if_mesh_fails": False,
        "write_binary_ply": True,
        "quality_min_points": 500,
        "quality_min_planes": 1,
        "quality_max_removed_ratio": 0.95,
        "strict_quality_gate": False,
        "fail_on_quality_gate": False,
        "semantics_enabled": True,
        "yolo_enabled": True,
        "yolo_detections_jsonl": "data/semantics/site01/yolo_detections.jsonl",
        "yolo_summary_json": "data/semantics/site01/yolo_summary.json",
        "yolo_transient_classes": ["person", "truck", "crane"],
        "vlm_enabled": True,
        "vlm_scene_report_json": "data/semantics/site01/vlm_scene_report.json",
        "vlm_summary_json": "data/semantics/site01/vlm_summary.json",
    }
    semantics_dir = root / "data" / "semantics" / "site01"
    semantics_dir.mkdir(parents=True, exist_ok=True)
    (semantics_dir / "yolo_detections.jsonl").write_text(
        json.dumps({"image_path": "a.jpg", "detections": [{"class": "person"}, {"class": "wall"}]}) + "\n",
        encoding="utf-8",
    )
    (semantics_dir / "vlm_scene_report.json").write_text(
        json.dumps({"summary": "synthetic construction scene", "cleanup_hints": ["preserve wall and floor planes"]}),
        encoding="utf-8",
    )
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage07_smoke_") as tmp:
        root = Path(tmp)
        input_cloud = root / "data" / "dense" / "site01" / "fused.ply"
        _make_cloud(input_cloud)
        cfg_path = _write_config(root, input_cloud)
        cfg = load_config(cfg_path)
        report = run_cleanup(cfg, force=True, log_level="ERROR")
        cleaned = Path(report["outputs"]["cleaned_cloud"])
        planes_json = Path(report["outputs"]["planes_json"])
        if not cleaned.exists() or cleaned.stat().st_size <= 0:
            raise SystemExit("STAGE_07_SMOKE_FAILED missing cleaned cloud")
        if report["planes"]["count"] < 1:
            raise SystemExit("STAGE_07_SMOKE_FAILED expected at least one plane")
        if not planes_json.exists() or planes_json.stat().st_size <= 0:
            raise SystemExit("STAGE_07_SMOKE_FAILED missing planes json")
        print(f"STAGE_07_SMOKE_OK points={report['cleanup']['cleaned_count']} planes={report['planes']['count']} mesh={report['mesh']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
