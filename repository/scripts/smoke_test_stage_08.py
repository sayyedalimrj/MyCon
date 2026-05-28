#!/usr/bin/env python3
"""Fast Stage 8 smoke test using synthetic scan and synthetic BIM fallback."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
try:
    import open3d as o3d
except ModuleNotFoundError:
    print("SMOKE_SKIP_MISSING_DEPENDENCY script=smoke_test_stage_08.py dependency=open3d")
    raise SystemExit(0)
import yaml

from pipeline.common.config import load_config
from pipeline.stage_08_bim_eval.run_registration import run_registration


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _write_plane_box_cloud(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(0.0, 2.0, 35)
    ys = np.linspace(0.0, 1.2, 25)
    zs = np.linspace(0.0, 1.8, 25)
    pts: list[list[float]] = []
    colors: list[list[float]] = []
    for x in xs:
        for y in ys:
            pts.append([float(x), float(y), 0.0])
            colors.append([0.55, 0.55, 0.55])
    for x in xs:
        for z in zs:
            pts.append([float(x), 0.0, float(z)])
            colors.append([0.70, 0.70, 0.70])
    for y in ys:
        for z in zs:
            pts.append([0.0, float(y), float(z)])
            colors.append([0.60, 0.60, 0.60])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    assert o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def _write_config(root: Path) -> Path:
    cfg = {
        "project": {"name": "site01", "run_id": "smoke_stage_08", "root": str(root), "random_seed": 42},
        "inputs": {"video": "data/raw/site01.mp4", "ifc": "data/bim/design/model.ifc", "schedule": "data/bim/design/schedule.csv"},
        "paths": {
            "normalized_video": "data/normalized/site01_normalized.mp4",
            "metadata_json": "data/normalized/site01_metadata.json",
            "quality_csv": "data/normalized/site01_frame_quality.csv",
            "keyframes_dir": "data/frames/key/site01",
            "manifest_csv": "data/frames/key/site01_manifest.csv",
            "contact_sheet": "data/frames/key/site01_contact_sheet.jpg",
            "sfm_dir": "data/sfm/site01",
            "colmap_db": "data/sfm/site01/database.db",
            "sparse_dir": "data/sparse/site01",
            "sparse_report_json": "runs/smoke_stage_08/reports/sparse_stats.json",
            "sparse_refined_dir": "data/sparse_refined/site01",
            "dense_workspace": "data/dense/site01",
            "fused_ply": "data/dense/site01/fused.ply",
            "da3_dir": "data/da3/site01",
            "clean_dir": "data/clean/site01",
            "bim_aligned_dir": "data/bim/aligned/site01",
            "metrics_dir": "data/bim/metrics/site01",
            "sparse_mask_dir": "data/masks/site01",
        },
        "video": {"normalize_fps": 30, "sample_fps_for_quality": 2},
        "video_quality": {"min_blur_laplacian": 80, "max_duplicate_similarity": 0.96, "max_exposure_jump": 0.25, "min_motion_score": 0.01, "max_motion_score": 0.75},
        "keyframes": {"min_time_gap_sec": 0.5, "max_frames_first_run": 250},
        "colmap": {"executable": "colmap", "camera_model": "SIMPLE_RADIAL", "single_camera": True},
        "refinement": {"enabled": True, "output_sparse_dir": "data/sparse_refined/site01/0", "report_json": "runs/smoke_stage_08/reports/refinement_stats.json"},
        "dense": {"workspace_dir": "data/dense/site01", "fused_ply": "data/dense/site01/fused.ply", "max_image_size": 1200, "geom_consistency": True},
        "da3": {"enabled": "auto", "model": "DA3Metric-Large", "activate_if_dense_coverage_below": 0.72},
        "cleanup": {
            "voxel_size_m": 0.02,
            "statistical_nb_neighbors": 20,
            "statistical_std_ratio": 2.0,
            "radius_nb_points": 8,
            "radius_m": 0.08,
            "mesh_enabled": True,
        },
        "bim": {
            "units": "meters",
            "scan_input_candidates": ["data/clean/site01/cleaned_cloud.ply", "data/dense/site01/fused.ply"],
            "bim_reference_ply": "data/bim/aligned/site01/bim_reference.ply",
            "bim_reference_mesh": "data/bim/aligned/site01/bim_reference_mesh.ply",
            "scan_aligned_ply": "data/bim/aligned/site01/scan_aligned.ply",
            "transform_json": "data/bim/aligned/site01/transform_scan_to_bim.json",
            "element_metadata_jsonl": "data/bim/aligned/site01/bim_elements.jsonl",
            "registration_report_json": "runs/smoke_stage_08/reports/registration_report.json",
            "allow_synthetic_ifc_fallback_for_tests": True,
            "estimate_initial_scale_from_bbox": False,
            "coarse_fpfh_enabled": False,
            "icp_enabled": True,
            "icp_estimation": "point_to_point",
            "icp_stages": ["point_to_point"],
            "icp_max_corr_distance_m": 0.25,
            "quality_min_icp_fitness": 0.01,
            "quality_max_icp_rmse_m": 0.5,
            "fail_on_low_registration_quality": False,
        },
        "progress": {"coverage_threshold": 0.65, "deviation_threshold_m": 0.05},
    }
    base_path = PROJECT_ROOT / "configs" / "site01.yaml"
    if base_path.exists():
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        if isinstance(base, dict):
            _deep_update(base, cfg)
            cfg = base

    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage08_smoke_") as tmp:
        root = Path(tmp)
        _write_plane_box_cloud(root / "data" / "clean" / "site01" / "cleaned_cloud.ply")
        cfg = load_config(_write_config(root))
        report = run_registration(cfg, force=True, log_level="ERROR")
        if report.get("status") != "complete":
            raise SystemExit("STAGE_08_SMOKE_FAILED incomplete report")
        outputs = report.get("outputs", {})
        for key in ["bim_reference_ply", "scan_aligned_ply", "transform_json", "report_json"]:
            path = Path(outputs[key])
            if not path.exists() or path.stat().st_size <= 0:
                raise SystemExit(f"STAGE_08_SMOKE_FAILED missing {key}: {path}")
        fitness = report.get("icp", {}).get("fitness")
        print(f"STAGE_08_SMOKE_OK fitness={fitness} scan={outputs['scan_aligned_ply']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
