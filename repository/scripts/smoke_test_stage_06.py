#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import yaml

from pipeline.common.config import load_config
from pipeline.stage_06_da3_assist.run_da3_assist import run_da3_assist


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((24, 32, 3), value, dtype=np.uint8)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"Could not write {path}")


def _write_sparse_text(root: Path) -> None:
    text_dir = root / "data" / "da3" / "site01" / "sparse_txt"
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / "cameras.txt").write_text("# cameras\n1 PINHOLE 32 24 30 30 16 12\n", encoding="utf-8")

    points_lines = ["# points"]
    images_lines = ["# images"]
    pid = 1
    for img_idx in range(1, 5):
        name = f"site01_kf_{img_idx:05d}_f{img_idx:06d}.jpg"
        tx = float(img_idx - 1) * 0.05
        header = f"{img_idx} 1 0 0 0 {tx:.6f} 0 0 1 {name}"
        pairs = []
        for y in range(4, 20, 4):
            for x in range(4, 28, 4):
                z = 4.0 + 0.02 * x + 0.03 * y
                X = (x - 16) * z / 30.0 - tx
                Y = (y - 12) * z / 30.0
                points_lines.append(f"{pid} {X:.6f} {Y:.6f} {z:.6f} 180 180 180 0.1 {img_idx} 0")
                pairs.extend([f"{float(x):.3f}", f"{float(y):.3f}", str(pid)])
                pid += 1
        images_lines.append(header)
        images_lines.append(" ".join(pairs))

    (text_dir / "points3D.txt").write_text("\n".join(points_lines) + "\n", encoding="utf-8")
    (text_dir / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf-8")


def _write_project(root: Path) -> Path:
    image_dir = root / "data" / "sfm" / "site01" / "images"
    depth_dir = root / "data" / "da3" / "site01" / "raw_depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 5):
        image_name = f"site01_kf_{idx:05d}_f{idx:06d}.jpg"
        _write_image(image_dir / image_name, 40 + idx * 30)
        yy, xx = np.mgrid[0:24, 0:32]
        metric_depth = 4.0 + 0.02 * xx + 0.03 * yy
        raw_depth = metric_depth / 2.0
        np.save(depth_dir / f"{Path(image_name).stem}.npy", raw_depth.astype(np.float32))

    dense_report = root / "runs" / "smoke_stage_06" / "reports" / "dense_summary.json"
    dense_report.parent.mkdir(parents=True, exist_ok=True)
    dense_report.write_text(
        yaml.safe_dump(
            {
                "dense_stats": {
                    "fused_vertex_count": 5,
                    "points_per_input_image": 1.25,
                    "depth_map_ratio": 0.1,
                    "input_image_count": 4,
                },
                "quality_gate": {"passed": False, "warnings": ["smoke weak dense"]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _write_sparse_text(root)

    cfg = {
        "project": {"name": "site01", "run_id": "smoke_stage_06", "root": str(root), "random_seed": 42},
        "inputs": {"video": "data/raw/site01.mp4", "ifc": "data/bim/design/model.ifc", "schedule": "data/bim/design/schedule.csv"},
        "paths": {
            "dense_summary_json": "runs/smoke_stage_06/reports/dense_summary.json",
            "sparse_refined_dir": "data/sparse_refined/site01/0",
            "sfm_images_dir": "data/sfm/site01/images",
            "keyframes_dir": "data/frames/key/site01",
            "da3_dir": "data/da3/site01",
            "da3_decision_json": "data/da3/site01/decision.json",
            "da3_depth_manifest_csv": "data/da3/site01/depth_manifest.csv",
            "da3_alignment_manifest_csv": "data/da3/site01/alignment_manifest.csv",
            "da3_fusion_plan_json": "data/da3/site01/fusion_plan.json",
            "da3_assisted_ply": "data/da3/site01/da3_assisted_points.ply",
            "da3_report_json": "runs/smoke_stage_06/reports/da3_summary.json",
            "normalized_video": "data/normalized/site01_normalized.mp4",
            "metadata_json": "data/normalized/site01_metadata.json",
            "quality_csv": "data/normalized/site01_frame_quality.csv",
            "manifest_csv": "data/frames/key/site01_manifest.csv",
            "contact_sheet": "data/frames/key/site01_contact_sheet.jpg",
            "sfm_dir": "data/sfm/site01",
            "colmap_db": "data/sfm/site01/database.db",
            "sparse_dir": "data/sparse/site01",
            "sparse_report_json": "runs/smoke_stage_06/reports/sparse_stats.json",
            "sparse_refined_dir": "data/sparse_refined/site01/0",
            "dense_workspace": "data/dense/site01",
            "fused_ply": "data/dense/site01/fused.ply",
            "da3_dir": "data/da3/site01",
            "clean_dir": "data/clean/site01",
            "bim_aligned_dir": "data/bim/aligned/site01",
            "metrics_dir": "data/bim/metrics/site01",
        },
        "video": {"normalize_fps": 30, "sample_fps_for_quality": 2},
        "video_quality": {"min_blur_laplacian": 80, "max_duplicate_similarity": 0.96, "max_exposure_jump": 0.25, "min_motion_score": 0.01, "max_motion_score": 0.75, "adaptive_blur_update_min_ratio": 0.55, "quality_weights": {"sharpness": 0.25, "exposure": 0.2, "motion": 0.15, "novelty": 0.2, "feature_density": 0.2}},
        "keyframes": {"min_time_gap_sec": 0.5, "max_frames_first_run": 250, "min_segment_duration_sec": 1.0, "min_segment_frames": 2, "max_segment_gap_sec": 2.0, "contact_sheet_thumb_width": 240, "contact_sheet_max_images": 80, "jpeg_quality": 92, "selection_quality_weight": 0.5, "selection_novelty_weight": 0.3, "selection_feature_weight": 0.2, "dense_keep_ratio": 1.0, "fallback_min_keyframes": 2, "allow_relaxed_fallback": True, "fallback_blur_ratio": 0.65, "fallback_exposure_multiplier": 1.5, "fallback_motion_multiplier": 1.15, "random_seek_extraction": False, "reject_stage1_warnings": False, "reject_low_feature_density": False, "verify_frame_index_bounds": True, "verify_timestamp_frame_consistency": True, "max_timestamp_frame_index_drift_sec": 0.1, "emergency_fallback_if_no_keyframes": True},
        "colmap": {"executable": "colmap", "camera_model": "SIMPLE_RADIAL", "single_camera": True, "feature_type": "ALIKED_N16ROT", "matcher_type": "ALIKED_LIGHTGLUE", "fallback_feature_type": "SIFT", "fallback_matcher_type": "SIFT_LIGHTGLUE", "enable_fallback": True, "matching_strategy": "sequential", "sequential_overlap": 3, "sequential_quadratic_overlap": False, "sequential_loop_detection": False, "stage_images_mode": "copy", "min_input_images": 2, "aliked_max_num_features": 128, "sift_max_num_features": 128, "mapper_min_num_matches": 3, "mapper_multiple_models": True, "mapper_extract_colors": False, "qt_qpa_platform": "offscreen", "download_models": False, "use_existing_masks": False, "require_masks": False},
        "dense": {"max_image_size": 1600, "geom_consistency": True, "patch_window_radius": 5, "filter_min_ncc": 0.05},
        "refinement": {"enabled": True, "method": "colmap_bundle_adjustment"},
        "da3": {
            "enabled": "auto",
            "provider": "precomputed",
            "depth_input_dir": "data/da3/site01/raw_depth",
            "depth_output_dir": "data/da3/site01/raw_depth",
            "aligned_depth_dir": "data/da3/site01/aligned_depth",
            "sparse_text_dir": "data/da3/site01/sparse_txt",
            "depth_file_extensions": [".npy"],
            "activate_if_quality_gate_failed": True,
            "activate_if_fused_vertices_below": 50000,
            "activate_if_points_per_image_below": 100.0,
            "activate_if_depth_map_ratio_below": 0.5,
            "fail_if_required_but_unavailable": False,
            "max_images": 4,
            "min_alignment_anchors": 6,
            "max_alignment_rmse_m": 0.2,
            "min_alignment_inlier_ratio": 0.25,
            "alignment_method": "scale_only_ransac",
            "alignment_ransac_iterations": 80,
            "alignment_ransac_inlier_abs_m": 0.1,
            "alignment_ransac_inlier_rel": 0.03,
            "alignment_depth_bucket_count": 8,
            "quality_min_aligned_depth_maps": 1,
            "fuse_aligned_depth": True,
            "fusion_stride": 8,
            "fusion_max_points": 10000,
            "fusion_binary_ply": True,
            "fusion_edge_aware_filter": True,
            "fusion_edge_threshold_m": 1.0,
        },
        "bim": {"units": "meters", "icp_max_corr_distance_m": 0.08},
        "progress": {"coverage_threshold": 0.65, "deviation_threshold_m": 0.05},
    }

    cfg_path = root / "config.yaml"
    cfg.setdefault(
        "cleanup",
        {
            "voxel_size_m": 0.02,
            "statistical_nb_neighbors": 20,
            "statistical_std_ratio": 2.0,
            "radius_nb_points": 8,
            "radius_m": 0.08,
            "plane_distance_threshold_m": 0.03,
            "plane_ransac_n": 3,
            "plane_num_iterations": 1000,
            "min_plane_points": 500,
            "mesh_enabled": True,
            "mesh_method": "poisson",
            "poisson_depth": 9,
            "normal_radius_m": 0.08,
            "normal_max_nn": 30,
        },
    )

    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage06_smoke_") as tmp:
        root = Path(tmp)
        cfg_path = _write_project(root)
        cfg = load_config(cfg_path)
        report = run_da3_assist(cfg, force=True, log_level="ERROR")
        if report["status"] not in {"completed", "completed_with_alignment_warnings"}:
            raise SystemExit(f"STAGE_06_SMOKE_FAILED unexpected status {report['status']}")
        assisted = Path(report["outputs"]["assisted_ply"])
        if not assisted.exists() or assisted.stat().st_size <= 0:
            raise SystemExit("STAGE_06_SMOKE_FAILED missing assisted PLY")
        print(f"STAGE_06_SMOKE_OK status={report['status']} assisted={assisted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
