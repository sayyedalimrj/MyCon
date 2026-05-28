#!/usr/bin/env python3
"""Smoke test Stage 10 Copilot with mocked metric artifacts."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.common.config import load_config  # noqa: E402
from pipeline.stage_10_copilot.api import ask_copilot  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _write_config(root: Path) -> Path:
    metrics = root / "data" / "bim" / "metrics" / "site01"
    _write_csv(metrics / "element_metrics.csv", [{
        "global_id": "WALL-001",
        "element_name": "East Wall",
        "coverage": "0.74",
        "mean_deviation_m": "0.028",
        "max_deviation_m": "0.091",
        "confidence": "0.68",
    }])
    _write_csv(metrics / "activity_progress.csv", [{
        "activity_id": "ACT-100",
        "planned_percent": "80",
        "actual_percent": "62",
        "status": "behind",
        "delay_days": "3",
    }])
    (metrics / "deviation_summary.json").write_text(json.dumps({"max_deviation_m": 0.091, "hotspots": 1}), encoding="utf-8")
    (metrics / "coverage_summary.json").write_text(json.dumps({"undercovered_regions": ["upper_right"]}), encoding="utf-8")
    (metrics / "registration_quality.json").write_text(json.dumps({"fitness": 0.82, "inlier_rmse": 0.035}), encoding="utf-8")
    cloud = root / "data" / "clean" / "site01" / "cleaned_cloud.ply"
    cloud.parent.mkdir(parents=True, exist_ok=True)
    cloud.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="utf-8")

    cfg = {
        "project": {"name": "site01", "run_id": "stage10_smoke", "root": str(root), "random_seed": 42},
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
            "sparse_report_json": "runs/stage10_smoke/reports/sparse_stats.json",
            "sparse_refined_dir": "data/sparse_refined/site01",
            "dense_workspace": "data/dense/site01",
            "fused_ply": "data/dense/site01/fused.ply",
            "da3_dir": "data/da3/site01",
            "clean_dir": "data/clean/site01",
            "clean_cloud": "data/clean/site01/cleaned_cloud.ply",
            "clean_mesh": "data/clean/site01/mesh.ply",
            "planes_json": "data/clean/site01/planes.json",
            "bim_aligned_dir": "data/bim/aligned/site01",
            "metrics_dir": "data/bim/metrics/site01",
        },
        "video": {"normalize_fps": 30, "sample_fps_for_quality": 2, "clear_rotation_metadata": True, "force_constant_frame_rate": True, "cfr_option": "auto"},
        "video_quality": {"min_blur_laplacian": 80, "max_duplicate_similarity": 0.96, "adaptive_blur_update_min_ratio": 0.55, "quality_weights": {"sharpness": 0.25, "exposure": 0.2, "motion": 0.15, "novelty": 0.2, "feature_density": 0.2}},
        "keyframes": {"min_time_gap_sec": 0.5, "max_frames_first_run": 250},
        "colmap": {"executable": "colmap"},
        "refinement": {"enabled": True},
        "dense": {"enabled": True},
        "da3": {"enabled": "auto"},
        "cleanup": {"enabled": True},
        "bim": {"units": "meters", "icp_max_corr_distance_m": 0.08},
        "progress": {"coverage_threshold": 0.65, "deviation_threshold_m": 0.05},
        "copilot": {
            "enabled": True,
            "default_view": "front",
            "low_confidence_threshold": 0.65,
            "paths": {
                "evidence_dir": "runs/stage10_smoke/copilot/evidence",
                "render_dir": "runs/stage10_smoke/copilot/renders",
                "default_pointcloud": "data/clean/site01/cleaned_cloud.ply",
                "element_metrics_csv": "data/bim/metrics/site01/element_metrics.csv",
                "activity_progress_csv": "data/bim/metrics/site01/activity_progress.csv",
                "deviation_summary_json": "data/bim/metrics/site01/deviation_summary.json",
                "coverage_summary_json": "data/bim/metrics/site01/coverage_summary.json",
                "registration_quality_json": "data/bim/metrics/site01/registration_quality.json",
            },
            "vlm": {"provider": "mock", "model": "Qwen/Qwen3-VL-8B-Instruct"},
        },
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
    with tempfile.TemporaryDirectory(prefix="stage10_smoke_") as tmp:
        root = Path(tmp)
        cfg = load_config(_write_config(root))
        response = ask_copilot(cfg, {
            "question": "Has this wall been executed and can I accept it?",
            "selected_element_id": "WALL-001",
            "selected_activity_id": "ACT-100",
            "current_view": "front",
        })
        if "STAGE" in response.get("answer", "__missing__"):
            raise SystemExit("unexpected placeholder answer")
        evidence = Path(response["evidence_package_path"])
        if not evidence.exists():
            raise SystemExit("missing evidence package")
        if len(response.get("generated_view_paths", {})) < 2:
            raise SystemExit("missing rendered views")
        print(f"STAGE_10_SMOKE_OK confidence={response['confidence']} evidence={response['evidence_package_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
