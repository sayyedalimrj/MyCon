#!/usr/bin/env python3
"""Patch configs/site01.yaml with Stage 5 dense stereo settings."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _set_default(mapping: dict[str, Any], key: str, value: Any, force: bool) -> None:
    if force or key not in mapping:
        mapping[key] = value


def patch_config(path: Path, force_safe_defaults: bool = False, force_update_report_path: bool = False) -> None:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config is not a YAML mapping: {path}")
    project = cfg.setdefault("project", {})
    run_id = str(project.get("run_id", "2026-04-30_site01_baseline"))
    paths = cfg.setdefault("paths", {})
    dense = cfg.setdefault("dense", {})

    _set_default(paths, "dense_workspace", "data/dense/site01", force_safe_defaults)
    _set_default(paths, "fused_ply", "data/dense/site01/fused.ply", force_safe_defaults)

    defaults = {
        "enabled": True,
        "input_sparse_refined_dir": "data/sparse_refined/site01/0",
        "input_images_dir": "data/sfm/site01/images",
        "workspace_dir": "data/dense/site01",
        "fused_ply": "data/dense/site01/fused.ply",
        "workspace_format": "COLMAP",
        "output_type": "COLMAP",
        "report_json": f"runs/{run_id}/reports/dense_summary.json",
        "command_history_json": "data/dense/site01/command_history.json",
        "min_input_images": 2,
        "require_cuda": True,
        "cuda_preflight": True,
        "gpu_preflight": True,
        "require_visible_gpu": False,
        "adaptive_gpu_profile": True,
        "adaptive_image_count_caps": True,
        "patch_match_gpu_index": "auto",
        "patch_match_num_threads": -1,
        "num_patch_match_src_images": 15,
        "max_image_size": 1600,
        "patch_match_max_image_size": 1600,
        "patch_window_radius": 5,
        "patch_window_step": 1,
        "patch_num_samples": 15,
        "patch_num_iterations": 5,
        "patch_match_cache_size": 32,
        "geom_consistency": True,
        "geom_consistency_regularizer": 0.3,
        "geom_consistency_max_cost": 5.0,
        "patch_filter": True,
        "filter_min_ncc": 0.05,
        "filter_min_triangulation_angle": 3.0,
        "filter_min_num_consistent": 2,
        "filter_geom_consistency_max_cost": 2.0,
        "fusion_input_type": "geometric",
        "fusion_max_image_size": -1,
        "fusion_min_num_pixels": 3,
        "fusion_max_reproj_error": 2.0,
        "fusion_max_depth_error": 0.02,
        "quality_min_fused_points": 1000,
        "quality_min_fused_points_per_image": 20.0,
        "quality_min_depth_map_ratio_warning": 0.25,
        "fail_on_quality_gate": True,
        "use_existing_masks": False,
        "mask_path": "data/masks/site01",
        "require_masks": False,
        "fusion_bounding_box": None,
        "keep_workspace": True,
    }
    for key, value in defaults.items():
        force = force_safe_defaults or (force_update_report_path and key in {"report_json", "command_history_json"})
        _set_default(dense, key, value, force)

    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch YAML config with Stage 5 dense stereo defaults")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--force-update-report-path", action="store_true")
    parser.add_argument("--force-safe-defaults", action="store_true")
    args = parser.parse_args()
    patch_config(Path(args.config), args.force_safe_defaults, args.force_update_report_path)
    print(f"STAGE_05_CONFIG_PATCHED {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
