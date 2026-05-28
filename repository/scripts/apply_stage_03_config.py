#!/usr/bin/env python3
"""Patch configs/site01.yaml with Stage 3 COLMAP defaults without overwriting Stage 1/2 keys."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

STAGE3_DEFAULTS: dict[str, Any] = {
    "paths": {
        "sfm_dir": "data/sfm/site01",
        "colmap_db": "data/sfm/site01/database.db",
        "sparse_dir": "data/sparse/site01",
        "sparse_report_json": "runs/2026-04-30_site01_baseline/reports/sparse_stats.json",
        "sparse_mask_dir": "data/masks/site01",
    },
    "colmap": {
        "executable": "colmap",
        "camera_model": "SIMPLE_RADIAL",
        "single_camera": True,
        "feature_type": "ALIKED_N16ROT",
        "matcher_type": "ALIKED_LIGHTGLUE",
        "fallback_feature_type": "SIFT",
        "fallback_matcher_type": "SIFT_LIGHTGLUE",
        "enable_fallback": True,
        "allow_sift_bruteforce_emergency": False,
        "emergency_matcher_type": "SIFT_BRUTEFORCE",
        "matching_strategy": "sequential",
        "sequential_overlap": 15,
        "sequential_quadratic_overlap": True,
        "sequential_loop_detection": False,
        "sequential_vocab_tree_path": None,
        "min_input_images": 2,
        "stage_images_mode": "copy",
        "aliked_max_num_features": 2048,
        "sift_max_num_features": 2048,
        "mapper_min_num_matches": 15,
        "mapper_multiple_models": True,
        "mapper_extract_colors": False,
        "qt_qpa_platform": "offscreen",
        "download_models": True,
        "model_cache_dir": "data/sfm/model_cache",
        "model_download_timeout_sec": 1800,
        "use_existing_masks": False,
        "mask_path": "data/masks/site01",
        "require_masks": False,
    },
}


def deep_merge_missing(target: dict[str, Any], defaults: dict[str, Any]) -> None:
    for key, value in defaults.items():
        if isinstance(value, dict):
            existing = target.setdefault(key, {})
            if not isinstance(existing, dict):
                target[key] = {}
                existing = target[key]
            deep_merge_missing(existing, value)
        else:
            target.setdefault(key, value)


def apply_safe_stage3_defaults(data: dict[str, Any]) -> None:
    """Overwrite only the safe Stage 3 baseline keys that changed after review."""
    paths = data.setdefault("paths", {})
    colmap = data.setdefault("colmap", {})
    paths.setdefault("sparse_mask_dir", "data/masks/site01")
    colmap["stage_images_mode"] = "copy"
    colmap["aliked_max_num_features"] = 2048
    colmap["sift_max_num_features"] = 2048
    colmap["sequential_overlap"] = 15
    colmap.setdefault("sequential_quadratic_overlap", True)
    colmap.setdefault("sequential_loop_detection", False)
    colmap.setdefault("sequential_vocab_tree_path", None)
    colmap.setdefault("use_existing_masks", False)
    colmap.setdefault("mask_path", "data/masks/site01")
    colmap.setdefault("require_masks", False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Stage 3 config defaults")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--force-update-report-path", action="store_true", help="Update sparse_report_json to current project.run_id")
    parser.add_argument("--force-safe-defaults", action="store_true", help="Overwrite reviewed safe Stage 3 defaults such as feature limits and copy mode")
    args = parser.parse_args()
    path = Path(args.config)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    deep_merge_missing(data, STAGE3_DEFAULTS)
    if args.force_safe_defaults:
        apply_safe_stage3_defaults(data)
    if args.force_update_report_path:
        run_id = data.get("project", {}).get("run_id", "2026-04-30_site01_baseline")
        data.setdefault("paths", {})["sparse_report_json"] = f"runs/{run_id}/reports/sparse_stats.json"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"STAGE_03_CONFIG_PATCHED {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
