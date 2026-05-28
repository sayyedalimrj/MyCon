#!/usr/bin/env python3
"""Patch a project config with Stage 8 BIM registration defaults."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


STAGE8_DEFAULTS: dict[str, Any] = {
    "paths": {"bim_aligned_dir": "data/bim/aligned/site01"},
    "bim": {
        "enabled": True,
        "units": "meters",
        "scan_input_candidates": [
            "data/clean/site01/cleaned_cloud.ply",
            "data/clean/site01/clean.ply",
            "data/clean/site01/cleaned_cloud.pcd",
            "data/da3/site01/da3_assisted_points.ply",
            "data/dense/site01/fused.ply",
        ],
        "bim_reference_ply": "data/bim/aligned/site01/bim_reference.ply",
        "bim_reference_mesh": "data/bim/aligned/site01/bim_reference_mesh.ply",
        "scan_aligned_ply": "data/bim/aligned/site01/scan_aligned.ply",
        "transform_json": "data/bim/aligned/site01/transform_scan_to_bim.json",
        "element_metadata_jsonl": "data/bim/aligned/site01/bim_elements.jsonl",
        "registration_report_json": "runs/2026-04-30_site01_baseline/reports/registration_report.json",
        "element_classes": ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcDoor", "IfcWindow", "IfcStair", "IfcRailing"],
        "bim_sample_points": 200000,
        "allow_synthetic_ifc_fallback_for_tests": False,
        "initial_scale_strategy": "fixed_1",
        "known_initial_scale": 1.0,
        "estimate_initial_scale_from_bbox": False,
        "min_initial_scale": 0.01,
        "max_initial_scale": 100.0,
        "coarse_fpfh_enabled": True,
        "coarse_voxel_size_m": 0.0,
        "coarse_distance_multiplier": 2.5,
        "coarse_min_fitness_accept": 0.05,
        "visible_shell_filter_enabled": False,
        "visible_shell_camera_radius_multiplier": 2.0,
        "visible_shell_hpr_radius_multiplier": 20.0,
        "visible_shell_min_keep_ratio": 0.20,
        "visible_shell_max_removed_ratio": 0.80,
        "schedule_filter_enabled": False,
        "schedule_filter_csv": "data/bim/design/schedule.csv",
        "schedule_filter_keep_unmatched": True,
        "current_project_day": None,
        "schedule_filter_allowed_statuses": ["done", "complete", "completed", "in_progress", "started", "active"],
        "icp_enabled": True,
        "icp_estimation": "point_to_point_then_plane",
        "icp_stages": ["point_to_point", "point_to_plane"],
        "icp_voxel_size_m": 0.0,
        "icp_max_corr_distance_m": 0.08,
        "icp_normal_radius_m": 0.08,
        "icp_normal_max_nn": 40,
        "icp_max_iteration": 80,
        "icp_point_to_point_max_iteration": 80,
        "icp_point_to_plane_max_iteration": 25,
        "icp_relative_fitness": 1.0e-6,
        "icp_relative_rmse": 1.0e-6,
        "quality_min_icp_fitness": 0.05,
        "quality_max_icp_rmse_m": 0.25,
        "quality_scale_warning_ratio": 0.10,
        "quality_nn_sample_limit": 200000,
        "fail_on_low_registration_quality": False,
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


def apply_safe_stage8_defaults(data: dict[str, Any]) -> None:
    name = str(data.get("project", {}).get("name", "site01"))
    run_id = str(data.get("project", {}).get("run_id", "2026-04-30_site01_baseline"))
    paths = data.setdefault("paths", {})
    bim = data.setdefault("bim", {})
    paths["bim_aligned_dir"] = f"data/bim/aligned/{name}"
    bim["bim_reference_ply"] = f"data/bim/aligned/{name}/bim_reference.ply"
    bim["bim_reference_mesh"] = f"data/bim/aligned/{name}/bim_reference_mesh.ply"
    bim["scan_aligned_ply"] = f"data/bim/aligned/{name}/scan_aligned.ply"
    bim["transform_json"] = f"data/bim/aligned/{name}/transform_scan_to_bim.json"
    bim["element_metadata_jsonl"] = f"data/bim/aligned/{name}/bim_elements.jsonl"
    bim["registration_report_json"] = f"runs/{run_id}/reports/registration_report.json"
    bim["scan_input_candidates"] = [
        f"data/clean/{name}/cleaned_cloud.ply",
        f"data/clean/{name}/clean.ply",
        f"data/clean/{name}/cleaned_cloud.pcd",
        f"data/da3/{name}/da3_assisted_points.ply",
        f"data/dense/{name}/fused.ply",
    ]
    bim["allow_synthetic_ifc_fallback_for_tests"] = False
    bim["initial_scale_strategy"] = "fixed_1"
    bim["known_initial_scale"] = 1.0
    bim["estimate_initial_scale_from_bbox"] = False
    bim["coarse_fpfh_enabled"] = True
    bim["visible_shell_filter_enabled"] = False
    bim["schedule_filter_enabled"] = False
    bim["schedule_filter_csv"] = "data/bim/design/schedule.csv"
    bim["schedule_filter_keep_unmatched"] = True
    bim["icp_enabled"] = True
    bim["icp_estimation"] = "point_to_point_then_plane"
    bim["icp_stages"] = ["point_to_point", "point_to_plane"]
    bim["icp_max_corr_distance_m"] = 0.08
    bim["icp_point_to_point_max_iteration"] = 80
    bim["icp_point_to_plane_max_iteration"] = 25
    bim["fail_on_low_registration_quality"] = False


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Stage 8 BIM registration config defaults")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--force-update-report-path", action="store_true")
    parser.add_argument("--force-safe-defaults", action="store_true")
    args = parser.parse_args()
    path = Path(args.config)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    deep_merge_missing(data, STAGE8_DEFAULTS)
    if args.force_safe_defaults:
        apply_safe_stage8_defaults(data)
    if args.force_update_report_path:
        apply_safe_stage8_defaults(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"STAGE_08_CONFIG_PATCHED {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
