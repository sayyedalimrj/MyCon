#!/usr/bin/env python3
"""Patch configs/site01.yaml with Stage 4 sparse refinement defaults."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

STAGE4_DEFAULTS: dict[str, Any] = {
    "paths": {
        "sparse_refined_dir": "data/sparse_refined/site01",
    },
    "refinement": {
        "enabled": True,
        "method": "colmap_bundle_adjustment",
        "input_sparse_dir": "data/sparse/site01/0",
        "output_sparse_dir": "data/sparse_refined/site01/0",
        "work_dir": "data/sparse_refined/site01/_work",
        "report_json": "runs/2026-04-30_site01_baseline/reports/refinement_stats.json",
        "command_history_json": "data/sparse_refined/site01/command_history.json",
        "validate_binary_before_refinement": True,
        "keep_work_dir": True,
        "ba_max_num_iterations": 100,
        "ba_max_linear_solver_iterations": 200,
        "ba_function_tolerance": 1.0e-6,
        "ba_gradient_tolerance": 1.0e-10,
        "ba_parameter_tolerance": 1.0e-8,
        "refine_focal_length": True,
        "refine_principal_point": False,
        "refine_extra_params": True,
        "quality_gate_min_registered_images": 2,
        "quality_gate_min_points": 1,
        "quality_gate_max_point_loss_ratio": 0.40,
        "quality_gate_fail_on_point_loss": True,
        "quality_gate_max_reprojection_error_increase_ratio": 0.10,
        "quality_gate_max_reprojection_error_increase_abs_px": 0.25,
        "quality_gate_fail_on_reprojection_error_increase": True,
        "ba_rounds": 1,
        "ba_num_threads": -1,
        "fail_on_quality_gate": True,
        "pixsfm_enabled": False,
        "pixsfm_allow_missing": True,
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


def apply_safe_stage4_defaults(data: dict[str, Any]) -> None:
    project_name = str(data.get("project", {}).get("name", "site01"))
    run_id = str(data.get("project", {}).get("run_id", "2026-04-30_site01_baseline"))
    paths = data.setdefault("paths", {})
    refinement = data.setdefault("refinement", {})
    paths.setdefault("sparse_refined_dir", f"data/sparse_refined/{project_name}")
    refinement["method"] = "colmap_bundle_adjustment"
    refinement["input_sparse_dir"] = f"data/sparse/{project_name}/0"
    refinement["output_sparse_dir"] = f"data/sparse_refined/{project_name}/0"
    refinement["work_dir"] = f"data/sparse_refined/{project_name}/_work"
    refinement["report_json"] = f"runs/{run_id}/reports/refinement_stats.json"
    refinement["command_history_json"] = f"data/sparse_refined/{project_name}/command_history.json"
    refinement["validate_binary_before_refinement"] = True
    refinement["ba_rounds"] = 1
    refinement["ba_num_threads"] = -1
    refinement["refine_principal_point"] = False
    refinement["quality_gate_min_registered_images"] = 2
    refinement["quality_gate_min_points"] = 1
    refinement["quality_gate_max_point_loss_ratio"] = 0.40
    refinement["quality_gate_fail_on_point_loss"] = True
    refinement["quality_gate_max_reprojection_error_increase_ratio"] = 0.10
    refinement["quality_gate_max_reprojection_error_increase_abs_px"] = 0.25
    refinement["quality_gate_fail_on_reprojection_error_increase"] = True
    refinement["pixsfm_enabled"] = False
    refinement["pixsfm_allow_missing"] = True
    refinement["fail_on_quality_gate"] = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Stage 4 refinement config defaults")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--force-update-report-path", action="store_true", help="Update report paths to current project.run_id")
    parser.add_argument("--force-safe-defaults", action="store_true", help="Overwrite reviewed safe Stage 4 defaults")
    args = parser.parse_args()

    path = Path(args.config)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    deep_merge_missing(data, STAGE4_DEFAULTS)
    if args.force_safe_defaults:
        apply_safe_stage4_defaults(data)
    if args.force_update_report_path:
        name = str(data.get("project", {}).get("name", "site01"))
        rid = str(data.get("project", {}).get("run_id", "2026-04-30_site01_baseline"))
        data.setdefault("paths", {})["sparse_refined_dir"] = f"data/sparse_refined/{name}"
        data.setdefault("refinement", {})["report_json"] = f"runs/{rid}/reports/refinement_stats.json"
        data.setdefault("refinement", {})["command_history_json"] = f"data/sparse_refined/{name}/command_history.json"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"STAGE_04_CONFIG_PATCHED {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
