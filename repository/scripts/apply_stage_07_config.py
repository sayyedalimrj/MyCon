#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch configs/site01.yaml with Stage 7 cleanup settings.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--additions", type=Path, default=Path("configs/stage_07_cleanup_additions.yaml"))
    parser.add_argument("--force-update-report-path", action="store_true")
    parser.add_argument("--force-safe-defaults", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    additions = yaml.safe_load(args.additions.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not isinstance(additions, dict):
        raise SystemExit("Config and additions must be YAML mappings.")
    merged = _deep_merge(cfg, additions)

    run_id = merged.get("project", {}).get("run_id", "2026-04-30_site01_baseline")
    if args.force_update_report_path:
        merged.setdefault("paths", {})["cleanup_report_json"] = f"runs/{run_id}/reports/cleanup_summary.json"
        merged.setdefault("cleanup", {})["report_json"] = f"runs/{run_id}/reports/cleanup_summary.json"
        merged.setdefault("cleanup", {})["log_path"] = f"runs/{run_id}/logs/stage_07_cleanup.log"

    if args.force_safe_defaults:
        cleanup = merged.setdefault("cleanup", {})
        cleanup["strict_quality_gate"] = False
        cleanup["fail_on_quality_gate"] = False
        cleanup["mesh_enabled"] = True
        cleanup["mesh_method"] = "ball_pivoting"
        cleanup["fail_if_mesh_fails"] = False
        cleanup["normal_guided_plane_extraction"] = True
        cleanup["dynamic_voxel_enabled"] = True
        cleanup["normal_orientation_strategy"] = "none"
        cleanup["semantics_enabled"] = True
        cleanup["yolo_enabled"] = True
        cleanup["vlm_enabled"] = True
        cleanup["semantic_color_filter_enabled"] = True

    args.config.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    print(f"STAGE_07_CONFIG_PATCHED {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
