#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def deep_update(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch configs/site01.yaml with Stage 6 DA3 assistance defaults.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--additions", type=Path, default=Path("configs/stage_06_da3_additions.yaml"))
    parser.add_argument("--force-update-report-path", action="store_true")
    parser.add_argument("--force-safe-defaults", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    additions = yaml.safe_load(args.additions.read_text(encoding="utf-8"))
    deep_update(cfg, additions)

    run_id = cfg.get("project", {}).get("run_id", "2026-04-30_site01_baseline")
    if args.force_update_report_path:
        cfg.setdefault("paths", {})["da3_report_json"] = f"runs/{run_id}/reports/da3_summary.json"

    if args.force_safe_defaults:
        cfg.setdefault("da3", {})
        cfg["da3"]["enabled"] = "auto"
        cfg["da3"]["provider"] = cfg["da3"].get("provider", "precomputed")
        cfg["da3"]["fail_if_required_but_unavailable"] = False
        cfg["da3"]["fuse_aligned_depth"] = True

    args.config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"STAGE_06_CONFIG_PATCHED {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
