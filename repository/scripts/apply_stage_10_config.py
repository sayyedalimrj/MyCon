#!/usr/bin/env python3
"""Patch configs/site01.yaml with Stage 10 Copilot additions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Stage 10 Copilot config additions.")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--additions", default="configs/stage_10_copilot_additions.yaml")
    parser.add_argument("--force-safe-defaults", action="store_true")
    parser.add_argument("--force-local-vlm", action="store_true", help="Configure Ollama local/offline VLM instead of mock.")
    args = parser.parse_args()

    config_path = Path(args.config)
    additions_path = Path(args.additions)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    additions = yaml.safe_load(additions_path.read_text(encoding="utf-8"))
    _deep_update(cfg, additions)

    if args.force_safe_defaults:
        cfg.setdefault("copilot", {}).setdefault("vlm", {})["provider"] = "mock"
        cfg.setdefault("copilot", {}).setdefault("yolo", {})["enabled"] = True

    if args.force_local_vlm:
        vlm = cfg.setdefault("copilot", {}).setdefault("vlm", {})
        vlm["provider"] = "ollama_local"
        vlm["endpoint"] = "http://host.docker.internal:11434/api/chat"
        vlm["model"] = "qwen3-vl:8b"
        vlm["local_only"] = True
        vlm["fallback_to_mock_when_unavailable"] = False
        vlm["require_real_vlm"] = True

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"STAGE_10_CONFIG_PATCHED {config_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
