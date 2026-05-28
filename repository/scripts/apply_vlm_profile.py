from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a VLM profile YAML to a base config without downloading models.")
    parser.add_argument("--base", default="configs/site01.yaml")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_path = Path(args.base)
    profile_path = Path(args.profile)
    output_path = Path(args.output)

    if not base_path.exists():
        raise SystemExit(f"MISSING_BASE_CONFIG: {base_path}")
    if not profile_path.exists():
        raise SystemExit(f"MISSING_PROFILE_CONFIG: {profile_path}")
    if output_path.exists() and not args.force:
        raise SystemExit(f"REFUSING_TO_OVERWRITE_WITHOUT_FORCE: {output_path}")

    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))

    if not isinstance(base, dict) or not isinstance(profile, dict):
        raise SystemExit("INVALID_YAML_STRUCTURE")

    merged = deep_update(base, profile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print("VLM_PROFILE_APPLIED")
    print(f"base={base_path}")
    print(f"profile={profile_path}")
    print(f"output={output_path}")
    print("copilot.vlm.provider:", merged.get("copilot", {}).get("vlm", {}).get("provider"))
    print("copilot.vlm.model:", merged.get("copilot", {}).get("vlm", {}).get("model"))
    print("vlm_qa.provider:", merged.get("vlm_qa", {}).get("provider"))
    print("vlm_qa.model:", merged.get("vlm_qa", {}).get("model"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
