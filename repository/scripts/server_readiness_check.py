from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CheckResult:
    key: str
    status: str
    required_for: str
    message: str


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def _walk_get(data: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur



def _first_cfg(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = _walk_get(data, key, None)
        if value not in {None, ""}:
            return value
    return default


def _path_exists(root: Path, raw: str | None) -> bool:
    if not raw:
        return False
    p = Path(raw)
    if not p.is_absolute():
        p = root / p
    return p.exists()


def _which(name: str) -> str | None:
    return shutil.which(name)


def _command_ok(command: list[str], timeout: int = 8) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0, result.stdout[:800]
    except Exception as exc:
        return False, str(exc)


def build_readiness_report(config_path: Path, *, check_host_tools: bool = True) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    root = Path(_walk_get(cfg, "project.root", ".")).resolve()

    checks: list[CheckResult] = []

    def add(key: str, ok: bool, required_for: str, good: str, bad: str) -> None:
        checks.append(CheckResult(key, "ok" if ok else "missing", required_for, good if ok else bad))

    add(
        "config.site01",
        config_path.exists(),
        "all",
        f"Config exists: {config_path}",
        f"Config missing: {config_path}",
    )

    required_paths = {
        "sfm_images": _walk_get(cfg, "cams_gs.source_images_dir", "data/sfm/site01/images"),
        "dense_fused": "data/dense/site01/fused.ply",
        "cleaned_cloud": _walk_get(cfg, "paths.clean_cloud", "data/clean/site01/cleaned_cloud.ply"),
        "clean_mesh": _walk_get(cfg, "paths.clean_mesh", "data/clean/site01/mesh.ply"),
        "real_or_demo_ifc": _first_cfg(cfg, ["bim.ifc_path", "inputs.ifc", "bim.input_ifc"], "data/bim/design/model.ifc"),
        "metric_anchors": _first_cfg(cfg, ["metric_alignment.metric_anchors_csv", "metric_alignment.anchors_csv"], "data/bim/design/metric_anchors.csv"),
        "known_distances": _first_cfg(cfg, ["metric_alignment.known_distances_csv", "metric_alignment.distance_csv"], "data/bim/design/known_distances.csv"),
        "visual_observations": "data/bim/design/visual_anchor_observations.csv",
    }

    stage_map = {
        "sfm_images": "Stage 4.5 / server SfM",
        "dense_fused": "Stage 7",
        "cleaned_cloud": "Stage 7.5 / Stage 8",
        "clean_mesh": "Stage 7.5 / viewer",
        "real_or_demo_ifc": "Stage 8",
        "metric_anchors": "Stage 8 metric alignment",
        "known_distances": "Stage 8 metric alignment",
        "visual_observations": "server-only visual anchor triangulation",
    }

    for key, raw in required_paths.items():
        exists = _path_exists(root, raw)
        add(
            f"path.{key}",
            exists,
            stage_map.get(key, "pipeline"),
            f"Found {raw}",
            f"Missing {raw}",
        )

    vlm_model = _walk_get(cfg, "copilot.vlm.hf_model") or _walk_get(cfg, "copilot.vlm.model")
    expected_vlm = "Qwen/Qwen3-VL-8B-Thinking"
    checks.append(
        CheckResult(
            "vlm.qwen_profile",
            "ok" if str(vlm_model) == expected_vlm or "8b-thinking" in str(vlm_model).lower() else "warn",
            "Stage 10 real VLM",
            f"Configured VLM model: {vlm_model}",
        )
    )

    cache_root = _walk_get(cfg, "model_cache.root_dir", "model_cache")
    checks.append(
        CheckResult(
            "model_cache.root_dir",
            "ok" if cache_root else "missing",
            "server model download/cache",
            f"Configured model cache root: {cache_root}",
        )
    )

    if check_host_tools:
        for tool, required_for in [
            ("nvidia-smi", "server GPU validation"),
            ("ollama", "local Qwen VLM optional"),
            ("ns-train", "3DGS/CAMS-GS optional"),
            ("ns-export", "3DGS/CAMS-GS optional"),
            ("PotreeConverter", "Potree viewer optional"),
            ("pdal", "point cloud conversion optional"),
        ]:
            found = _which(tool)
            checks.append(
                CheckResult(
                    f"tool.{tool}",
                    "ok" if found else "missing",
                    required_for,
                    f"{tool}: {found}" if found else f"{tool}: not found in PATH",
                )
            )

        ok, output = _command_ok(["nvidia-smi"], timeout=5)
        checks.append(
            CheckResult(
                "gpu.nvidia_smi_runtime",
                "ok" if ok else "missing",
                "server GPU validation",
                "nvidia-smi runs successfully" if ok else f"nvidia-smi failed or unavailable: {output[:300]}",
            )
        )

    server_required_missing = [
        item.key
        for item in checks
        if item.status == "missing"
        and item.required_for
        in {
            "Stage 8",
            "server-only visual anchor triangulation",
            "server GPU validation",
            "Stage 10 real VLM",
        }
    ]

    payload = {
        "status": "ready_or_laptop_stub" if not server_required_missing else "server_inputs_missing",
        "config": str(config_path),
        "project_root": str(root),
        "checks": [asdict(c) for c in checks],
        "summary": {
            "ok": sum(1 for c in checks if c.status == "ok"),
            "warn": sum(1 for c in checks if c.status == "warn"),
            "missing": sum(1 for c in checks if c.status == "missing"),
            "server_required_missing": server_required_missing,
        },
        "notes": [
            "This script never downloads models.",
            "This script never runs dense reconstruction, 3DGS training, or VLM inference.",
            "Missing visual_anchor_observations.csv is expected on laptop and must be resolved on server/project data.",
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Check server readiness without downloads or heavy processing.")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--output", default="runs/2026-04-30_site01_baseline/reports/server_readiness_report.json")
    parser.add_argument("--no-host-tools", action="store_true")
    args = parser.parse_args()

    report = build_readiness_report(Path(args.config), check_host_tools=not args.no_host_tools)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "SERVER_READINESS_CHECK_OK "
        f"status={report['status']} "
        f"ok={report['summary']['ok']} "
        f"missing={report['summary']['missing']} "
        f"output={output}"
    )
    if report["summary"]["server_required_missing"]:
        print("server_required_missing=" + ",".join(report["summary"]["server_required_missing"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
