from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ReadinessItem:
    key: str
    status: str
    required_on_server: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cfg_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _project_root(cfg: dict[str, Any]) -> Path:
    return Path(str(_cfg_get(cfg, "project.root", "."))).resolve()


def _resolve(root: Path, raw: Any) -> Path | None:
    if raw in (None, ""):
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def _first_existing_config_path(cfg: dict[str, Any], keys: list[str]) -> tuple[Path | None, str | None]:
    root = _project_root(cfg)
    for key in keys:
        raw = _cfg_get(cfg, key)
        path = _resolve(root, raw)
        if path is not None:
            return path, key
    return None, None


def _path_item(
    cfg: dict[str, Any],
    key: str,
    keys: list[str],
    required_on_server: bool,
    *,
    default_path: str | None = None,
) -> ReadinessItem:
    path, source_key = _first_existing_config_path(cfg, keys)

    if path is None and default_path:
        root = _project_root(cfg)
        fallback = _resolve(root, default_path)
        if fallback is not None:
            if fallback.exists():
                return ReadinessItem(key, "ok", required_on_server, f"default={fallback}")
            return ReadinessItem(key, "missing_path", required_on_server, f"default={fallback}")

    if path is None:
        return ReadinessItem(key, "missing_config_key", required_on_server, "checked=" + ",".join(keys))

    if path.exists():
        return ReadinessItem(key, "ok", required_on_server, f"{source_key}={path}")

    return ReadinessItem(key, "missing_path", required_on_server, f"{source_key}={path}")


def _tool_item(tool: str, required_on_server: bool) -> ReadinessItem:
    found = shutil.which(tool)
    if found:
        return ReadinessItem(f"tool.{tool}", "ok", required_on_server, found)
    return ReadinessItem(f"tool.{tool}", "missing_tool", required_on_server, tool)


def _gpu_item() -> ReadinessItem:
    nvidia = shutil.which("nvidia-smi")
    if not nvidia:
        return ReadinessItem("gpu.nvidia_smi_runtime", "missing_tool", True, "nvidia-smi not found")

    try:
        result = subprocess.run(
            [nvidia, "-L"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return ReadinessItem("gpu.nvidia_smi_runtime", "error", True, str(exc))

    output = result.stdout.strip()
    if result.returncode == 0 and output:
        return ReadinessItem("gpu.nvidia_smi_runtime", "ok", True, output.splitlines()[0])
    return ReadinessItem("gpu.nvidia_smi_runtime", "unavailable", True, output or "nvidia-smi returned no GPU")


def build_server_readiness_gate(config_path: Path) -> dict[str, Any]:
    cfg = _load_config(config_path)

    items: list[ReadinessItem] = [
        _path_item(
            cfg,
            "path.raw_video_or_images",
            [
                "inputs.video",
                "inputs.raw_video",
                "paths.raw_video",
                "ingest.input_video",
                "ingest.video_path",
                "paths.images_dir",
                "keyframes.images_dir",
            ],
            True,
        ),
        _path_item(
            cfg,
            "path.ifc",
            [
                "inputs.ifc",
                "bim.ifc_path",
                "bim.design_ifc",
                "paths.ifc",
                "paths.ifc_path",
            ],
            True,
        ),
        _path_item(
            cfg,
            "path.schedule_csv",
            [
                "inputs.schedule_csv",
                "schedule.csv",
                "progress.schedule_csv",
                "bim.schedule_filter_csv",
                "paths.schedule_csv",
            ],
            True,
        ),
        _path_item(
                cfg,
                "path.element_activity_map",
                [
                    "inputs.element_activity_map_csv",
                    "inputs.element_activity_map",
                    "progress.element_activity_map_csv",
                    "progress.element_activity_map",
                    "paths.element_activity_map_csv",
                    "paths.element_activity_map",
                    "paths.activity_map_csv",
                    "bim.element_activity_map_csv",
                    "bim.element_activity_map",
                    "bim.activity_map_csv",
                ],
                True,
                default_path="data/bim/design/element_activity_map.csv",
            ),
        _path_item(
            cfg,
            "path.metric_anchors_csv",
            [
                "metric_alignment.metric_anchors_csv",
                "metric_alignment.anchors_csv",
                "paths.metric_anchors_csv",
                "inputs.metric_anchors_csv",
            ],
            False,
        ),
        _path_item(
            cfg,
            "path.visual_observations",
            [
                "metric_alignment.visual_anchor_observations_csv",
                "visual_anchor_observations.csv",
                "paths.visual_anchor_observations_csv",
                "inputs.visual_anchor_observations_csv",
            ],
            False,
        ),
        _path_item(
            cfg,
            "path.known_distances_csv",
            [
                "metric_alignment.known_distances_csv",
                "metric_alignment.distance_csv",
                "paths.known_distances_csv",
                "inputs.known_distances_csv",
            ],
            False,
        ),
    ]

    items.extend(
        [
            _tool_item("python3", True),
            _tool_item("docker", False),
            _tool_item("nvidia-smi", True),
            _gpu_item(),
        ]
    )

    model = (
        _cfg_get(cfg, "copilot.vlm.model")
        or _cfg_get(cfg, "model_cache.qwen_vlm.ollama_model")
        or ""
    )
    hf_model = (
        _cfg_get(cfg, "copilot.vlm.hf_model")
        or _cfg_get(cfg, "model_cache.qwen_vlm.hf_model")
        or ""
    )

    if "qwen3-vl:8b-thinking" in str(model).lower() or "qwen/qwen3-vl-8b-thinking" in str(hf_model).lower():
        items.append(ReadinessItem("model.qwen3_vl_8b_thinking_config", "ok", True, f"ollama={model} hf={hf_model}"))
    else:
        items.append(ReadinessItem("model.qwen3_vl_8b_thinking_config", "missing_or_mismatch", True, f"ollama={model} hf={hf_model}"))

    status_by_key = {item.key: item.status for item in items}
    has_metric_route = (
        status_by_key.get("path.metric_anchors_csv") == "ok"
        or status_by_key.get("path.visual_observations") == "ok"
    )

    if has_metric_route:
        items.append(ReadinessItem("metricization.route", "ok", True, "metric anchors or visual observations available"))
    else:
        items.append(ReadinessItem("metricization.route", "missing_metric_route", True, "need metric_anchors_csv or visual_anchor_observations_csv"))

    server_blockers = [
        item.to_dict()
        for item in items
        if item.required_on_server and item.status not in {"ok"}
    ]

    warnings = [
        item.to_dict()
        for item in items
        if not item.required_on_server and item.status not in {"ok"}
    ]

    return {
        "stage": "server_readiness_strict_gate",
        "status": "ready" if not server_blockers else "server_inputs_missing",
        "passed": not server_blockers,
        "config_path": str(config_path),
        "items": [item.to_dict() for item in items],
        "server_blockers": server_blockers,
        "warnings": warnings,
        "summary": {
            "ok": sum(1 for item in items if item.status == "ok"),
            "server_blocker_count": len(server_blockers),
            "warning_count": len(warnings),
        },
    }


def write_server_readiness_gate_report(config_path: Path, output_json: Path) -> dict[str, Any]:
    report = build_server_readiness_gate(config_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
