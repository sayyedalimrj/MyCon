from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from pipeline.common.config import load_config

from .config_access import cfg_get, cfg_int, project_name, run_id
from .input_selection import missing_required_inputs, stage75_paths
from .io_utils import clean_dir_guarded, write_json_atomic
from .qa_metrics import collect_stage75_metrics, evaluate_stage75_quality
from .rendering import render_stage75_views


def _deterministic_mock_observations(metrics: dict[str, Any], quality_gate: dict[str, Any]) -> list[str]:
    cloud = metrics.get("cleaned_cloud", {})
    mesh = metrics.get("mesh", {})
    planes = metrics.get("planes", {})
    observations = [
        f"Cleaned cloud contains {cloud.get('point_count')} points.",
        f"Finite point ratio is {cloud.get('finite_ratio')}.",
        f"Mesh status is {mesh.get('status')}.",
        f"Detected plane count is {planes.get('plane_count')}.",
        f"QA gate status is {quality_gate.get('status')} with confidence {quality_gate.get('confidence')}.",
    ]
    if quality_gate.get("warnings"):
        observations.append("Warnings should be reviewed before BIM registration.")
    if quality_gate.get("failures"):
        observations.append("Failures indicate the reconstruction should not proceed to BIM registration without review.")
    return observations


def run_vlm_qa(cfg: Any, *, force: bool = False, log_level: str = "INFO") -> dict[str, Any]:
    started = time.perf_counter()
    paths = stage75_paths(cfg)

    missing = missing_required_inputs(paths)
    if missing:
        raise RuntimeError("Missing Stage 7.5 inputs: " + "; ".join(missing))

    clean_dir_guarded(paths["output_dir"], force=force, required_token="vlm_qa")

    metrics = collect_stage75_metrics(paths)
    quality_gate = evaluate_stage75_quality(cfg, metrics)
    max_render_points = cfg_int(cfg, "vlm_qa.max_render_points", 50000)
    render_paths = render_stage75_views(paths, metrics, quality_gate, max_points=max_render_points)

    provider = str(cfg_get(cfg, "vlm_qa.provider", cfg_get(cfg, "copilot.vlm.provider", "mock")))
    evidence = {
        "stage": "stage_07_5_vlm_qa",
        "project": {"name": project_name(cfg), "run_id": run_id(cfg)},
        "provider": provider,
        "status": quality_gate["status"],
        "confidence": quality_gate["confidence"],
        "inputs": {
            "cleaned_cloud": paths["cleaned_cloud"].as_posix(),
            "mesh": paths["mesh"].as_posix() if paths.get("mesh") else None,
            "planes_json": paths["planes_json"].as_posix() if paths.get("planes_json") else None,
            "cleanup_report": paths["cleanup_report"].as_posix(),
        },
        "render_paths": render_paths,
        "metrics": metrics,
        "quality_gate": quality_gate,
        "mock_observations": _deterministic_mock_observations(metrics, quality_gate),
        "interpretation_rule": "This Stage 7.5 QA is pre-BIM visual/geometric evidence. It does not prove construction progress by itself.",
        "elapsed_sec": time.perf_counter() - started,
    }

    summary = {
        "stage": evidence["stage"],
        "project": evidence["project"],
        "status": evidence["status"],
        "confidence": evidence["confidence"],
        "quality_gate": quality_gate,
        "render_paths": render_paths,
        "evidence_json": paths["evidence_json"].as_posix(),
        "elapsed_sec": evidence["elapsed_sec"],
    }

    write_json_atomic(paths["evidence_json"], evidence)
    write_json_atomic(paths["summary_json"], summary)

    print(
        "STAGE_07_5_VLM_QA_OK "
        f"status={summary['status']} confidence={summary['confidence']} "
        f"evidence={paths['evidence_json']}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 7.5: VLM/visual QA for cleaned reconstruction outputs")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        run_vlm_qa(cfg, force=args.force, log_level=args.log_level)
    except Exception as exc:
        print(f"STAGE_07_5_VLM_QA_FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
