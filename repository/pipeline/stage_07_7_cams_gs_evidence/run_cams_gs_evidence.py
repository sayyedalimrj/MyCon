from __future__ import annotations

import argparse
import html
import logging
import time
from pathlib import Path
from typing import Any

from .config_access import project_name, run_id, stage77_paths
from .input_selection import select_inputs
from .io_utils import ensure_dir, file_size, write_json_atomic

LOGGER_NAME = "pipeline.stage_07_7_cams_gs_evidence"


def _load_config(path: Path) -> Any:
    from pipeline.common.config import load_config

    return load_config(path)


def _status_from_inputs(inputs: Any) -> tuple[str, str]:
    stage45_status = str(inputs.stage45_manifest.get("status") or inputs.training_status.get("status") or "missing")
    training_executed = bool(inputs.training_status.get("training_executed", False))

    if not inputs.stage45_manifest:
        return "skipped_missing_stage45_manifest", "missing"
    if stage45_status.startswith("skipped"):
        return "skipped_stage45_not_prepared", "missing"
    if training_executed:
        return "ready_for_gaussian_viewer", "trained_or_training_available"
    return "prepared_no_training", "prepared_stub_only"


def _write_html(path: Path, evidence: dict[str, Any]) -> None:
    ensure_dir(path.parent)

    rows = []
    for key, value in evidence.get("key_facts", {}).items():
        rows.append(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>")

    warnings = evidence.get("warnings", [])
    warning_items = "\n".join(f"<li>{html.escape(str(w))}</li>" for w in warnings) or "<li>None</li>"

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CAMS-GS / 3DGS Evidence</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ width: 260px; background: #f7f7f7; }}
    code {{ background: #f3f3f3; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>CAMS-GS / 3DGS Evidence Package</h1>
  <p>This package is for visualization readiness only. It is not metric truth.</p>

  <div class="card">
    <h2>Status</h2>
    <table>{''.join(rows)}</table>
  </div>

  <div class="card">
    <h2>Warnings</h2>
    <ul>{warning_items}</ul>
  </div>

  <div class="card">
    <h2>Next action</h2>
    <p>{html.escape(str(evidence.get("recommended_next_action", "")))}</p>
  </div>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def run_cams_gs_evidence(cfg: Any, *, force: bool = False, log_level: str = "INFO") -> dict[str, Any]:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    logger = logging.getLogger(LOGGER_NAME)

    paths = stage77_paths(cfg)
    ensure_dir(paths["output_dir"])
    ensure_dir(paths["viewer_html"].parent)

    inputs = select_inputs(cfg)
    status, readiness = _status_from_inputs(inputs)

    stage45_inputs = inputs.stage45_manifest.get("inputs", {}) if inputs.stage45_manifest else {}
    stage45_training = inputs.stage45_manifest.get("training", {}) if inputs.stage45_manifest else {}

    warnings = list(inputs.warnings)
    if status == "prepared_no_training":
        warnings.append("cams_gs_training_not_executed_visual_viewer_not_available_yet")

    recommended_next_action = (
        "Install/connect Nerfstudio Splatfacto or another 3DGS trainer, run training, then regenerate this evidence package."
        if status == "prepared_no_training"
        else "Review the generated Gaussian viewer artifacts."
        if status == "ready_for_gaussian_viewer"
        else "Run Stage 4.5 CAMS-GS prepare first."
    )

    evidence = {
        "stage": "stage_07_7_cams_gs_evidence",
        "status": status,
        "project": project_name(cfg),
        "run_id": run_id(cfg),
        "purpose": "optional_gaussian_visualization_evidence",
        "is_metric_truth": False,
        "readiness": readiness,
        "key_facts": {
            "stage45_status": inputs.stage45_manifest.get("status") or inputs.training_status.get("status") or "missing",
            "image_count": stage45_inputs.get("image_count"),
            "selected_image_count": stage45_inputs.get("selected_image_count"),
            "training_executed": inputs.training_status.get("training_executed", False),
            "nerfstudio_method": stage45_training.get("nerfstudio_method"),
            "viewer_export_artifact_count": len(inputs.viewer_export_manifest.get("artifacts", [])) if inputs.viewer_export_manifest else 0,
        },
        "inputs": {
            "stage45_manifest_json": paths["stage45_manifest_json"].as_posix(),
            "stage45_training_status_json": paths["stage45_training_status_json"].as_posix(),
            "stage45_dataset_dir": paths["stage45_dataset_dir"].as_posix(),
            "viewer_export_manifest_json": paths["viewer_export_manifest_json"].as_posix(),
        },
        "outputs": {
            "evidence_json": paths["evidence_json"].as_posix(),
            "summary_json": paths["summary_json"].as_posix(),
            "viewer_html": paths["viewer_html"].as_posix(),
            "viewer_manifest_json": paths["viewer_manifest_json"].as_posix(),
        },
        "stage45_manifest_preview": {
            "status": inputs.stage45_manifest.get("status"),
            "manifest_json": inputs.stage45_manifest_path.as_posix(),
            "dataset_dir": stage45_inputs.get("dataset_dir") or inputs.dataset_dir.as_posix(),
        },
        "warnings": warnings,
        "recommended_next_action": recommended_next_action,
        "created_at_unix": time.time(),
    }

    viewer_manifest = {
        "stage": evidence["stage"],
        "status": status,
        "is_metric_truth": False,
        "viewer_html": paths["viewer_html"].as_posix(),
        "evidence_json": paths["evidence_json"].as_posix(),
        "artifacts": [
            {
                "key": "cams_gs_train_manifest",
                "kind": "json",
                "path": paths["stage45_manifest_json"].as_posix(),
                "exists": paths["stage45_manifest_json"].exists(),
                "size_bytes": file_size(paths["stage45_manifest_json"]),
            },
            {
                "key": "cams_gs_training_status",
                "kind": "json",
                "path": paths["stage45_training_status_json"].as_posix(),
                "exists": paths["stage45_training_status_json"].exists(),
                "size_bytes": file_size(paths["stage45_training_status_json"]),
            },
            {
                "key": "cams_gs_dataset_dir",
                "kind": "directory",
                "path": paths["stage45_dataset_dir"].as_posix(),
                "exists": paths["stage45_dataset_dir"].exists(),
                "size_bytes": 0,
            },
        ],
        "created_at_unix": evidence["created_at_unix"],
    }

    summary = {
        "stage": evidence["stage"],
        "status": status,
        "project": project_name(cfg),
        "run_id": run_id(cfg),
        "readiness": readiness,
        "is_metric_truth": False,
        "training_executed": inputs.training_status.get("training_executed", False),
        "image_count": stage45_inputs.get("image_count"),
        "selected_image_count": stage45_inputs.get("selected_image_count"),
        "warnings": warnings,
        "evidence_json": paths["evidence_json"].as_posix(),
        "viewer_html": paths["viewer_html"].as_posix(),
    }

    write_json_atomic(paths["evidence_json"], evidence)
    write_json_atomic(paths["viewer_manifest_json"], viewer_manifest)
    write_json_atomic(paths["summary_json"], summary)
    _write_html(paths["viewer_html"], evidence)

    logger.info("Stage 7.7 CAMS-GS evidence complete: %s", paths["evidence_json"])

    print(
        "STAGE_07_7_CAMS_GS_EVIDENCE_OK "
        f"status={status} "
        f"readiness={readiness} "
        f"evidence={paths['evidence_json'].as_posix()} "
        f"viewer={paths['viewer_html'].as_posix()}"
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Package optional CAMS-GS / 3DGS evidence artifacts.")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    run_cams_gs_evidence(cfg, force=args.force, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
