from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from .config_access import bool_value, cfg_get, project_name, run_id, stage45_paths
from .input_selection import select_inputs
from .io_utils import clean_dir_guarded, ensure_dir, file_size, write_json_atomic

LOGGER_NAME = "pipeline.stage_04_5_cams_gs"


class Stage45CamsGsError(RuntimeError):
    """Raised when CAMS-GS preparation fails."""


def _load_config(path: Path) -> Any:
    from pipeline.common.config import load_config

    return load_config(path)


def _tool_status() -> dict[str, str | None]:
    return {
        "ns-train": shutil.which("ns-train"),
        "ns-process-data": shutil.which("ns-process-data"),
        "ns-export": shutil.which("ns-export"),
        "python": shutil.which("python3") or shutil.which("python"),
    }


def _write_dataset_stub(dataset_dir: Path, inputs: Any, *, project: str) -> dict[str, Any]:
    ensure_dir(dataset_dir)
    images_list = dataset_dir / "selected_images.txt"
    images_list.write_text("\n".join(p.as_posix() for p in inputs.selected_images) + ("\n" if inputs.selected_images else ""), encoding="utf-8")

    readme = dataset_dir / "README.md"
    readme.write_text(
        "# CAMS-GS / 3DGS Dataset Stub\n\n"
        "This directory is a preparation stub for later Nerfstudio/Splatfacto integration.\n\n"
        f"Project: {project}\n"
        f"Source images: {inputs.images_dir.as_posix()}\n"
        f"Selected image count: {len(inputs.selected_images)}\n"
        f"Sparse model: {inputs.sparse_model_dir.as_posix()}\n",
        encoding="utf-8",
    )

    return {
        "dataset_dir": dataset_dir.as_posix(),
        "selected_images_txt": images_list.as_posix(),
        "readme": readme.as_posix(),
        "selected_image_count": len(inputs.selected_images),
    }


def _suggested_commands(cfg: Any, dataset_dir: Path) -> dict[str, str]:
    model = str(cfg_get(cfg, "cams_gs.nerfstudio_method", "splatfacto"))
    max_steps = int(cfg_get(cfg, "cams_gs.max_num_iterations", 7000) or 7000)

    return {
        "nerfstudio_train": f"ns-train {model} --data {dataset_dir.as_posix()} --max-num-iterations {max_steps}",
        "note": "This command is not executed unless cams_gs.execute_training=true and dependencies are installed.",
    }


def run_cams_gs_prepare(cfg: Any, *, force: bool = False, log_level: str = "INFO") -> dict[str, Any]:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    logger = logging.getLogger(LOGGER_NAME)

    paths = stage45_paths(cfg)
    clean_dir_guarded(paths["output_dir"], force=force, required_token="cams_gs")
    ensure_dir(paths["output_dir"])
    ensure_dir(paths["nerfstudio_dataset_dir"])

    inputs = select_inputs(cfg)
    tools = _tool_status()
    execute_training = bool_value(cfg_get(cfg, "cams_gs.execute_training", False))

    dataset_stub = _write_dataset_stub(paths["nerfstudio_dataset_dir"], inputs, project=project_name(cfg))
    suggested_commands = _suggested_commands(cfg, paths["nerfstudio_dataset_dir"])

    warnings = list(inputs.warnings)
    training_executed = False

    if inputs.image_count == 0:
        status = "skipped_missing_images"
    elif execute_training and not tools.get("ns-train"):
        status = "skipped_dependency_unavailable"
        warnings.append("ns-train_not_found_training_not_executed")
    elif execute_training:
        status = "training_execution_disabled_in_skeleton"
        warnings.append("skeleton_does_not_run_training_yet")
    else:
        status = "prepared"

    manifest = {
        "stage": "stage_04_5_cams_gs_prepare",
        "status": status,
        "project": project_name(cfg),
        "run_id": run_id(cfg),
        "purpose": "optional_3dgs_visualization_prepare",
        "is_metric_truth": False,
        "metric_truth_note": "CAMS-GS/3DGS is for real-time visualization and evidence rendering only. Stage 8/9 remain metric authority.",
        "paths": {
            "output_dir": paths["output_dir"].as_posix(),
            "nerfstudio_dataset_dir": paths["nerfstudio_dataset_dir"].as_posix(),
            "manifest_json": paths["manifest_json"].as_posix(),
            "training_status_json": paths["training_status_json"].as_posix(),
            "report_json": paths["report_json"].as_posix(),
        },
        "inputs": {
            "images_dir": inputs.images_dir.as_posix(),
            "sparse_model_dir": inputs.sparse_model_dir.as_posix(),
            "colmap_workspace": inputs.colmap_workspace.as_posix(),
            "image_count": inputs.image_count,
            "selected_image_count": len(inputs.selected_images),
            "sparse_model_dir_exists": inputs.sparse_model_dir.exists(),
        },
        "dataset_stub": dataset_stub,
        "external_tools": tools,
        "training": {
            "execute_training_requested": execute_training,
            "training_executed": training_executed,
            "nerfstudio_method": str(cfg_get(cfg, "cams_gs.nerfstudio_method", "splatfacto")),
            "suggested_commands": suggested_commands,
        },
        "warnings": warnings,
        "created_at_unix": time.time(),
    }

    status_payload = {
        "stage": manifest["stage"],
        "status": status,
        "training_executed": training_executed,
        "warnings": warnings,
        "manifest_json": paths["manifest_json"].as_posix(),
        "created_at_unix": manifest["created_at_unix"],
    }

    write_json_atomic(paths["manifest_json"], manifest)
    write_json_atomic(paths["training_status_json"], status_payload)
    write_json_atomic(paths["report_json"], {
        "stage": manifest["stage"],
        "status": status,
        "project": project_name(cfg),
        "run_id": run_id(cfg),
        "image_count": inputs.image_count,
        "selected_image_count": len(inputs.selected_images),
        "training_executed": training_executed,
        "warnings": warnings,
        "manifest_json": paths["manifest_json"].as_posix(),
        "training_status_json": paths["training_status_json"].as_posix(),
    })

    logger.info("Stage 4.5 CAMS-GS prepare complete: %s", paths["manifest_json"])

    print(
        "STAGE_04_5_CAMS_GS_PREPARE_OK "
        f"status={status} "
        f"images={inputs.image_count} "
        f"selected={len(inputs.selected_images)} "
        f"manifest={paths['manifest_json'].as_posix()}"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare optional Stage 4.5 CAMS-GS / 3DGS training artifacts.")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    run_cams_gs_prepare(cfg, force=args.force, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
