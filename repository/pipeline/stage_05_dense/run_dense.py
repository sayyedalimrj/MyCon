"""Stage 5 CLI: COLMAP dense stereo with weak-texture defaults."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from pipeline.common.config import load_config
from pipeline.stage_03_colmap.colmap_cli import ColmapExecutionError, ColmapRunner

from .colmap_dense import run_image_undistorter, run_patch_match_stereo, run_stereo_fusion
from .config_access import ConfigOverlay, Stage5ConfigError, cfg_bool, cfg_get, project_name, resolve_project_path, run_id
from .dense_stats import build_dense_stats, evaluate_quality_gate
from .io_utils import clean_dense_workspace, write_json_atomic
from .model_io import DenseContractError, assert_fused_ply, require_image_dir, require_sparse_model
from .gpu_preflight import build_dense_runtime_profile


def _configure_logger(cfg: Any, log_level: str) -> logging.Logger:
    logger = logging.getLogger("pipeline.stage_05_dense")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.addHandler(stream)
    try:
        root = resolve_project_path(cfg, "project.root", "/workspace")
        log_file = root / "runs" / run_id(cfg) / "logs" / "stage_05_dense.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception:
        pass
    return logger


def _input_sparse(cfg: Any, override: str | None) -> Path:
    if override:
        raw = Path(override)
        return raw if raw.is_absolute() else resolve_project_path(cfg, "project.root", "/workspace") / raw
    return resolve_project_path(cfg, "dense.input_sparse_refined_dir", "data/sparse_refined/site01/0")


def _input_images(cfg: Any, override: str | None) -> Path:
    if override:
        raw = Path(override)
        return raw if raw.is_absolute() else resolve_project_path(cfg, "project.root", "/workspace") / raw
    return resolve_project_path(cfg, "dense.input_images_dir", "data/sfm/site01/images")


def _workspace(cfg: Any, override: str | None) -> Path:
    if override:
        raw = Path(override)
        return raw if raw.is_absolute() else resolve_project_path(cfg, "project.root", "/workspace") / raw
    return resolve_project_path(cfg, "dense.workspace_dir", f"data/dense/{project_name(cfg)}")


def _fused_ply(cfg: Any, workspace: Path) -> Path:
    configured = cfg_get(cfg, "dense.fused_ply", None) or cfg_get(cfg, "paths.fused_ply", None)
    if configured:
        raw = Path(str(configured))
        return raw if raw.is_absolute() else resolve_project_path(cfg, "project.root", "/workspace") / raw
    return workspace / "fused.ply"


def _report_path(cfg: Any) -> Path:
    return resolve_project_path(cfg, "dense.report_json", f"runs/{run_id(cfg)}/reports/dense_summary.json")


def _history_path(cfg: Any) -> Path:
    return resolve_project_path(cfg, "dense.command_history_json", f"data/dense/{project_name(cfg)}/command_history.json")


def run_dense(
    cfg: Any,
    force: bool = False,
    log_level: str = "INFO",
    input_sparse: str | None = None,
    input_images: str | None = None,
    workspace_override: str | None = None,
    skip_patch_match: bool = False,
) -> dict[str, Any]:
    start = time.time()
    logger = _configure_logger(cfg, log_level)
    name = project_name(cfg)
    rid = run_id(cfg)
    logger.info("Starting Stage 5 dense stereo for project=%s run_id=%s", name, rid)

    sparse_model = _input_sparse(cfg, input_sparse)
    images_dir = _input_images(cfg, input_images)
    workspace = _workspace(cfg, workspace_override)
    fused_ply = _fused_ply(cfg, workspace)
    report_path = _report_path(cfg)
    history_path = _history_path(cfg)

    require_sparse_model(sparse_model, "Stage 5 input refined")
    images = require_image_dir(images_dir, int(cfg_get(cfg, "dense.min_input_images", 2)))
    if workspace.exists() and not force:
        raise FileExistsError(f"Stage 5 dense workspace already exists. Use --force to overwrite: {workspace}")

    runner = ColmapRunner(
        executable=str(cfg_get(cfg, "colmap.executable", "colmap")),
        logger=logger,
        qt_qpa_platform=str(cfg_get(cfg, "colmap.qt_qpa_platform", "offscreen")),
    )
    colmap_path = runner.ensure_available()
    logger.info("COLMAP executable: %s", colmap_path)

    runtime_profile = build_dense_runtime_profile(runner, cfg, logger, input_image_count=len(images)) if not skip_patch_match else None
    cfg_runtime: Any = ConfigOverlay(cfg, runtime_profile.overrides) if runtime_profile and runtime_profile.overrides else cfg

    clean_dense_workspace(workspace, resolve_project_path(cfg, "project.root", "/workspace"), force=force)
    # Keep fused_ply inside the workspace by default. If user configured an external path, remove stale file on --force.
    if fused_ply.exists() and force:
        fused_ply.unlink()

    run_image_undistorter(runner, cfg_runtime, images_dir, sparse_model, workspace, logger)
    if not skip_patch_match:
        run_patch_match_stereo(runner, cfg_runtime, workspace, logger)
        run_stereo_fusion(runner, cfg_runtime, workspace, fused_ply, logger)
    else:
        logger.warning("Skipping patch_match_stereo and stereo_fusion because --skip-patch-match was supplied.")

    assert_fused_ply(fused_ply)
    stats = build_dense_stats(workspace, fused_ply, input_image_count=len(images))
    quality_gate = evaluate_quality_gate(cfg_runtime, stats)
    if not quality_gate["passed"] and cfg_bool(cfg, "dense.fail_on_quality_gate", True):
        write_json_atomic(history_path, {"commands": runner.history_as_dicts(), "quality_gate": quality_gate})
        raise RuntimeError("Stage 5 quality gate failed: " + "; ".join(quality_gate["failures"]))

    elapsed = time.time() - start
    report = {
        "stage": "stage_05_dense_stereo",
        "project": name,
        "run_id": rid,
        "status": "ok",
        "elapsed_sec": elapsed,
        "input_sparse_model_dir": str(sparse_model),
        "input_images_dir": str(images_dir),
        "workspace_dir": str(workspace),
        "fused_ply": str(fused_ply),
        "report_path": str(report_path),
        "command_history_path": str(history_path),
        "dense_stats": stats,
        "quality_gate": quality_gate,
        "dense_runtime_profile": {
            "cuda_build_detected": runtime_profile.cuda_build_detected if runtime_profile else None,
            "visible_gpus": [gpu.__dict__ for gpu in runtime_profile.visible_gpus] if runtime_profile else [],
            "selected_gpu_index": runtime_profile.selected_gpu_index if runtime_profile else None,
            "overrides": runtime_profile.overrides if runtime_profile else {},
            "notes": runtime_profile.notes if runtime_profile else [],
        },
        "weak_texture_preset": {
            "max_image_size": cfg_get(cfg_runtime, "dense.max_image_size", 1600),
            "patch_match_max_image_size": cfg_get(cfg_runtime, "dense.patch_match_max_image_size", 1600),
            "patch_window_radius": cfg_get(cfg_runtime, "dense.patch_window_radius", 5),
            "filter_min_ncc": cfg_get(cfg_runtime, "dense.filter_min_ncc", 0.05),
            "geom_consistency": cfg_get(cfg_runtime, "dense.geom_consistency", True),
            "patch_match_gpu_index": cfg_get(cfg_runtime, "dense.patch_match_gpu_index", "-1"),
        },
    }
    write_json_atomic(report_path, report)
    write_json_atomic(history_path, {"commands": runner.history_as_dicts(), "quality_gate": quality_gate})
    logger.info(
        "Stage 5 complete in %.3fs; fused_vertices=%s workspace=%s",
        elapsed,
        stats.get("fused_vertex_count"),
        workspace,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 5: COLMAP dense stereo with weak-texture preset")
    parser.add_argument("--config", required=True, help="Path to YAML config, e.g. configs/site01.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dense workspace")
    parser.add_argument("--input-sparse", default=None, help="Optional override for refined sparse model")
    parser.add_argument("--input-images", default=None, help="Optional override for undistorted source image dir")
    parser.add_argument("--workspace", default=None, help="Optional override for dense workspace")
    parser.add_argument("--skip-patch-match", action="store_true", help="Developer-only: skip heavy MVS commands")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        cfg = load_config(Path(args.config))
        report = run_dense(
            cfg,
            force=args.force,
            log_level=args.log_level,
            input_sparse=args.input_sparse,
            input_images=args.input_images,
            workspace_override=args.workspace,
            skip_patch_match=args.skip_patch_match,
        )
    except (Stage5ConfigError, DenseContractError, ColmapExecutionError, FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"STAGE_05_DENSE_FAILED: {exc}", file=sys.stderr)
        return 1
    stats = report.get("dense_stats", {})
    print(
        "STAGE_05_DENSE_OK "
        f"vertices={stats.get('fused_vertex_count')} "
        f"depth_maps={stats.get('depth_map_count')} "
        f"fused={report.get('fused_ply')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
