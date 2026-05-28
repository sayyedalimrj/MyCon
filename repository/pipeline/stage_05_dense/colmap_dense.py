"""COLMAP dense stereo command construction and execution."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pipeline.stage_03_colmap.colmap_cli import ColmapRunner

from .config_access import cfg_bool, cfg_float, cfg_get, cfg_int, resolve_project_path


def _record_text(record: Any) -> str:
    return "\n".join(getattr(record, "stdout_tail", []) or [])


def _supported_options(runner: ColmapRunner, command: str, logger: logging.Logger) -> set[str]:
    try:
        record = runner.run([command, "-h"], name=f"{command}:help", check=False)
    except Exception as exc:  # pragma: no cover - defensive, exercised by runtime only.
        logger.warning("Could not probe COLMAP %s options: %s", command, exc)
        return set()
    supported: set[str] = set()
    for line in _record_text(record).splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            supported.add(stripped.split()[0])
    if not supported:
        logger.warning("Could not parse supported options for COLMAP command: %s", command)
    return supported


def _append_if_supported(args: list[str], supported: set[str], option: str, value: object, logger: logging.Logger) -> None:
    if supported and option in supported:
        args.extend([option, str(value)])
    else:
        logger.warning("Skipping unsupported or unverified COLMAP dense option: %s", option)


def _bool01(value: bool) -> str:
    return "1" if value else "0"


def run_image_undistorter(
    runner: ColmapRunner,
    cfg: Any,
    image_path: Path,
    sparse_model: Path,
    workspace: Path,
    logger: logging.Logger,
) -> None:
    supported = _supported_options(runner, "image_undistorter", logger)
    args = [
        "image_undistorter",
        "--image_path",
        str(image_path),
        "--input_path",
        str(sparse_model),
        "--output_path",
        str(workspace),
        "--output_type",
        str(cfg_get(cfg, "dense.output_type", "COLMAP")),
    ]
    max_image_size = cfg_int(cfg, "dense.max_image_size", 1600)
    if max_image_size > 0:
        _append_if_supported(args, supported, "--max_image_size", max_image_size, logger)
    src_images = cfg_int(cfg, "dense.num_patch_match_src_images", 15)
    if src_images > 0:
        _append_if_supported(args, supported, "--num_patch_match_src_images", src_images, logger)

    # Optional semantic/static masks are intentionally hook-only. The current
    # COLMAP build may not expose a dense-stage mask option, so the argument is
    # passed only when the command help explicitly supports it.
    if cfg_bool(cfg, "dense.use_existing_masks", False):
        mask_dir = resolve_project_path(cfg, "dense.mask_path", "data/masks/site01")
        if not mask_dir.exists():
            if cfg_bool(cfg, "dense.require_masks", False):
                raise FileNotFoundError(f"Dense mask path does not exist: {mask_dir}")
            logger.warning("Dense mask path does not exist; continuing without masks: %s", mask_dir)
        else:
            _append_if_supported(args, supported, "--ImageReader.mask_path", mask_dir, logger)
            _append_if_supported(args, supported, "--mask_path", mask_dir, logger)
    runner.run(args, name="image_undistorter")


def run_patch_match_stereo(runner: ColmapRunner, cfg: Any, workspace: Path, logger: logging.Logger) -> None:
    supported = _supported_options(runner, "patch_match_stereo", logger)
    args = [
        "patch_match_stereo",
        "--workspace_path",
        str(workspace),
        "--workspace_format",
        str(cfg_get(cfg, "dense.workspace_format", "COLMAP")),
    ]
    option_map: list[tuple[str, object]] = [
        ("--PatchMatchStereo.max_image_size", cfg_int(cfg, "dense.patch_match_max_image_size", 1600)),
        ("--PatchMatchStereo.gpu_index", str(cfg_get(cfg, "dense.patch_match_gpu_index", "-1"))),
        ("--PatchMatchStereo.window_radius", cfg_int(cfg, "dense.patch_window_radius", 5)),
        ("--PatchMatchStereo.window_step", cfg_int(cfg, "dense.patch_window_step", 1)),
        ("--PatchMatchStereo.num_samples", cfg_int(cfg, "dense.patch_num_samples", 15)),
        ("--PatchMatchStereo.num_iterations", cfg_int(cfg, "dense.patch_num_iterations", 5)),
        ("--PatchMatchStereo.geom_consistency", _bool01(cfg_bool(cfg, "dense.geom_consistency", True))),
        ("--PatchMatchStereo.geom_consistency_regularizer", cfg_float(cfg, "dense.geom_consistency_regularizer", 0.3)),
        ("--PatchMatchStereo.geom_consistency_max_cost", cfg_float(cfg, "dense.geom_consistency_max_cost", 5.0)),
        ("--PatchMatchStereo.filter", _bool01(cfg_bool(cfg, "dense.patch_filter", True))),
        ("--PatchMatchStereo.filter_min_ncc", cfg_float(cfg, "dense.filter_min_ncc", 0.05)),
        ("--PatchMatchStereo.filter_min_triangulation_angle", cfg_float(cfg, "dense.filter_min_triangulation_angle", 3.0)),
        ("--PatchMatchStereo.filter_min_num_consistent", cfg_int(cfg, "dense.filter_min_num_consistent", 2)),
        ("--PatchMatchStereo.filter_geom_consistency_max_cost", cfg_float(cfg, "dense.filter_geom_consistency_max_cost", 2.0)),
        ("--PatchMatchStereo.cache_size", cfg_int(cfg, "dense.patch_match_cache_size", 32)),
        ("--PatchMatchStereo.num_threads", cfg_int(cfg, "dense.patch_match_num_threads", -1)),
    ]
    for option, value in option_map:
        if isinstance(value, int) and option.endswith("max_image_size") and value <= 0:
            continue
        _append_if_supported(args, supported, option, value, logger)
    runner.run(args, name="patch_match_stereo")


def run_stereo_fusion(runner: ColmapRunner, cfg: Any, workspace: Path, fused_ply: Path, logger: logging.Logger) -> None:
    supported = _supported_options(runner, "stereo_fusion", logger)
    fused_ply.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "stereo_fusion",
        "--workspace_path",
        str(workspace),
        "--workspace_format",
        str(cfg_get(cfg, "dense.workspace_format", "COLMAP")),
        "--input_type",
        str(cfg_get(cfg, "dense.fusion_input_type", "geometric")),
        "--output_path",
        str(fused_ply),
    ]
    option_map: list[tuple[str, object]] = [
        ("--StereoFusion.max_image_size", cfg_int(cfg, "dense.fusion_max_image_size", -1)),
        ("--StereoFusion.min_num_pixels", cfg_int(cfg, "dense.fusion_min_num_pixels", 3)),
        ("--StereoFusion.max_reproj_error", cfg_float(cfg, "dense.fusion_max_reproj_error", 2.0)),
        ("--StereoFusion.max_depth_error", cfg_float(cfg, "dense.fusion_max_depth_error", 0.02)),
    ]
    bounding_box = cfg_get(cfg, "dense.fusion_bounding_box", None)
    if bounding_box:
        if not isinstance(bounding_box, (list, tuple)) or len(bounding_box) != 6:
            raise ValueError("dense.fusion_bounding_box must contain six numeric values: min_x,min_y,min_z,max_x,max_y,max_z")
        option_map.append(("--StereoFusion.bounding_box", ",".join(str(float(v)) for v in bounding_box)))
    for option, value in option_map:
        if option.endswith("max_image_size") and int(value) <= 0:
            continue
        _append_if_supported(args, supported, option, value, logger)
    runner.run(args, name="stereo_fusion")
