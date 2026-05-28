"""Build and run COLMAP feature extraction commands."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .colmap_cli import ColmapRunner
from .config_access import Stage3ConfigError, bool_to_colmap, cfg_bool, cfg_get, cfg_int, resolve_project_path


def _mask_path_args(cfg: Any) -> list[str]:
    """Return COLMAP mask_path args when an existing mask directory is configured.

    Stage 3 intentionally does not run YOLO/SAM/semantic models. That would add
    an optional dependency before the sparse baseline is proven. However, COLMAP
    supports masks natively, so we expose a safe hook for precomputed masks.
    Masks must be named exactly like the images and live in one configured
    directory. If masks are required and missing, fail fast; otherwise no-op.
    """
    use_masks = cfg_bool(cfg, "colmap.use_existing_masks", False)
    require_masks = cfg_bool(cfg, "colmap.require_masks", False)
    raw_mask_path = cfg_get(cfg, "colmap.mask_path", cfg_get(cfg, "paths.sparse_mask_dir", None))
    if not use_masks and not require_masks:
        return []
    if raw_mask_path is None:
        if require_masks:
            raise Stage3ConfigError("colmap.require_masks=true but no colmap.mask_path or paths.sparse_mask_dir is configured.")
        return []
    mask_dir = resolve_project_path(cfg, "colmap.mask_path") if cfg_get(cfg, "colmap.mask_path", None) is not None else resolve_project_path(cfg, "paths.sparse_mask_dir")
    if not mask_dir.exists() or not mask_dir.is_dir():
        if require_masks:
            raise Stage3ConfigError(f"Required COLMAP mask directory does not exist: {mask_dir}")
        return []
    if not any(mask_dir.iterdir()):
        if require_masks:
            raise Stage3ConfigError(f"Required COLMAP mask directory is empty: {mask_dir}")
        return []
    return ["--ImageReader.mask_path", str(mask_dir)]


def build_feature_extractor_args(
    cfg: Any,
    database_path: Path,
    image_path: Path,
    feature_type: str,
    model_options: list[str] | None = None,
) -> list[str]:
    feature_upper = feature_type.upper()
    args = [
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--ImageReader.camera_model",
        str(cfg_get(cfg, "colmap.camera_model", "SIMPLE_RADIAL")),
        "--ImageReader.single_camera",
        bool_to_colmap(cfg_bool(cfg, "colmap.single_camera", True)),
        "--FeatureExtraction.type",
        feature_upper,
    ]
    args.extend(_mask_path_args(cfg))
    if feature_upper.startswith("ALIKED"):
        # Keep the default conservative. LightGlue memory grows rapidly with
        # feature count; 2048 matches COLMAP's ALIKED default and is safer for
        # construction laptops/workstations than 8192.
        args += ["--AlikedExtraction.max_num_features", str(cfg_int(cfg, "colmap.aliked_max_num_features", 2048))]
    elif feature_upper == "SIFT":
        args += ["--SiftExtraction.max_num_features", str(cfg_int(cfg, "colmap.sift_max_num_features", 2048))]
    if model_options:
        # Keep model path options after method-specific options; COLMAP accepts both orders.
        args.extend(model_options)
    return args


def extract_features(
    runner: ColmapRunner,
    cfg: Any,
    database_path: Path,
    image_path: Path,
    feature_type: str,
    model_options: list[str] | None = None,
) -> None:
    args = build_feature_extractor_args(cfg, database_path, image_path, feature_type, model_options)
    runner.run(args, name=f"feature_extractor:{feature_type}")
