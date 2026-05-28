"""Strict YAML configuration loading for the file-contract pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import yaml


class ConfigError(RuntimeError):
    """Raised when the YAML configuration is invalid."""


@dataclass(frozen=True)
class PipelineConfig:
    """Loaded and validated pipeline configuration."""

    path: Path
    data: Mapping[str, Any]

    @property
    def project_root(self) -> Path:
        return Path(str(self.require("project.root")))

    @property
    def project_name(self) -> str:
        return str(self.require("project.name"))

    @property
    def run_id(self) -> str:
        return str(self.require("project.run_id"))

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, Mapping):
            raise ConfigError(f"Config section '{name}' is missing or is not a mapping.")
        return value

    def require(self, dotted_key: str) -> Any:
        current: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise ConfigError(f"Missing required config key: {dotted_key}")
            current = current[part]
        if current is None:
            raise ConfigError(f"Required config key is null: {dotted_key}")
        return current

    def get(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current


_REQUIRED_SECTIONS: tuple[str, ...] = (
    "project", "inputs", "paths", "video", "video_quality", "keyframes",
    "colmap", "refinement", "dense", "da3", "cleanup", "bim", "progress",
)

_REQUIRED_KEYS: tuple[str, ...] = (
    "project.name", "project.run_id", "project.root", "project.random_seed",
    "inputs.video", "inputs.ifc", "inputs.schedule",
    "paths.normalized_video", "paths.metadata_json", "paths.quality_csv",
    "paths.keyframes_dir", "paths.manifest_csv", "paths.contact_sheet",
    "paths.colmap_db", "paths.sparse_dir", "paths.sparse_refined_dir",
    "paths.dense_workspace", "paths.fused_ply", "paths.da3_dir", "paths.clean_dir",
    "paths.bim_aligned_dir", "paths.metrics_dir",
    "video.normalize_fps", "video.sample_fps_for_quality", "video.output_pix_fmt",
    "video.codec", "video.codec_fallback", "video.crf", "video.preset",
    "video.overwrite", "video.skip_reencode_if_compliant", "video.preserve_audio",
    "video.clear_rotation_metadata", "video.force_constant_frame_rate", "video.cfr_option",
    "video.verify_cfr_after_normalization", "video.cfr_tolerance_fps",
    "video.ffmpeg_loglevel",
    "video_quality.min_blur_laplacian", "video_quality.adaptive_blur_enabled",
    "video_quality.adaptive_blur_window", "video_quality.adaptive_blur_multiplier",
    "video_quality.adaptive_blur_floor", "video_quality.adaptive_blur_update_min_ratio",
    "video_quality.adaptive_blur_min_window_samples", "video_quality.max_duplicate_similarity",
    "video_quality.max_exposure_jump", "video_quality.min_motion_score",
    "video_quality.max_motion_score", "video_quality.fast_seek",
    "video_quality.scoring_max_width", "video_quality.scoring_max_height",
    "video_quality.use_histogram_similarity", "video_quality.histogram_bins",
    "video_quality.use_feature_density", "video_quality.feature_detector",
    "video_quality.feature_max_keypoints", "video_quality.target_feature_count",
    "video_quality.min_feature_density_score", "video_quality.reject_low_feature_density",
    "video_quality.rolling_shutter_warning_threshold", "video_quality.jitter_warning_threshold",
    "video_quality.reject_rolling_shutter_warning", "video_quality.reject_jitter_warning",
    "video_quality.quality_weights",
    "keyframes.min_time_gap_sec", "keyframes.max_frames_first_run",
    "keyframes.min_segment_duration_sec", "keyframes.min_segment_frames", "keyframes.max_segment_gap_sec",
    "keyframes.contact_sheet_thumb_width", "keyframes.contact_sheet_max_images", "keyframes.jpeg_quality",
    "keyframes.selection_quality_weight", "keyframes.selection_novelty_weight", "keyframes.selection_feature_weight",
    "keyframes.dense_keep_ratio", "keyframes.fallback_min_keyframes", "keyframes.allow_relaxed_fallback",
    "keyframes.fallback_blur_ratio", "keyframes.fallback_exposure_multiplier", "keyframes.fallback_motion_multiplier",
    "keyframes.random_seek_extraction", "keyframes.reject_stage1_warnings", "keyframes.reject_low_feature_density",
    "keyframes.verify_frame_index_bounds", "keyframes.verify_timestamp_frame_consistency",
    "keyframes.max_timestamp_frame_index_drift_sec", "keyframes.emergency_fallback_if_no_keyframes",
    "colmap.camera_model", "colmap.single_camera", "colmap.feature_type",
    "colmap.matcher_type", "colmap.fallback_feature_type", "colmap.fallback_matcher_type",
    "refinement.run_bundle_adjustment", "refinement.pixsfm_enabled",
    "dense.max_image_size", "dense.geom_consistency", "dense.patch_window_radius", "dense.filter_min_ncc",
    "da3.enabled", "da3.model", "da3.activate_if_dense_coverage_below",
    "cleanup.voxel_size_m", "cleanup.statistical_nb_neighbors", "cleanup.statistical_std_ratio",
    "bim.units", "bim.icp_max_corr_distance_m",
    "progress.coverage_threshold", "progress.deviation_threshold_m",
)

_WEIGHT_KEYS: tuple[str, ...] = (
    "sharpness", "exposure_stability", "motion", "novelty", "feature_density",
)


def load_config(config_path: str | Path, *, root_override: str | Path | None = None) -> PipelineConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, MutableMapping):
        raise ConfigError(f"Config file must contain a YAML mapping: {path}")
    if root_override is not None:
        project = loaded.setdefault("project", {})
        if not isinstance(project, MutableMapping):
            raise ConfigError("Config section 'project' must be a mapping.")
        project["root"] = str(Path(root_override).expanduser().resolve())
    cfg = PipelineConfig(path=path, data=loaded)
    validate_config(cfg)
    return cfg


def validate_config(cfg: PipelineConfig) -> None:
    for section_name in _REQUIRED_SECTIONS:
        cfg.section(section_name)
    for dotted_key in _REQUIRED_KEYS:
        cfg.require(dotted_key)
    weights = cfg.require("video_quality.quality_weights")
    if not isinstance(weights, Mapping):
        raise ConfigError("video_quality.quality_weights must be a mapping.")
    for key in _WEIGHT_KEYS:
        if key not in weights:
            raise ConfigError(f"Missing required config key: video_quality.quality_weights.{key}")
    _positive(cfg, "video.normalize_fps")
    _positive(cfg, "video.sample_fps_for_quality")
    _positive(cfg, "video_quality.min_blur_laplacian")
    _range(cfg, "video_quality.max_duplicate_similarity", 0.0, 1.0)
    _range(cfg, "video_quality.adaptive_blur_update_min_ratio", 0.0, 1.0)
    _positive(cfg, "video.cfr_tolerance_fps")
    _positive(cfg, "video_quality.scoring_max_width")
    _positive(cfg, "video_quality.scoring_max_height")
    _positive(cfg, "video_quality.adaptive_blur_min_window_samples")
    if str(cfg.require("video.cfr_option")).lower() not in {"auto", "fps_mode", "vsync"}:
        raise ConfigError("video.cfr_option must be one of: auto, fps_mode, vsync.")
    _positive(cfg, "keyframes.min_time_gap_sec")
    _positive(cfg, "keyframes.max_frames_first_run")
    _positive(cfg, "keyframes.min_segment_duration_sec")
    _positive(cfg, "keyframes.min_segment_frames")
    _positive(cfg, "keyframes.max_segment_gap_sec")
    _positive(cfg, "keyframes.contact_sheet_thumb_width")
    _positive(cfg, "keyframes.contact_sheet_max_images")
    _range(cfg, "keyframes.jpeg_quality", 1.0, 100.0)
    _range(cfg, "keyframes.selection_quality_weight", 0.0, 1.0)
    _range(cfg, "keyframes.selection_novelty_weight", 0.0, 1.0)
    _range(cfg, "keyframes.selection_feature_weight", 0.0, 1.0)
    _range(cfg, "keyframes.dense_keep_ratio", 0.0, 1.0)
    _positive(cfg, "keyframes.fallback_min_keyframes")
    _range(cfg, "keyframes.fallback_blur_ratio", 0.0, 1.0)
    _range(cfg, "keyframes.fallback_exposure_multiplier", 1.0, 5.0)
    _range(cfg, "keyframes.fallback_motion_multiplier", 1.0, 5.0)
    _positive(cfg, "keyframes.max_timestamp_frame_index_drift_sec")
    if (float(cfg.require("keyframes.selection_quality_weight")) +
            float(cfg.require("keyframes.selection_novelty_weight")) +
            float(cfg.require("keyframes.selection_feature_weight"))) <= 0.0:
        raise ConfigError("At least one keyframe selection weight must be positive.")
    _range(cfg, "video_quality.max_exposure_jump", 0.0, 1.0)
    _range(cfg, "video_quality.min_motion_score", 0.0, 1.0)
    _range(cfg, "video_quality.max_motion_score", 0.0, 1.0)
    if float(cfg.require("video_quality.min_motion_score")) >= float(cfg.require("video_quality.max_motion_score")):
        raise ConfigError("video_quality.min_motion_score must be smaller than max_motion_score.")
    if _looks_like_windows_path(str(cfg.project_root)):
        raise ConfigError("project.root must be a Linux/container path such as /workspace, not a Windows/UNC path.")


def required_config_keys() -> Iterable[str]:
    return _REQUIRED_KEYS


def _positive(cfg: PipelineConfig, dotted_key: str) -> None:
    try:
        value = float(cfg.require(dotted_key))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{dotted_key} must be numeric.") from exc
    if value <= 0:
        raise ConfigError(f"{dotted_key} must be positive.")


def _range(cfg: PipelineConfig, dotted_key: str, low: float, high: float) -> None:
    try:
        value = float(cfg.require(dotted_key))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{dotted_key} must be numeric.") from exc
    if not low <= value <= high:
        raise ConfigError(f"{dotted_key} must be between {low} and {high}.")


def _looks_like_windows_path(value: str) -> bool:
    return value.startswith("\\\\") or (len(value) >= 3 and value[1:3] == ":\\")
