"""Typed read-only schema views over :class:`PipelineConfig`.

This module is *layered on top of* the existing
:class:`pipeline.common.config.PipelineConfig` and its dotted-key access
helpers. It does **not** replace them. Every existing call site that uses
``cfg_get(cfg, "dotted.key", default)`` continues to work unchanged.

Why a layer instead of a replacement
------------------------------------

The repository contains 14 stage entry points and roughly 530 leaf config
keys spread across 19 top-level YAML sections. Each stage already has its own
``config_access.py`` with a private ``cfg_get/cfg_bool/cfg_int/cfg_float``
helper trio. Replacing that surface with a strict Pydantic-style schema
would (a) be a multi-thousand-line change, (b) introduce a new dependency
(Pydantic is not currently installed), and (c) conflict with the existing
:class:`PipelineConfig` validator which is already exercised by tests.

The right minimum for Phase 1 is a typed *view* layer:

- Stages that need typed access can call ``ProjectSchema.from_config(cfg)``
  and get a ``frozen`` dataclass with validated, primitive-typed fields.
- Stages that don't (yet) need it keep using ``cfg_get`` / ``cfg.require``.
- The schemas double as a machine-readable description of the config keys
  each stage actually consumes — which is exactly what
  :mod:`pipeline.common.registry` needs to populate ``required_config_keys``
  on every :class:`StageDescriptor`.

Schemas
-------

- :class:`ProjectSchema` — ``project.*``
- :class:`InputsSchema` — ``inputs.*``
- :class:`PathsSchema` — ``paths.*``
- :class:`Stage01IngestSchema` — keys Stage 1 reads
- :class:`Stage02KeyframesSchema` — keys Stage 2 reads
- :class:`Stage03ColmapSchema` — keys Stage 3 reads
- :class:`Stage04RefinementSchema` — keys Stage 4 reads
- :class:`Stage05DenseSchema` — keys Stage 5 reads
- :class:`Stage06DA3Schema` — keys Stage 6 reads
- :class:`Stage07CleanupSchema` — keys Stage 7 reads
- :class:`Stage08BimEvalSchema` — keys Stage 8 reads
- :class:`Stage09ProgressSchema` — keys Stage 9 reads
- :class:`Stage10CopilotSchema` — keys Stage 10 reads

Each schema is a frozen dataclass with a class method
``from_config(cfg) -> Schema`` that pulls and validates the keys it owns. Each
schema also exposes a :meth:`required_config_keys` class method returning the
exact dotted keys it reads, which the registry uses for static documentation.

Validation
----------

Schemas are intentionally conservative. Each leaf field has a single primitive
Python type (``int``, ``float``, ``str``, ``bool``, ``Path``). Numeric fields
that have an obvious physical meaning (a tolerance in meters, a fitness score
on [0, 1]) document their valid range in the docstring; the call sites that
read them are responsible for enforcing range invariants — exactly as today.

What this module does NOT do
----------------------------

- It does not validate the entire YAML. ``PipelineConfig.validate_config``
  is the authoritative validator and is still called by ``load_config``.
- It does not change how stages are invoked.
- It does not produce new artifacts. That is :mod:`pipeline.common.provenance`.
- It does not register stages. That is :mod:`pipeline.common.registry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipeline.common.config import ConfigError, PipelineConfig

__all__ = [
    "ConfigSchemaError",
    "ProjectSchema",
    "InputsSchema",
    "PathsSchema",
    "Stage01IngestSchema",
    "Stage02KeyframesSchema",
    "Stage03ColmapSchema",
    "Stage04RefinementSchema",
    "Stage05DenseSchema",
    "Stage06DA3Schema",
    "Stage07CleanupSchema",
    "Stage08BimEvalSchema",
    "Stage09ProgressSchema",
    "Stage10CopilotSchema",
    "ALL_STAGE_SCHEMAS",
]


class ConfigSchemaError(ConfigError):
    """Raised when a typed schema view cannot be built from a config."""


# ---------------------------------------------------------------------------
# Internal coercion helpers.
#
# The existing per-stage `cfg_int / cfg_float / cfg_bool` helpers raise stage-
# specific exceptions. The schema layer is stage-agnostic, so it raises
# `ConfigSchemaError`. Behavior (what passes, what doesn't) matches the
# existing helpers as closely as possible to avoid drift.
# ---------------------------------------------------------------------------

def _require(cfg: PipelineConfig, dotted: str) -> Any:
    try:
        return cfg.require(dotted)
    except ConfigError as exc:
        raise ConfigSchemaError(str(exc)) from exc


def _get(cfg: PipelineConfig, dotted: str, default: Any = None) -> Any:
    return cfg.get(dotted, default)


def _as_int(value: Any, dotted: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigSchemaError(f"Config key {dotted} must be an integer, got {value!r}") from exc


def _as_float(value: Any, dotted: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigSchemaError(f"Config key {dotted} must be a float, got {value!r}") from exc


def _as_bool(value: Any) -> bool:
    """Mirror the per-stage cfg_bool semantics so this module reads identically."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_str(value: Any, dotted: str, *, allow_empty: bool = False) -> str:
    if value is None:
        raise ConfigSchemaError(f"Config key {dotted} must be a non-null string")
    s = str(value)
    if not allow_empty and not s:
        raise ConfigSchemaError(f"Config key {dotted} must be a non-empty string")
    return s


def _as_path(value: Any, dotted: str) -> Path:
    if value is None:
        raise ConfigSchemaError(f"Config key {dotted} must be a path-like string")
    return Path(str(value))


def _as_str_list(value: Any, dotted: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise ConfigSchemaError(f"Config key {dotted} must be a list or comma-separated string, got {value!r}")


# ---------------------------------------------------------------------------
# Project / inputs / paths schemas.
#
# These three are shared across every stage; modeling them once means the
# per-stage schemas can compose them rather than re-validate.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectSchema:
    """Typed view of the ``project`` section.

    Fields
    ------
    name : str
        Short project identifier; used in default paths and report stamps.
    run_id : str
        Stable identifier for this run; reports/logs are written under
        ``runs/<run_id>/``.
    root : Path
        Absolute project root inside the container; relative paths in YAML
        are resolved against this. The base config validator already
        rejects Windows-style roots; this schema just types it as ``Path``.
    random_seed : int
        Project-wide RNG seed. Every stage should derive its per-call seeds
        from this via :mod:`pipeline.common.determinism`.
    """

    name: str
    run_id: str
    root: Path
    random_seed: int

    @classmethod
    def required_config_keys(cls) -> tuple[str, ...]:
        return ("project.name", "project.run_id", "project.root", "project.random_seed")

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "ProjectSchema":
        return cls(
            name=_as_str(_require(cfg, "project.name"), "project.name"),
            run_id=_as_str(_require(cfg, "project.run_id"), "project.run_id"),
            root=_as_path(_require(cfg, "project.root"), "project.root"),
            random_seed=_as_int(_require(cfg, "project.random_seed"), "project.random_seed"),
        )


@dataclass(frozen=True)
class InputsSchema:
    """Typed view of the ``inputs`` section."""

    video: Path
    ifc: Path
    schedule: Path

    @classmethod
    def required_config_keys(cls) -> tuple[str, ...]:
        return ("inputs.video", "inputs.ifc", "inputs.schedule")

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "InputsSchema":
        return cls(
            video=_as_path(_require(cfg, "inputs.video"), "inputs.video"),
            ifc=_as_path(_require(cfg, "inputs.ifc"), "inputs.ifc"),
            schedule=_as_path(_require(cfg, "inputs.schedule"), "inputs.schedule"),
        )


@dataclass(frozen=True)
class PathsSchema:
    """Typed view of the ``paths`` section.

    Only the keys that have a stable required slot in the upstream
    :func:`pipeline.common.config.validate_config` ``_REQUIRED_KEYS`` list
    are typed here. Stages that need additional path-like keys read them
    via :mod:`pipeline.common.paths` (which already returns ``Path`` and
    handles ``project.root`` resolution).
    """

    normalized_video: Path
    metadata_json: Path
    quality_csv: Path
    keyframes_dir: Path
    manifest_csv: Path
    contact_sheet: Path
    colmap_db: Path
    sparse_dir: Path
    sparse_refined_dir: Path
    dense_workspace: Path
    fused_ply: Path
    da3_dir: Path
    clean_dir: Path
    bim_aligned_dir: Path
    metrics_dir: Path

    @classmethod
    def required_config_keys(cls) -> tuple[str, ...]:
        return (
            "paths.normalized_video",
            "paths.metadata_json",
            "paths.quality_csv",
            "paths.keyframes_dir",
            "paths.manifest_csv",
            "paths.contact_sheet",
            "paths.colmap_db",
            "paths.sparse_dir",
            "paths.sparse_refined_dir",
            "paths.dense_workspace",
            "paths.fused_ply",
            "paths.da3_dir",
            "paths.clean_dir",
            "paths.bim_aligned_dir",
            "paths.metrics_dir",
        )

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "PathsSchema":
        return cls(
            normalized_video=_as_path(_require(cfg, "paths.normalized_video"), "paths.normalized_video"),
            metadata_json=_as_path(_require(cfg, "paths.metadata_json"), "paths.metadata_json"),
            quality_csv=_as_path(_require(cfg, "paths.quality_csv"), "paths.quality_csv"),
            keyframes_dir=_as_path(_require(cfg, "paths.keyframes_dir"), "paths.keyframes_dir"),
            manifest_csv=_as_path(_require(cfg, "paths.manifest_csv"), "paths.manifest_csv"),
            contact_sheet=_as_path(_require(cfg, "paths.contact_sheet"), "paths.contact_sheet"),
            colmap_db=_as_path(_require(cfg, "paths.colmap_db"), "paths.colmap_db"),
            sparse_dir=_as_path(_require(cfg, "paths.sparse_dir"), "paths.sparse_dir"),
            sparse_refined_dir=_as_path(_require(cfg, "paths.sparse_refined_dir"), "paths.sparse_refined_dir"),
            dense_workspace=_as_path(_require(cfg, "paths.dense_workspace"), "paths.dense_workspace"),
            fused_ply=_as_path(_require(cfg, "paths.fused_ply"), "paths.fused_ply"),
            da3_dir=_as_path(_require(cfg, "paths.da3_dir"), "paths.da3_dir"),
            clean_dir=_as_path(_require(cfg, "paths.clean_dir"), "paths.clean_dir"),
            bim_aligned_dir=_as_path(_require(cfg, "paths.bim_aligned_dir"), "paths.bim_aligned_dir"),
            metrics_dir=_as_path(_require(cfg, "paths.metrics_dir"), "paths.metrics_dir"),
        )


# ---------------------------------------------------------------------------
# Per-stage schemas.
#
# Each stage schema models *only the keys its run_*.py reads*, plus the shared
# project/inputs/paths blocks. The full YAML may contain many more keys; those
# remain accessible to the stage via cfg_get as they are today. These schemas
# are the documentation-grade summary of "what the stage requires", and they
# back the StageDescriptor.required_config_keys field in the registry.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage01IngestSchema:
    """Typed view of the keys Stage 1 (ingest) reads outside ``project``/``paths``/``inputs``."""

    project: ProjectSchema
    inputs: InputsSchema
    paths: PathsSchema
    normalize_fps: float
    sample_fps_for_quality: float
    crf: int
    preset: str

    @classmethod
    def required_config_keys(cls) -> tuple[str, ...]:
        return (
            *ProjectSchema.required_config_keys(),
            *InputsSchema.required_config_keys(),
            *PathsSchema.required_config_keys(),
            "video.normalize_fps",
            "video.sample_fps_for_quality",
            "video.crf",
            "video.preset",
        )

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "Stage01IngestSchema":
        return cls(
            project=ProjectSchema.from_config(cfg),
            inputs=InputsSchema.from_config(cfg),
            paths=PathsSchema.from_config(cfg),
            normalize_fps=_as_float(_require(cfg, "video.normalize_fps"), "video.normalize_fps"),
            sample_fps_for_quality=_as_float(
                _require(cfg, "video.sample_fps_for_quality"), "video.sample_fps_for_quality"
            ),
            crf=_as_int(_require(cfg, "video.crf"), "video.crf"),
            preset=_as_str(_require(cfg, "video.preset"), "video.preset"),
        )


@dataclass(frozen=True)
class Stage02KeyframesSchema:
    project: ProjectSchema
    paths: PathsSchema
    min_time_gap_sec: float
    max_frames_first_run: int
    selection_quality_weight: float
    selection_novelty_weight: float
    selection_feature_weight: float

    @classmethod
    def required_config_keys(cls) -> tuple[str, ...]:
        return (
            *ProjectSchema.required_config_keys(),
            *PathsSchema.required_config_keys(),
            "keyframes.min_time_gap_sec",
            "keyframes.max_frames_first_run",
            "keyframes.selection_quality_weight",
            "keyframes.selection_novelty_weight",
            "keyframes.selection_feature_weight",
        )

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "Stage02KeyframesSchema":
        return cls(
            project=ProjectSchema.from_config(cfg),
            paths=PathsSchema.from_config(cfg),
            min_time_gap_sec=_as_float(_require(cfg, "keyframes.min_time_gap_sec"), "keyframes.min_time_gap_sec"),
            max_frames_first_run=_as_int(
                _require(cfg, "keyframes.max_frames_first_run"), "keyframes.max_frames_first_run"
            ),
            selection_quality_weight=_as_float(
                _require(cfg, "keyframes.selection_quality_weight"), "keyframes.selection_quality_weight"
            ),
            selection_novelty_weight=_as_float(
                _require(cfg, "keyframes.selection_novelty_weight"), "keyframes.selection_novelty_weight"
            ),
            selection_feature_weight=_as_float(
                _require(cfg, "keyframes.selection_feature_weight"), "keyframes.selection_feature_weight"
            ),
        )


@dataclass(frozen=True)
class Stage03ColmapSchema:
    project: ProjectSchema
    paths: PathsSchema
    feature_type: str
    matcher_type: str
    fallback_feature_type: str
    fallback_matcher_type: str
    enable_fallback: bool
    camera_model: str
    single_camera: bool

    @classmethod
    def required_config_keys(cls) -> tuple[str, ...]:
        return (
            *ProjectSchema.required_config_keys(),
            *PathsSchema.required_config_keys(),
            "colmap.feature_type",
            "colmap.matcher_type",
            "colmap.fallback_feature_type",
            "colmap.fallback_matcher_type",
            "colmap.camera_model",
            "colmap.single_camera",
        )

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "Stage03ColmapSchema":
        return cls(
            project=ProjectSchema.from_config(cfg),
            paths=PathsSchema.from_config(cfg),
            feature_type=_as_str(_require(cfg, "colmap.feature_type"), "colmap.feature_type"),
            matcher_type=_as_str(_require(cfg, "colmap.matcher_type"), "colmap.matcher_type"),
            fallback_feature_type=_as_str(
                _require(cfg, "colmap.fallback_feature_type"), "colmap.fallback_feature_type"
            ),
            fallback_matcher_type=_as_str(
                _require(cfg, "colmap.fallback_matcher_type"), "colmap.fallback_matcher_type"
            ),
            enable_fallback=_as_bool(_get(cfg, "colmap.enable_fallback", True)),
            camera_model=_as_str(_require(cfg, "colmap.camera_model"), "colmap.camera_model"),
            single_camera=_as_bool(_require(cfg, "colmap.single_camera")),
        )


@dataclass(frozen=True)
class Stage04RefinementSchema:
    project: ProjectSchema
    paths: PathsSchema
    run_bundle_adjustment: bool
    pixsfm_enabled: bool
    ba_max_num_iterations: int
    ba_rounds: int

    @classmethod
    def required_config_keys(cls) -> tuple[str, ...]:
        return (
            *ProjectSchema.required_config_keys(),
            *PathsSchema.required_config_keys(),
            "refinement.run_bundle_adjustment",
            "refinement.pixsfm_enabled",
        )

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "Stage04RefinementSchema":
        return cls(
            project=ProjectSchema.from_config(cfg),
            paths=PathsSchema.from_config(cfg),
            run_bundle_adjustment=_as_bool(_require(cfg, "refinement.run_bundle_adjustment")),
            pixsfm_enabled=_as_bool(_require(cfg, "refinement.pixsfm_enabled")),
            ba_max_num_iterations=_as_int(_get(cfg, "refinement.ba_max_num_iterations", 100), "refinement.ba_max_num_iterations"),
            ba_rounds=_as_int(_get(cfg, "refinement.ba_rounds", 1), "refinement.ba_rounds"),
        )


@dataclass(frozen=True)
class Stage05DenseSchema:
    project: ProjectSchema
    paths: PathsSchema
    max_image_size: int
    geom_consistency: bool
    patch_window_radius: int
    filter_min_ncc: float

    @classmethod
    def required_config_keys(cls) -> tuple[str, ...]:
        return (
            *ProjectSchema.required_config_keys(),
            *PathsSchema.required_config_keys(),
            "dense.max_image_size",
            "dense.geom_consistency",
            "dense.patch_window_radius",
            "dense.filter_min_ncc",
        )

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "Stage05DenseSchema":
        return cls(
            project=ProjectSchema.from_config(cfg),
            paths=PathsSchema.from_config(cfg),
            max_image_size=_as_int(_require(cfg, "dense.max_image_size"), "dense.max_image_size"),
            geom_consistency=_as_bool(_require(cfg, "dense.geom_consistency")),
            patch_window_radius=_as_int(_require(cfg, "dense.patch_window_radius"), "dense.patch_window_radius"),
            filter_min_ncc=_as_float(_require(cfg, "dense.filter_min_ncc"), "dense.filter_min_ncc"),
        )


@dataclass(frozen=True)
class Stage06DA3Schema:
    project: ProjectSchema
    paths: PathsSchema
    enabled: str  # "auto" | "true" | "false" — string because YAML allows the literal "auto"
    model: str
    activate_if_dense_coverage_below: float
    provider: str
    fail_if_required_but_unavailable: bool

    @classmethod
    def required_config_keys(cls) -> tuple[str, ...]:
        return (
            *ProjectSchema.required_config_keys(),
            *PathsSchema.required_config_keys(),
            "da3.enabled",
            "da3.model",
            "da3.activate_if_dense_coverage_below",
        )

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "Stage06DA3Schema":
        return cls(
            project=ProjectSchema.from_config(cfg),
            paths=PathsSchema.from_config(cfg),
            enabled=_as_str(_require(cfg, "da3.enabled"), "da3.enabled"),
            model=_as_str(_require(cfg, "da3.model"), "da3.model"),
            activate_if_dense_coverage_below=_as_float(
                _require(cfg, "da3.activate_if_dense_coverage_below"),
                "da3.activate_if_dense_coverage_below",
            ),
            provider=_as_str(_get(cfg, "da3.provider", "precomputed"), "da3.provider"),
            fail_if_required_but_unavailable=_as_bool(
                _get(cfg, "da3.fail_if_required_but_unavailable", False)
            ),
        )


@dataclass(frozen=True)
class Stage07CleanupSchema:
    project: ProjectSchema
    paths: PathsSchema
    voxel_size_m: float
    statistical_nb_neighbors: int
    statistical_std_ratio: float

    @classmethod
    def required_config_keys(cls) -> tuple[str, ...]:
        return (
            *ProjectSchema.required_config_keys(),
            *PathsSchema.required_config_keys(),
            "cleanup.voxel_size_m",
            "cleanup.statistical_nb_neighbors",
            "cleanup.statistical_std_ratio",
        )

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "Stage07CleanupSchema":
        return cls(
            project=ProjectSchema.from_config(cfg),
            paths=PathsSchema.from_config(cfg),
            voxel_size_m=_as_float(_require(cfg, "cleanup.voxel_size_m"), "cleanup.voxel_size_m"),
            statistical_nb_neighbors=_as_int(
                _require(cfg, "cleanup.statistical_nb_neighbors"), "cleanup.statistical_nb_neighbors"
            ),
            statistical_std_ratio=_as_float(
                _require(cfg, "cleanup.statistical_std_ratio"), "cleanup.statistical_std_ratio"
            ),
        )


@dataclass(frozen=True)
class Stage08BimEvalSchema:
    project: ProjectSchema
    paths: PathsSchema
    units: str
    icp_max_corr_distance_m: float
    icp_robust_loss: str
    icp_robust_loss_k_m: float
    initial_scale_strategy: str
    coarse_fpfh_enabled: bool

    @classmethod
    def required_config_keys(cls) -> tuple[str, ...]:
        return (
            *ProjectSchema.required_config_keys(),
            *PathsSchema.required_config_keys(),
            "bim.units",
            "bim.icp_max_corr_distance_m",
        )

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "Stage08BimEvalSchema":
        return cls(
            project=ProjectSchema.from_config(cfg),
            paths=PathsSchema.from_config(cfg),
            units=_as_str(_require(cfg, "bim.units"), "bim.units"),
            icp_max_corr_distance_m=_as_float(
                _require(cfg, "bim.icp_max_corr_distance_m"), "bim.icp_max_corr_distance_m"
            ),
            icp_robust_loss=_as_str(_get(cfg, "bim.icp_robust_loss", "none"), "bim.icp_robust_loss"),
            icp_robust_loss_k_m=_as_float(
                _get(cfg, "bim.icp_robust_loss_k_m", 0.05), "bim.icp_robust_loss_k_m"
            ),
            initial_scale_strategy=_as_str(
                _get(cfg, "bim.initial_scale_strategy", "fixed_1"), "bim.initial_scale_strategy"
            ),
            coarse_fpfh_enabled=_as_bool(_get(cfg, "bim.coarse_fpfh_enabled", True)),
        )


@dataclass(frozen=True)
class Stage09ProgressSchema:
    project: ProjectSchema
    paths: PathsSchema
    coverage_threshold: float
    deviation_threshold_m: float
    bidirectional_metrics_enabled: bool
    bootstrap_iterations: int

    @classmethod
    def required_config_keys(cls) -> tuple[str, ...]:
        return (
            *ProjectSchema.required_config_keys(),
            *PathsSchema.required_config_keys(),
            "progress.coverage_threshold",
            "progress.deviation_threshold_m",
        )

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "Stage09ProgressSchema":
        return cls(
            project=ProjectSchema.from_config(cfg),
            paths=PathsSchema.from_config(cfg),
            coverage_threshold=_as_float(
                _require(cfg, "progress.coverage_threshold"), "progress.coverage_threshold"
            ),
            deviation_threshold_m=_as_float(
                _require(cfg, "progress.deviation_threshold_m"), "progress.deviation_threshold_m"
            ),
            bidirectional_metrics_enabled=_as_bool(
                _get(cfg, "progress.bidirectional_metrics_enabled", True)
            ),
            bootstrap_iterations=_as_int(
                _get(cfg, "progress.bootstrap_iterations", 1000), "progress.bootstrap_iterations"
            ),
        )


@dataclass(frozen=True)
class Stage10CopilotSchema:
    project: ProjectSchema
    paths: PathsSchema
    vlm_provider: str
    vlm_endpoint: str
    vlm_model: str
    vlm_local_only: bool
    vlm_fallback_to_mock_when_unavailable: bool

    @classmethod
    def required_config_keys(cls) -> tuple[str, ...]:
        # Stage 10 only requires project / paths from the strict required-keys
        # set. Everything else has YAML-level defaults via cfg_get.
        return (
            *ProjectSchema.required_config_keys(),
            *PathsSchema.required_config_keys(),
        )

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "Stage10CopilotSchema":
        return cls(
            project=ProjectSchema.from_config(cfg),
            paths=PathsSchema.from_config(cfg),
            vlm_provider=_as_str(_get(cfg, "copilot.vlm.provider", "mock"), "copilot.vlm.provider"),
            vlm_endpoint=_as_str(
                _get(cfg, "copilot.vlm.endpoint", ""),
                "copilot.vlm.endpoint",
                allow_empty=True,
            ),
            vlm_model=_as_str(_get(cfg, "copilot.vlm.model", "mock"), "copilot.vlm.model"),
            vlm_local_only=_as_bool(_get(cfg, "copilot.vlm.local_only", True)),
            vlm_fallback_to_mock_when_unavailable=_as_bool(
                _get(cfg, "copilot.vlm.fallback_to_mock_when_unavailable", True)
            ),
        )


# Public ordered tuple consumed by the registry to populate stage descriptors.
# Order matches the canonical pipeline execution order documented in
# scripts/run_pipeline_plan.py.
ALL_STAGE_SCHEMAS: tuple[type, ...] = (
    Stage01IngestSchema,
    Stage02KeyframesSchema,
    Stage03ColmapSchema,
    Stage04RefinementSchema,
    Stage05DenseSchema,
    Stage06DA3Schema,
    Stage07CleanupSchema,
    Stage08BimEvalSchema,
    Stage09ProgressSchema,
    Stage10CopilotSchema,
)
