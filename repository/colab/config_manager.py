"""Manage the per-run pipeline YAML config for Colab.

We *clone* ``configs/site01.yaml`` (the canonical fully-validated config)
and apply a small, well-defined set of overrides:

- ``project.name`` / ``project.run_id`` / ``project.root``
- ``inputs.video`` / ``inputs.ifc`` / ``inputs.schedule``
- Colab-safe defaults (smaller dense max image size, mock VLM, no real
  CUDA preflight failure, etc.) so the laptop-grade fixture and a
  modest free-Colab T4 both work.
- Free-form user overrides as a YAML snippet from the UI.

We never edit ``configs/site01.yaml`` in-place. The effective config is
written to ``<project_root>/configs/active.yaml`` and is the file passed
to every stage runner.
"""

from __future__ import annotations

import copy
import datetime as _dt
import shutil
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import yaml

from colab.log_capture import LogBuffer

# ---------------------------------------------------------------------------
# Execution profiles
# ---------------------------------------------------------------------------
#
# A *profile* is a named bundle of dotted-key overrides applied on top of the
# canonical ``configs/site01.yaml``. The Colab UI/notebook picks one of:
#
#   * ``colab_safe``  — bounded memory, mock VLM, precomputed DA3, no training.
#                       Always succeeds on a free T4; good for a first run.
#   * ``colab_gpu``   — full pipeline on a single Colab GPU: real dense/cleanup,
#                       real local VLM (when provisioned), DA3 precomputed.
#   * ``production``  — closest to the server contract: no artificial caps and
#                       quality gates left enabled. Intended for an A100/L4
#                       high-RAM runtime or an on-prem GPU box.
#
# Profiles never edit the base config in place; they feed
# ``build_effective_config(profile=...)`` which writes ``active.yaml``.

# Colab-safe overrides applied on top of configs/site01.yaml. Keys are dotted
# paths relative to the YAML root.
COLAB_SAFE_OVERRIDES: dict[str, Any] = {
    # --- Stage 1 (ingest) ---
    "video.normalize_fps": 24,
    "video.crf": 20,
    "video.preset": "fast",
    # --- Stage 2 (keyframes) ---
    "keyframes.max_frames_first_run": 120,
    # --- Stage 5 (dense) — keep memory bounded on a free T4. ---
    "dense.max_image_size": 1024,
    "dense.patch_match_max_image_size": 1024,
    "dense.fail_on_quality_gate": False,
    "dense.require_cuda": False,
    "dense.cuda_preflight": False,
    "dense.gpu_preflight": False,
    "dense.require_visible_gpu": False,
    "dense.adaptive_gpu_profile": True,
    # --- Stage 4.5 (CAMS-GS) — never train on free Colab. ---
    "cams_gs.execute_training": False,
    # --- Stage 6 (DA3 assist) — keep precomputed-only for safety. ---
    "da3.provider": "precomputed",
    "da3.fail_if_required_but_unavailable": False,
    # --- Stage 7 (cleanup) — gentle quality gates. ---
    "cleanup.fail_on_quality_gate": False,
    "cleanup.strict_quality_gate": False,
    # --- Stage 7.5 / 10 (VLM) — mock by default in Colab. ---
    "vlm_qa.provider": "mock",
    "copilot.vlm.provider": "mock",
    "copilot.vlm.fallback_to_mock_when_unavailable": True,
    "copilot.vlm.require_real_vlm": False,
    # --- BIM ---
    "bim.fail_on_low_registration_quality": False,
}

# Full single-GPU Colab run: real dense + cleanup, larger image budget, real
# local VLM wiring is added separately by provision_vlm() when available.
COLAB_GPU_OVERRIDES: dict[str, Any] = {
    "video.normalize_fps": 24,
    "video.crf": 20,
    "video.preset": "fast",
    "keyframes.max_frames_first_run": 240,
    "dense.max_image_size": 1600,
    "dense.patch_match_max_image_size": 1600,
    "dense.fail_on_quality_gate": False,
    "dense.require_cuda": False,
    "dense.cuda_preflight": False,
    "dense.gpu_preflight": False,
    "dense.require_visible_gpu": False,
    "dense.adaptive_gpu_profile": True,
    "cams_gs.execute_training": False,
    "da3.provider": "precomputed",
    "da3.fail_if_required_but_unavailable": False,
    "cleanup.fail_on_quality_gate": False,
    "cleanup.strict_quality_gate": False,
    # VLM kept mock here; provision_vlm() flips these to ollama_local when a
    # real model is successfully pulled.
    "vlm_qa.provider": "mock",
    "copilot.vlm.provider": "mock",
    "copilot.vlm.fallback_to_mock_when_unavailable": True,
    "copilot.vlm.require_real_vlm": False,
    "bim.fail_on_low_registration_quality": False,
}

# Production: leave the canonical server-grade settings mostly untouched. Only
# disable the hard CUDA preflight (so the same config works on heterogeneous
# GPUs) and keep mock-fallback on so an unreachable VLM never crashes a run.
PRODUCTION_OVERRIDES: dict[str, Any] = {
    "dense.require_cuda": False,
    "dense.cuda_preflight": False,
    "copilot.vlm.fallback_to_mock_when_unavailable": True,
}

PROFILES: dict[str, dict[str, Any]] = {
    "colab_safe": COLAB_SAFE_OVERRIDES,
    "colab_gpu": COLAB_GPU_OVERRIDES,
    "production": PRODUCTION_OVERRIDES,
}

DEFAULT_PROFILE = "colab_gpu"


def profile_overrides(profile: str) -> dict[str, Any]:
    """Return the dotted-key override bundle for a named profile."""
    if profile not in PROFILES:
        raise ValueError(
            f"unknown profile {profile!r}; choose one of {sorted(PROFILES)}"
        )
    return dict(PROFILES[profile])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_dotted(target: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: Any = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, MutableMapping):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value


def _deep_merge(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> None:
    for k, v in src.items():
        if (
            k in dst
            and isinstance(dst[k], MutableMapping)
            and isinstance(v, Mapping)
        ):
            _deep_merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def default_run_id(prefix: str = "colab") -> str:
    return f"{_dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{prefix}"


def list_base_configs(repo_root: Path) -> list[str]:
    cfg_dir = Path(repo_root) / "configs"
    if not cfg_dir.is_dir():
        return []
    return sorted(p.name for p in cfg_dir.glob("*.yaml") if p.is_file())


# ---------------------------------------------------------------------------
# Build & write effective config
# ---------------------------------------------------------------------------


def build_effective_config(
    *,
    repo_root: Path,
    base_config_name: str = "site01.yaml",
    project_root: Path,
    run_id: str,
    project_name: str = "colab_run",
    video_path: Optional[Path] = None,
    ifc_path: Optional[Path] = None,
    schedule_path: Optional[Path] = None,
    apply_safe_overrides: bool = True,
    profile: Optional[str] = None,
    override_dict: Optional[Mapping[str, Any]] = None,
    user_overrides_yaml: Optional[str] = None,
    log: Optional[LogBuffer] = None,
) -> dict[str, Any]:
    """Return the effective YAML data dict (not yet written to disk).

    Override precedence (lowest to highest):

    1. ``configs/<base_config_name>`` (canonical config).
    2. Mandatory project-level rewrites (root/run_id/report paths).
    3. Inputs (video/ifc/schedule) when supplied.
    4. The selected execution *profile* (``colab_safe`` / ``colab_gpu`` /
       ``production``). For backwards compatibility, when ``profile`` is None
       and ``apply_safe_overrides`` is True we use ``colab_safe``.
    5. ``override_dict`` — dotted-key overrides supplied programmatically
       (e.g. the real-VLM wiring returned by ``models.provision_vlm``).
    6. ``user_overrides_yaml`` — free-form YAML mapping from the UI.
    """
    base_path = Path(repo_root) / "configs" / base_config_name
    if not base_path.exists():
        raise FileNotFoundError(f"base config not found: {base_path}")
    with base_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"base config is not a YAML mapping: {base_path}")

    # Mandatory project-level overrides — drive the entire path-resolution
    # contract used by pipeline.common.paths.resolve_project_path.
    _set_dotted(data, "project.name", project_name)
    _set_dotted(data, "project.run_id", run_id)
    _set_dotted(data, "project.root", str(Path(project_root).resolve()))

    # Re-anchor any absolute paths in `paths:` that referenced a different
    # run_id-bearing report path. We only touch the report/log files we
    # know the pipeline writes; the data/ paths are already relative to
    # project.root and don't need rewriting.
    _set_dotted(
        data,
        "paths.sparse_report_json",
        f"runs/{run_id}/reports/sparse_stats.json",
    )
    _set_dotted(
        data,
        "paths.da3_report_json",
        f"runs/{run_id}/reports/da3_summary.json",
    )
    _set_dotted(
        data,
        "paths.cleanup_report_json",
        f"runs/{run_id}/reports/cleanup_summary.json",
    )
    _set_dotted(
        data,
        "refinement.report_json",
        f"runs/{run_id}/reports/refinement_stats.json",
    )
    _set_dotted(
        data,
        "dense.report_json",
        f"runs/{run_id}/reports/dense_summary.json",
    )
    _set_dotted(
        data,
        "cleanup.report_json",
        f"runs/{run_id}/reports/cleanup_summary.json",
    )
    _set_dotted(
        data,
        "cleanup.log_path",
        f"runs/{run_id}/logs/stage_07_cleanup.log",
    )
    _set_dotted(
        data,
        "vlm_qa.summary_json",
        f"runs/{run_id}/reports/vlm_qa_summary.json",
    )
    _set_dotted(
        data,
        "bim.registration_report_json",
        f"runs/{run_id}/reports/registration_report.json",
    )
    _set_dotted(
        data,
        "metric_alignment.report_json",
        f"runs/{run_id}/reports/metric_alignment_report.json",
    )
    _set_dotted(
        data,
        "cams_gs.report_json",
        f"runs/{run_id}/reports/cams_gs_prepare_summary.json",
    )
    _set_dotted(
        data,
        "cams_gs_evidence.summary_json",
        f"runs/{run_id}/reports/cams_gs_evidence_summary.json",
    )
    _set_dotted(
        data,
        "copilot.paths.evidence_dir",
        f"runs/{run_id}/copilot/evidence",
    )
    _set_dotted(
        data,
        "copilot.paths.render_dir",
        f"runs/{run_id}/copilot/renders",
    )

    # Inputs — only touch when the user actually supplied a file.
    if video_path is not None:
        _set_dotted(data, "inputs.video", str(video_path))
    if ifc_path is not None:
        _set_dotted(data, "inputs.ifc", str(ifc_path))
    if schedule_path is not None:
        _set_dotted(data, "inputs.schedule", str(schedule_path))

    # Colab-safe / profile overrides.
    selected_profile = profile
    if selected_profile is None and apply_safe_overrides:
        selected_profile = "colab_safe"
    if selected_profile is not None:
        for k, v in profile_overrides(selected_profile).items():
            _set_dotted(data, k, v)
        if log is not None:
            log.append(f"[config] applied profile={selected_profile}")

    # Programmatic dotted-key overrides (e.g. real-VLM wiring).
    if override_dict:
        for k, v in override_dict.items():
            _set_dotted(data, k, v)
        if log is not None:
            log.append(f"[config] applied {len(override_dict)} programmatic override(s)")

    # Free-form user overrides (YAML mapping).
    if user_overrides_yaml and user_overrides_yaml.strip():
        try:
            extra = yaml.safe_load(user_overrides_yaml)
        except yaml.YAMLError as exc:
            raise ValueError(f"user overrides YAML failed to parse: {exc}") from exc
        if extra is None:
            extra = {}
        if not isinstance(extra, dict):
            raise ValueError(
                "user overrides must be a YAML mapping (e.g. 'dense:\\n  max_image_size: 800')"
            )
        _deep_merge(data, extra)

    if log is not None:
        log.append(
            f"[config] base={base_config_name} run_id={run_id} project_root={project_root}"
        )
    return data


def write_effective_config(
    *,
    data: Mapping[str, Any],
    out_path: Path,
    log: Optional[LogBuffer] = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(dict(data), sort_keys=False, default_flow_style=False)
    out_path.write_text(text, encoding="utf-8")
    if log is not None:
        log.append(f"[config] wrote {out_path} ({out_path.stat().st_size} bytes)")
    return out_path


def validate_effective_config(*, config_path: Path, repo_root: Path) -> tuple[bool, str]:
    """Validate the written config using ``pipeline.common.config.load_config``.

    This call requires the pipeline's runtime deps (PyYAML at minimum), but
    does *not* import open3d/cv2 — it only reads the YAML and checks keys.
    """
    import sys

    repo_root = Path(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from pipeline.common.config import load_config  # type: ignore

        load_config(config_path)
        return True, "config_validates"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def make_demo_assets(
    *,
    repo_root: Path,
    project_root: Path,
    log: Optional[LogBuffer] = None,
) -> dict[str, str]:
    """Copy a demo IFC + schedule fixture into the project's BIM folder.

    Lets a user run the BIM/schedule stages without uploading their own
    BIM. We use the fixture set the repo already ships under
    ``examples/end_to_end/inputs``.
    """
    repo_root = Path(repo_root)
    project_root = Path(project_root)
    out: dict[str, str] = {}

    bim_dir = project_root / "data" / "bim" / "design"
    bim_dir.mkdir(parents=True, exist_ok=True)

    src_schedule = repo_root / "examples" / "end_to_end" / "inputs" / "schedule.csv"
    if src_schedule.exists():
        dst = bim_dir / "schedule.csv"
        shutil.copy2(src_schedule, dst)
        out["schedule"] = str(dst)
        if log is not None:
            log.append(f"[demo] copied {src_schedule} -> {dst}")
    return out
