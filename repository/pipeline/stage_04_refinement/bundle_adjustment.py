"""Final COLMAP bundle adjustment for Stage 4."""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

from pipeline.stage_03_colmap.colmap_cli import ColmapRunner

from .config_access import bool_to_colmap, cfg_bool, cfg_float, cfg_int

_OPTION_RE = re.compile(r"(--[A-Za-z0-9_.]+)")


def _record_output_text(record: object) -> str:
    stdout_tail = list(getattr(record, "stdout_tail", []) or [])
    stderr_tail = list(getattr(record, "stderr_tail", []) or [])
    stdout = getattr(record, "stdout", "") or ""
    stderr = getattr(record, "stderr", "") or ""
    parts: list[str] = []
    parts.extend(str(item) for item in stdout_tail)
    parts.extend(str(item) for item in stderr_tail)
    if not parts and stdout:
        parts.append(str(stdout))
    if stderr:
        parts.append(str(stderr))
    return "\n".join(parts)


def collect_supported_bundle_adjuster_options(runner: ColmapRunner, logger: logging.Logger) -> set[str]:
    """Collect supported COLMAP bundle_adjuster options from `-h` output.

    COLMAP option names evolve across releases. Required input/output flags are
    always passed by the caller, but optional BundleAdjustment flags are passed
    only if they are found in the help output. If parsing fails, Stage 4 uses a
    minimal safe command instead of optimistically passing unknown flags.
    """
    record = runner.run(["bundle_adjuster", "-h"], name="bundle_adjuster:help", check=False)
    text = _record_output_text(record)
    options = set(_OPTION_RE.findall(text))
    if not options:
        logger.warning("Could not parse bundle_adjuster help; optional BA flags will be skipped.")
    return options


def _append_if_supported(args: list[str], supported: set[str], option: str, value: str, logger: logging.Logger) -> None:
    if supported and option in supported:
        args.extend([option, value])
    else:
        logger.warning("Skipping unsupported or unverified COLMAP bundle_adjuster option: %s", option)


def build_bundle_adjuster_args(
    cfg: Any,
    input_model_dir: Path,
    output_model_dir: Path,
    supported_options: set[str] | None = None,
    logger: logging.Logger | None = None,
) -> list[str]:
    logger = logger or logging.getLogger(__name__)
    supported = supported_options or set()
    args = [
        "bundle_adjuster",
        "--input_path",
        str(input_model_dir),
        "--output_path",
        str(output_model_dir),
    ]
    _append_if_supported(
        args,
        supported,
        "--BundleAdjustmentCeres.max_num_iterations",
        str(cfg_int(cfg, "refinement.ba_max_num_iterations", 100)),
        logger,
    )
    _append_if_supported(
        args,
        supported,
        "--BundleAdjustmentCeres.max_linear_solver_iterations",
        str(cfg_int(cfg, "refinement.ba_max_linear_solver_iterations", 200)),
        logger,
    )
    _append_if_supported(
        args,
        supported,
        "--BundleAdjustmentCeres.function_tolerance",
        f"{cfg_float(cfg, 'refinement.ba_function_tolerance', 1e-6):.12g}",
        logger,
    )
    _append_if_supported(
        args,
        supported,
        "--BundleAdjustmentCeres.gradient_tolerance",
        f"{cfg_float(cfg, 'refinement.ba_gradient_tolerance', 1e-10):.12g}",
        logger,
    )
    _append_if_supported(
        args,
        supported,
        "--BundleAdjustmentCeres.parameter_tolerance",
        f"{cfg_float(cfg, 'refinement.ba_parameter_tolerance', 1e-8):.12g}",
        logger,
    )
    _append_if_supported(
        args,
        supported,
        "--BundleAdjustment.refine_focal_length",
        bool_to_colmap(cfg_bool(cfg, "refinement.refine_focal_length", True)),
        logger,
    )
    _append_if_supported(
        args,
        supported,
        "--BundleAdjustment.refine_principal_point",
        bool_to_colmap(cfg_bool(cfg, "refinement.refine_principal_point", False)),
        logger,
    )
    _append_if_supported(
        args,
        supported,
        "--BundleAdjustment.refine_extra_params",
        bool_to_colmap(cfg_bool(cfg, "refinement.refine_extra_params", True)),
        logger,
    )
    _append_if_supported(
        args,
        supported,
        "--BundleAdjustmentCeres.num_threads",
        str(cfg_int(cfg, "refinement.ba_num_threads", -1)),
        logger,
    )
    return args


def _prepare_output_dir(path: Path) -> None:
    """Create a clean directory for COLMAP bundle_adjuster output.

    COLMAP 4.x requires --output_path to be an existing directory. Removing the
    old directory without recreating it causes `output_path is not a directory`.
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_final_bundle_adjustment(
    runner: ColmapRunner,
    cfg: Any,
    input_model_dir: Path,
    output_model_dir: Path,
    logger: logging.Logger,
) -> None:
    """Run one or more COLMAP bundle adjustment rounds.

    Multiple rounds are supported for experimentation, but the safe default is a
    single BA pass. Reprojection filtering is intentionally not forced here;
    Stage 5/7 will perform geometry filtering where the data contract is better
    defined.
    """
    supported = collect_supported_bundle_adjuster_options(runner, logger)
    rounds = max(1, cfg_int(cfg, "refinement.ba_rounds", 1))
    current_input = input_model_dir
    output_model_dir.parent.mkdir(parents=True, exist_ok=True)
    for round_idx in range(1, rounds + 1):
        round_output = output_model_dir if round_idx == rounds else output_model_dir.parent / f"{output_model_dir.name}_round_{round_idx}"
        _prepare_output_dir(round_output)
        args = build_bundle_adjuster_args(cfg, current_input, round_output, supported, logger)
        runner.run(args, name=f"bundle_adjuster:round_{round_idx}")
        current_input = round_output
