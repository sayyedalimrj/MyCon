"""Stage 4 CLI: sparse model refinement with mandatory COLMAP final bundle adjustment."""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from pipeline.common.config import load_config
from pipeline.stage_03_colmap.colmap_cli import ColmapExecutionError, ColmapRunner
from .bundle_adjustment import run_final_bundle_adjustment
from .config_access import Stage4ConfigError, cfg_bool, cfg_get, project_name, resolve_project_path, run_id
from .io_utils import clean_dir, write_json_atomic
from .model_io import SparseModelContractError, copy_sparse_model, resolve_sparse_component_dir, validate_sparse_model
from .pixsfm_optional import run_pixsfm_optional
from .refinement_stats import build_sparse_stats_via_colmap, evaluate_quality_gate


def _configure_logger(cfg: Any, log_level: str) -> logging.Logger:
    logger = logging.getLogger("pipeline.stage_04_refinement")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.addHandler(stream)
    try:
        root = resolve_project_path(cfg, "project.root", "/workspace")
        log_file = root / "runs" / run_id(cfg) / "logs" / "stage_04_refinement.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception:
        pass
    return logger


def _input_sparse_path(cfg: Any, input_override: str | None) -> Path:
    if input_override:
        raw = Path(input_override)
        return raw if raw.is_absolute() else resolve_project_path(cfg, "project.root", "/workspace") / raw
    configured = cfg_get(cfg, "refinement.input_sparse_dir", None)
    if configured:
        path = Path(str(configured))
        return path if path.is_absolute() else resolve_project_path(cfg, "project.root", "/workspace") / path
    # paths.sparse_dir usually points to data/sparse/site01, whose best component is 0.
    return resolve_project_path(cfg, "paths.sparse_dir", f"data/sparse/{project_name(cfg)}")


def _final_refined_component_dir(cfg: Any, output_override: str | None) -> Path:
    if output_override:
        raw = Path(output_override)
        path = raw if raw.is_absolute() else resolve_project_path(cfg, "project.root", "/workspace") / raw
    else:
        configured = cfg_get(cfg, "refinement.output_sparse_dir", None)
        if configured:
            path = Path(str(configured))
            path = path if path.is_absolute() else resolve_project_path(cfg, "project.root", "/workspace") / path
        else:
            path = resolve_project_path(cfg, "paths.sparse_refined_dir", f"data/sparse_refined/{project_name(cfg)}")
    # Accept either a parent dir (data/sparse_refined/site01) or an explicit component dir (.../0).
    return path if path.name == "0" else path / "0"


def _report_path(cfg: Any) -> Path:
    return resolve_project_path(
        cfg,
        "refinement.report_json",
        f"runs/{run_id(cfg)}/reports/refinement_stats.json",
    )


def _history_path(cfg: Any) -> Path:
    return resolve_project_path(
        cfg,
        "refinement.command_history_json",
        f"data/sparse_refined/{project_name(cfg)}/command_history.json",
    )


def _input_image_count_hint(cfg: Any, input_model_dir: Path) -> int:
    # Prefer Stage 3 active manifest length if available; fall back to binary stats later.
    manifest = resolve_project_path(cfg, "paths.manifest_csv", "data/frames/key/site01_manifest.csv")
    if manifest.exists():
        try:
            import csv

            with manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            keep_sparse = [row for row in rows if str(row.get("keep_sparse", "true")).strip().lower() in {"1", "true", "yes", "y"}]
            return max(1, len(keep_sparse) or len(rows))
        except Exception:
            pass
    # Conservative fallback; stats use this for registered_ratio denominator.
    return 1


def _check_output_contract(refined_model_dir: Path) -> None:
    validation = validate_sparse_model(refined_model_dir)
    if not validation["valid_binary_contract"]:
        missing = [name for name, info in validation["files"].items() if not info["nonempty"]]
        raise RuntimeError(f"Stage 4 output contract failed for {refined_model_dir}; missing/empty: {missing}")


def run_refinement(
    cfg: Any,
    force: bool = False,
    log_level: str = "INFO",
    input_sparse: str | None = None,
    output_sparse: str | None = None,
) -> dict[str, Any]:
    """Run Stage 4 final bundle adjustment and write the refined model contract."""
    start = time.time()
    logger = _configure_logger(cfg, log_level)
    name = project_name(cfg)
    rid = run_id(cfg)
    logger.info("Starting Stage 4 sparse refinement for project=%s run_id=%s", name, rid)

    input_component = resolve_sparse_component_dir(_input_sparse_path(cfg, input_sparse))
    final_component = _final_refined_component_dir(cfg, output_sparse)
    report_path = _report_path(cfg)
    history_path = _history_path(cfg)
    work_dir = resolve_project_path(cfg, "refinement.work_dir", f"data/sparse_refined/{name}/_work")

    if final_component.exists() and not force:
        raise FileExistsError(f"Stage 4 refined model already exists. Use --force to overwrite: {final_component}")
    if cfg_bool(cfg, "refinement.validate_binary_before_refinement", True):
        validation = validate_sparse_model(input_component)
        if not validation["valid_binary_contract"]:
            raise SparseModelContractError(f"Invalid Stage 4 input sparse model: {input_component}")

    runner = ColmapRunner(
        executable=str(cfg_get(cfg, "colmap.executable", "colmap")),
        logger=logger,
        qt_qpa_platform=str(cfg_get(cfg, "colmap.qt_qpa_platform", "offscreen")),
    )
    colmap_path = runner.ensure_available()
    logger.info("COLMAP executable: %s", colmap_path)

    clean_dir(work_dir, force=True)
    input_count = _input_image_count_hint(cfg, input_component)
    before_stats = build_sparse_stats_via_colmap(
        runner=runner,
        model_dir=input_component,
        input_image_count=input_count,
        attempt_name="stage_03_sparse_input",
        work_dir=work_dir / "stats",
    )

    ba_output = work_dir / "bundle_adjusted_model"
    pixsfm_result = run_pixsfm_optional(cfg, input_component, work_dir / "pixsfm_model", logger)

    run_final_bundle_adjustment(runner, cfg, input_component, ba_output, logger)
    _check_output_contract(ba_output)
    copy_sparse_model(ba_output, final_component, force=force)
    _check_output_contract(final_component)

    after_stats = build_sparse_stats_via_colmap(
        runner=runner,
        model_dir=final_component,
        input_image_count=input_count,
        attempt_name="stage_04_refined",
        work_dir=work_dir / "stats",
    )
    quality_gate = evaluate_quality_gate(cfg, before_stats, after_stats)
    if not quality_gate["passed"] and cfg_bool(cfg, "refinement.fail_on_quality_gate", True):
        write_json_atomic(history_path, {"commands": runner.history_as_dicts(), "quality_gate": quality_gate})
        raise RuntimeError("Stage 4 quality gate failed: " + "; ".join(quality_gate["failures"]))

    if not cfg_bool(cfg, "refinement.keep_work_dir", True):
        shutil.rmtree(work_dir, ignore_errors=True)

    elapsed = time.time() - start
    report: dict[str, Any] = {
        "stage": "stage_04_sparse_refinement",
        "project": name,
        "run_id": rid,
        "status": "ok",
        "elapsed_sec": elapsed,
        "method": "colmap_bundle_adjustment",
        "input_sparse_model_dir": str(input_component),
        "refined_sparse_model_dir": str(final_component),
        "work_dir": str(work_dir),
        "report_path": str(report_path),
        "command_history_path": str(history_path),
        "before_stats": before_stats,
        "after_stats": after_stats,
        "quality_gate": quality_gate,
        "pixsfm": pixsfm_result.to_dict(),
        "pycolmap_in_process_used": False,
    }
    write_json_atomic(report_path, report)
    write_json_atomic(history_path, {"commands": runner.history_as_dicts(), "quality_gate": quality_gate})
    logger.info(
        "Stage 4 complete in %.3fs; registered=%s points=%s refined=%s",
        elapsed,
        after_stats.get("registered_image_count"),
        after_stats.get("sparse_point_count"),
        final_component,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 4: sparse refinement with final COLMAP bundle adjustment")
    parser.add_argument("--config", required=True, help="Path to YAML config, e.g. configs/site01.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite existing Stage 4 outputs")
    parser.add_argument("--input-sparse", default=None, help="Optional override for input sparse model/component")
    parser.add_argument("--output-sparse", default=None, help="Optional override for output refined model/component")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        cfg = load_config(Path(args.config))
        report = run_refinement(
            cfg,
            force=args.force,
            log_level=args.log_level,
            input_sparse=args.input_sparse,
            output_sparse=args.output_sparse,
        )
    except (Stage4ConfigError, SparseModelContractError, ColmapExecutionError, FileExistsError, RuntimeError) as exc:
        print(f"STAGE_04_REFINEMENT_FAILED: {exc}", file=sys.stderr)
        return 1
    after = report.get("after_stats", {})
    print(
        "STAGE_04_REFINEMENT_OK "
        f"registered={after.get('registered_image_count')} "
        f"points={after.get('sparse_point_count')} "
        f"refined={report.get('refined_sparse_model_dir')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
