"""Stage 3 CLI: COLMAP sparse SfM from Stage 2 keyframes."""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from pipeline.common.config import load_config

from .build_database import make_attempt_workspace
from .colmap_cli import ColmapExecutionError, ColmapRunner
from .config_access import Stage3ConfigError, cfg_bool, cfg_get, project_name, resolve_project_path, run_id
from .extract_features import extract_features
from .io_utils import write_json_atomic
from .match_features import match_features
from .model_cache import feature_model_options, matcher_model_options
from .prepare_images import PreparedSparseInputs, prepare_sparse_inputs
from .reconstruct_sparse import find_best_sparse_model, promote_sparse_model, run_mapper
from .sparse_stats import build_sparse_stats


def _configure_logger(cfg: Any, log_level: str) -> logging.Logger:
    logger = logging.getLogger("pipeline.stage_03_colmap")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.addHandler(stream)
    try:
        root = resolve_project_path(cfg, "project.root", "/workspace")
        log_file = root / "runs" / run_id(cfg) / "logs" / "stage_03_colmap.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception:
        # Logging must never prevent a smoke test from running.
        pass
    return logger


def _attempts(cfg: Any, skip_fallback: bool) -> list[tuple[str, str, bool]]:
    main = (
        str(cfg_get(cfg, "colmap.feature_type", "ALIKED_N16ROT")),
        str(cfg_get(cfg, "colmap.matcher_type", "ALIKED_LIGHTGLUE")),
        False,
    )
    attempts = [main]
    if cfg_bool(cfg, "colmap.enable_fallback", True) and not skip_fallback:
        fallback = (
            str(cfg_get(cfg, "colmap.fallback_feature_type", "SIFT")),
            str(cfg_get(cfg, "colmap.fallback_matcher_type", "SIFT_LIGHTGLUE")),
            True,
        )
        if fallback[:2] != main[:2]:
            attempts.append(fallback)
    if cfg_bool(cfg, "colmap.allow_sift_bruteforce_emergency", False) and not skip_fallback:
        attempts.append(("SIFT", str(cfg_get(cfg, "colmap.emergency_matcher_type", "SIFT_BRUTEFORCE")), True))
    return attempts


def _check_final_outputs(database_path: Path, model_dir: Path) -> None:
    missing = []
    if not database_path.exists() or database_path.stat().st_size <= 0:
        missing.append(str(database_path))
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        path = model_dir / name
        if not path.exists() or path.stat().st_size <= 0:
            missing.append(str(path))
    if missing:
        raise RuntimeError("Stage 3 output contract failed. Missing/empty outputs:\n" + "\n".join(missing))


def _copy_database_to_final(source_db: Path, final_db: Path, force: bool) -> None:
    if final_db.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing COLMAP database without --force: {final_db}")
        final_db.unlink()
    final_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_db, final_db)


def run_sparse(cfg: Any, force: bool = False, log_level: str = "INFO", skip_fallback: bool = False) -> dict[str, Any]:
    start = time.time()
    logger = _configure_logger(cfg, log_level)
    name = project_name(cfg)
    rid = run_id(cfg)
    logger.info("Starting Stage 3 sparse SfM for project=%s run_id=%s", name, rid)
    prepared: PreparedSparseInputs = prepare_sparse_inputs(cfg, force=force, logger=logger)
    final_db = resolve_project_path(cfg, "paths.colmap_db", f"data/sfm/{name}/database.db")
    final_sparse_dir = resolve_project_path(cfg, "paths.sparse_dir", f"data/sparse/{name}")
    report_path = resolve_project_path(cfg, "paths.sparse_report_json", f"runs/{rid}/reports/sparse_stats.json")
    history_path = prepared.sfm_dir / "command_history.json"

    if final_db.exists() and not force:
        raise FileExistsError(f"Stage 3 final database already exists. Use --force to overwrite: {final_db}")
    if (final_sparse_dir / "0").exists() and not force:
        raise FileExistsError(f"Stage 3 final sparse model already exists. Use --force to overwrite: {final_sparse_dir / '0'}")

    runner = ColmapRunner(
        executable=str(cfg_get(cfg, "colmap.executable", "colmap")),
        logger=logger,
        qt_qpa_platform=str(cfg_get(cfg, "colmap.qt_qpa_platform", "offscreen")),
    )
    colmap_path = runner.ensure_available()
    logger.info("COLMAP executable: %s", colmap_path)

    errors: list[dict[str, object]] = []
    selected_report: dict[str, Any] | None = None
    selected_attempt = None
    selected_model_dir: Path | None = None

    for index, (feature_type, matcher_type, is_fallback) in enumerate(_attempts(cfg, skip_fallback), start=1):
        attempt = make_attempt_workspace(prepared.sfm_dir, index, feature_type, matcher_type, force=force)
        logger.info("Stage 3 attempt %s: feature=%s matcher=%s", attempt.name, feature_type, matcher_type)
        try:
            feature_options = feature_model_options(cfg, feature_type, logger)
            matcher_options = matcher_model_options(cfg, matcher_type, logger)
            extract_features(runner, cfg, attempt.database_path, prepared.stage_images_dir, feature_type, feature_options)
            match_features(runner, cfg, attempt.database_path, matcher_type, matcher_options)
            run_mapper(runner, cfg, attempt.database_path, prepared.stage_images_dir, attempt.sparse_attempt_dir)
            best = find_best_sparse_model(attempt.sparse_attempt_dir)
            final_model_dir = promote_sparse_model(best, final_sparse_dir, force=force, logger=logger)
            _copy_database_to_final(attempt.database_path, final_db, force=force)
            selected_model_dir = final_model_dir
            selected_attempt = attempt
            selected_report = build_sparse_stats(
                runner=runner,
                model_dir=final_model_dir,
                input_image_count=len(prepared.rows),
                attempt_name=attempt.name,
                fallback_used=is_fallback,
            )
            break
        except Exception as exc:  # noqa: BLE001 - preserve attempt failure and continue to configured fallback
            logger.exception("Stage 3 attempt failed: %s", attempt.name)
            errors.append(
                {
                    "attempt": attempt.name,
                    "feature_type": feature_type,
                    "matcher_type": matcher_type,
                    "fallback": is_fallback,
                    "error": str(exc),
                }
            )
            if index == len(_attempts(cfg, skip_fallback)):
                break

    if selected_report is None or selected_attempt is None or selected_model_dir is None:
        write_json_atomic(history_path, {"commands": runner.history_as_dicts(), "errors": errors})
        raise RuntimeError("All Stage 3 COLMAP attempts failed. See stage_03_colmap.log and command_history.json.")

    _check_final_outputs(final_db, selected_model_dir)
    elapsed = time.time() - start
    report = {
        "stage": "stage_03_colmap_sparse_sfm",
        "project": name,
        "run_id": rid,
        "elapsed_sec": elapsed,
        "status": "ok",
        "selected_attempt": selected_attempt.name,
        "feature_type": selected_attempt.feature_type,
        "matcher_type": selected_attempt.matcher_type,
        "fallback_used": bool(selected_report.get("fallback_used", False)),
        "input_manifest": str(prepared.manifest_csv),
        "active_manifest": str(prepared.active_manifest_csv),
        "stage_images_dir": str(prepared.stage_images_dir),
        "image_list_txt": str(prepared.image_list_txt),
        "database_path": str(final_db),
        "sparse_model_dir": str(selected_model_dir),
        "errors_before_success": errors,
        "sparse_stats": selected_report,
        "command_history_path": str(history_path),
    }
    write_json_atomic(report_path, report)
    write_json_atomic(history_path, {"commands": runner.history_as_dicts(), "errors": errors})
    logger.info(
        "Stage 3 complete in %.3fs; registered=%s/%s points=%s",
        elapsed,
        selected_report.get("registered_image_count"),
        len(prepared.rows),
        selected_report.get("sparse_point_count"),
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3: COLMAP sparse SfM from selected keyframes")
    parser.add_argument("--config", required=True, help="Path to YAML config, e.g. configs/site01.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite existing Stage 3 outputs")
    parser.add_argument("--skip-fallback", action="store_true", help="Disable fallback attempts for this run")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        cfg = load_config(Path(args.config))
        report = run_sparse(cfg, force=args.force, log_level=args.log_level, skip_fallback=args.skip_fallback)
    except (Stage3ConfigError, ColmapExecutionError, FileExistsError, RuntimeError) as exc:
        print(f"STAGE_03_COLMAP_FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "STAGE_03_COLMAP_OK "
        f"registered={report['sparse_stats'].get('registered_image_count')} "
        f"points={report['sparse_stats'].get('sparse_point_count')} "
        f"database={report['database_path']} "
        f"sparse={report['sparse_model_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
