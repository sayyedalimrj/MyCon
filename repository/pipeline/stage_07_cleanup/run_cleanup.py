from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from pipeline.common.config import load_config
except Exception:  # pragma: no cover
    load_config = None  # type: ignore

try:
    from pipeline.common.logging_utils import configure_logging
except Exception:  # pragma: no cover
    configure_logging = None  # type: ignore

from .config_access import bool_value, cfg_get, stage7_paths
from .input_selection import select_input_cloud
from .io_utils import clean_dir_guarded, ensure_dir, write_json_atomic
from .meshing import create_mesh
from .plane_extraction import extract_planes
from .point_cloud_cleanup import clean_point_cloud
from .semantic_context import build_semantic_context


class Stage7CleanupError(RuntimeError):
    """Stage 7 cleanup failed."""


def _configure_logger(cfg: Any, log_level: str) -> logging.Logger:
    if configure_logging is not None:
        try:
            configure_logging(log_level)
        except TypeError:
            pass
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("pipeline.stage_07_cleanup")
    log_path_value = cfg_get(cfg, "cleanup.log_path", None)
    if log_path_value:
        log_path = Path(str(log_path_value))
        if not log_path.is_absolute():
            root = Path(str(cfg_get(cfg, "project.root", ".")))
            log_path = root / log_path
        ensure_dir(log_path.parent)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path for h in logger.handlers):
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            logger.addHandler(handler)
    return logger


def _load_config(path: Path) -> Any:
    if load_config is None:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return load_config(path)


def _quality_gate(cfg: Any, cleanup: dict[str, Any], planes: list[Any], mesh: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    cleaned_count = int(cleanup.get("cleaned_count", 0))
    removed_ratio = float(cleanup.get("removed_ratio", 0.0))
    min_points = int(cfg_get(cfg, "cleanup.quality_min_points", 10_000))
    max_removed = float(cfg_get(cfg, "cleanup.quality_max_removed_ratio", 0.85))
    min_planes = int(cfg_get(cfg, "cleanup.quality_min_planes", 1))
    if cleaned_count < min_points:
        warnings.append(f"cleaned_point_count_below_quality_min:{cleaned_count}<{min_points}")
    if removed_ratio > max_removed:
        failures.append(f"removed_ratio_above_max:{removed_ratio:.3f}>{max_removed:.3f}")
    if len(planes) < min_planes:
        warnings.append(f"plane_count_below_quality_min:{len(planes)}<{min_planes}")
    if cfg_get(cfg, "cleanup.mesh_enabled", True) and mesh.get("status") not in {"ok", "disabled", "skipped_insufficient_points"}:
        warnings.append(f"mesh_status:{mesh.get('status')}")
    strict = bool_value(cfg_get(cfg, "cleanup.strict_quality_gate", False))
    passed = not failures and (not strict or not warnings)
    return {"passed": passed, "failures": failures, "warnings": warnings, "strict": strict}


def run_cleanup(cfg: Any, *, force: bool = False, log_level: str = "INFO") -> dict[str, Any]:
    started = time.time()
    logger = _configure_logger(cfg, log_level)
    paths = stage7_paths(cfg)

    if force:
        clean_dir_guarded(paths["clean_dir"], force=True, required_token="clean", logger=logger)
    ensure_dir(paths["clean_dir"])
    ensure_dir(paths["plane_clouds_dir"])
    ensure_dir(paths["report_json"].parent)

    logger.info("Starting Stage 7 cleanup for project=%s run_id=%s", cfg_get(cfg, "project.name", "unknown"), cfg_get(cfg, "project.run_id", "unknown"))
    selected = select_input_cloud(cfg)
    logger.info("Selected input cloud: %s (%s)", selected.path, selected.source)

    semantic_context = build_semantic_context(cfg, paths)
    pcd, cleanup_result = clean_point_cloud(cfg, selected.path, paths["downsampled_cloud"], paths["cleaned_cloud"], logger, semantic_context=semantic_context)
    plane_records = extract_planes(cfg, pcd, paths["plane_clouds_dir"], paths["planes_json"])
    mesh_result = create_mesh(cfg, pcd, paths["mesh_ply"], plane_records)

    cleanup_dict = asdict(cleanup_result)
    cleanup_dict["downsampled_path"] = cleanup_result.downsampled_path.as_posix()
    cleanup_dict["cleaned_path"] = cleanup_result.cleaned_path.as_posix()
    planes_list = [asdict(p) for p in plane_records]
    mesh_dict = asdict(mesh_result)
    quality_gate = _quality_gate(cfg, cleanup_dict, plane_records, mesh_dict)

    report: dict[str, Any] = {
        "stage": "stage_07_cleanup",
        "status": "ok" if quality_gate["passed"] else "quality_gate_failed",
        "project": {
            "name": cfg_get(cfg, "project.name", "unknown"),
            "run_id": cfg_get(cfg, "project.run_id", "unknown"),
        },
        "input": asdict(selected),
        "cleanup": cleanup_dict,
        "planes": {"count": len(planes_list), "records": planes_list, "planes_json": paths["planes_json"].as_posix()},
        "mesh": mesh_dict,
        "semantic_context": semantic_context,
        "quality_gate": quality_gate,
        "outputs": {
            "downsampled_cloud": paths["downsampled_cloud"].as_posix(),
            "cleaned_cloud": paths["cleaned_cloud"].as_posix(),
            "mesh_ply": paths["mesh_ply"].as_posix() if mesh_dict.get("mesh_path") else None,
            "planes_json": paths["planes_json"].as_posix(),
            "plane_clouds_dir": paths["plane_clouds_dir"].as_posix(),
            "report_json": paths["report_json"].as_posix(),
        },
        "duration_sec": round(time.time() - started, 3),
    }
    write_json_atomic(paths["report_json"], report)

    if not quality_gate["passed"] and bool_value(cfg_get(cfg, "cleanup.fail_on_quality_gate", False)):
        raise Stage7CleanupError("Stage 7 quality gate failed: " + "; ".join(quality_gate["failures"] + quality_gate["warnings"]))

    print(
        "STAGE_07_CLEANUP_OK "
        f"points={cleanup_result.cleaned_count} planes={len(plane_records)} "
        f"mesh_status={mesh_result.status} report={paths['report_json']}"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 7 Open3D cleanup, meshing, and plane extraction.")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config.")
    parser.add_argument("--force", action="store_true", help="Remove existing Stage 7 output directory before running.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    args = parser.parse_args(argv)

    try:
        cfg = _load_config(args.config)
        run_cleanup(cfg, force=args.force, log_level=args.log_level)
        return 0
    except Exception as exc:
        print(f"STAGE_07_CLEANUP_FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
