from __future__ import annotations

import argparse
import logging
import time
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

from .colmap_model import export_sparse_text_model, read_colmap_text_model
from .config_access import bool_value, cfg_get, stage6_paths
from .dense_assessment import assess_dense_coverage
from .depth_alignment import align_depth_maps, write_alignment_manifest
from .depth_fusion import fuse_aligned_depths
from .depth_provider import configured_extensions, find_depth_maps, run_external_depth_provider, write_depth_manifest
from .io_utils import clean_dir_guarded, ensure_dir, write_json_atomic


class Stage6DA3Error(RuntimeError):
    """Stage 6 DA3 assistance failed."""


def _logger(log_level: str) -> logging.Logger:
    if configure_logging is not None:
        try:
            configure_logging(log_level)
        except TypeError:
            pass
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    return logging.getLogger("pipeline.stage_06_da3_assist")


def _load_config(path: Path) -> Any:
    if load_config is None:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return load_config(path)


def _provider_available(cfg: Any, depth_records: list[Any], provider_result: dict[str, Any]) -> bool:
    if depth_records:
        return True
    provider = str(cfg_get(cfg, "da3.provider", "precomputed")).strip().lower()
    if provider == "disabled":
        return False
    if provider == "external_command" and provider_result.get("status") == "ok":
        return True
    return False


def run_da3_assist(cfg: Any, *, force: bool = False, log_level: str = "INFO") -> dict[str, Any]:
    started = time.time()
    logger = _logger(log_level)
    paths = stage6_paths(cfg)

    da3_dir = paths["da3_dir"]
    if force:
        clean_dir_guarded(da3_dir, force=True, required_token="da3", logger=logger)
    ensure_dir(da3_dir)
    ensure_dir(paths["depth_output_dir"])
    ensure_dir(paths["aligned_depth_dir"])
    ensure_dir(paths["report_json"].parent)

    logger.info("Starting Stage 6 DA3 assistance for project=%s run_id=%s", cfg_get(cfg, "project.name", "unknown"), cfg_get(cfg, "project.run_id", "unknown"))

    assessment = assess_dense_coverage(cfg, paths["dense_summary_json"])
    decision = {
        "status": assessment.status,
        "should_activate": assessment.should_activate,
        "reasons": assessment.reasons,
        "dense_stats": assessment.dense_stats,
    }
    write_json_atomic(paths["decision_json"], decision)

    report: dict[str, Any] = {
        "stage": "stage_06_da3_assist",
        "status": assessment.status,
        "decision": decision,
        "outputs": {
            "decision_json": paths["decision_json"].as_posix(),
            "depth_manifest_csv": paths["depth_manifest_csv"].as_posix(),
            "alignment_manifest_csv": paths["alignment_manifest_csv"].as_posix(),
            "fusion_plan_json": paths["fusion_plan_json"].as_posix(),
            "assisted_ply": paths["assisted_ply"].as_posix(),
        },
        "provider": {},
        "alignment": {},
        "fusion": {},
        "duration_sec": None,
    }

    if not assessment.should_activate:
        write_depth_manifest(paths["depth_manifest_csv"], [])
        write_alignment_manifest(paths["alignment_manifest_csv"], [])
        write_json_atomic(paths["fusion_plan_json"], {"status": "skipped", "reason": assessment.status})
        report["status"] = "skipped_dense_sufficient"
        report["duration_sec"] = round(time.time() - started, 3)
        write_json_atomic(paths["report_json"], report)
        print(f"STAGE_06_DA3_OK status=skipped_dense_sufficient report={paths['report_json']}")
        return report

    provider = str(cfg_get(cfg, "da3.provider", "precomputed")).strip().lower()
    provider_result: dict[str, Any] = {"status": "not_run", "provider": provider}
    if provider == "external_command":
        logger.info("Running external DA3 depth provider")
        provider_result = run_external_depth_provider(cfg, paths["image_dir"], paths["depth_output_dir"], paths["depth_manifest_csv"])
        if provider_result.get("status") != "ok":
            logger.warning("External depth provider did not complete successfully: %s", provider_result.get("status"))

    max_images = int(cfg_get(cfg, "da3.max_images", 0))
    records = find_depth_maps(paths["image_dir"], paths["depth_input_dir"], extensions=configured_extensions(cfg), max_images=max_images or None)
    write_depth_manifest(paths["depth_manifest_csv"], records)
    report["provider"] = {
        "provider": provider,
        "provider_result": provider_result,
        "depth_record_count": len(records),
        "depth_input_dir": paths["depth_input_dir"].as_posix(),
    }

    if not _provider_available(cfg, records, provider_result):
        fail = bool_value(cfg_get(cfg, "da3.fail_if_required_but_unavailable", False))
        report["status"] = "required_but_provider_unavailable"
        report["duration_sec"] = round(time.time() - started, 3)
        write_alignment_manifest(paths["alignment_manifest_csv"], [])
        write_json_atomic(paths["fusion_plan_json"], {"status": "not_run", "reason": "provider_unavailable"})
        write_json_atomic(paths["report_json"], report)
        message = (
            "DA3 assistance is required by dense coverage thresholds, but no depth maps/provider are available. "
            "Provide precomputed DA3 depth maps or configure da3.external_command."
        )
        if fail:
            raise Stage6DA3Error(message)
        print(f"STAGE_06_DA3_OK status=required_but_provider_unavailable report={paths['report_json']}")
        return report

    logger.info("Preparing sparse model text for DA3 depth alignment")
    text_dir = export_sparse_text_model(cfg, paths["sparse_refined_dir"], paths["sparse_text_dir"])
    model = read_colmap_text_model(text_dir)

    logger.info("Aligning %d depth maps to COLMAP sparse frame", len(records))
    alignment_results = align_depth_maps(cfg, model, records, paths["aligned_depth_dir"])
    write_alignment_manifest(paths["alignment_manifest_csv"], alignment_results)
    ok_count = sum(1 for r in alignment_results if r.status == "ok")
    warning_count = sum(1 for r in alignment_results if r.status == "warning")
    failed_count = sum(1 for r in alignment_results if r.status == "failed")
    report["alignment"] = {"ok": ok_count, "warning": warning_count, "failed": failed_count, "total": len(alignment_results)}

    if bool_value(cfg_get(cfg, "da3.fuse_aligned_depth", True)):
        fusion = fuse_aligned_depths(cfg, model, alignment_results, paths["image_dir"], paths["assisted_ply"], paths["fusion_plan_json"])
    else:
        fusion = {"status": "disabled", "reason": "da3.fuse_aligned_depth is false"}
        write_json_atomic(paths["fusion_plan_json"], fusion)
    report["fusion"] = fusion

    min_ok = int(cfg_get(cfg, "da3.quality_min_aligned_depth_maps", 1))
    if ok_count < min_ok:
        report["status"] = "completed_with_alignment_warnings"
    else:
        report["status"] = "completed"

    report["duration_sec"] = round(time.time() - started, 3)
    write_json_atomic(paths["report_json"], report)
    print(f"STAGE_06_DA3_OK status={report['status']} aligned={ok_count} assisted_ply={paths['assisted_ply']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 6 conditional DA3 assistance.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    try:
        cfg = _load_config(args.config)
        run_da3_assist(cfg, force=args.force, log_level=args.log_level)
        return 0
    except Exception as exc:
        print(f"STAGE_06_DA3_FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
