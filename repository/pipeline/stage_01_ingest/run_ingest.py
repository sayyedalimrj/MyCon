"""CLI entrypoint for Stage 1: ingest and normalization."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from pipeline.common.config import ConfigError, PipelineConfig, load_config
from pipeline.common.logging_utils import setup_logging
from pipeline.common.paths import input_path, output_path, run_logs_dir, run_reports_dir, write_json_atomic
from pipeline.stage_01_ingest.frame_quality import compute_frame_quality_table
from pipeline.stage_01_ingest.metadata import extract_video_metadata, write_metadata_json
from pipeline.stage_01_ingest.normalize_video import normalize_video


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1: normalize video and compute frame quality.")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config, e.g. configs/site01.yaml")
    parser.add_argument("--force", action="store_true", help="Force re-normalization and overwrite outputs.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--root-override", type=Path, default=None, help="Optional project root override for tests/local runs.")
    return parser


def run_ingest(cfg: PipelineConfig, *, force: bool = False, log_level: str = "INFO") -> dict[str, Any]:
    logs_dir = run_logs_dir(cfg)
    reports_dir = run_reports_dir(cfg)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(name="pipeline.stage_01_ingest", level=log_level, log_file=logs_dir / "stage_01_ingest.log")
    start = time.perf_counter()
    raw_video = input_path(cfg, "video")
    normalized_video = output_path(cfg, "normalized_video")
    metadata_json = output_path(cfg, "metadata_json")
    quality_csv = output_path(cfg, "quality_csv")
    report_json = reports_dir / "stage_01_ingest_report.json"
    logger.info("Starting Stage 1 for project=%s run_id=%s", cfg.project_name, cfg.run_id)
    logger.info("Input video: %s", raw_video)
    if not raw_video.exists():
        raise FileNotFoundError(f"Configured input video does not exist: {raw_video}")
    normalization = normalize_video(raw_video, normalized_video, cfg, logger=logger, force=force)
    metadata = extract_video_metadata(normalized_video, logger=logger)
    write_metadata_json(metadata, metadata_json)
    quality_summary = compute_frame_quality_table(normalized_video, quality_csv, cfg, logger=logger)
    elapsed = time.perf_counter() - start
    report: dict[str, Any] = {
        "stage": "stage_01_ingest",
        "project": cfg.project_name,
        "run_id": cfg.run_id,
        "elapsed_sec": round(elapsed, 6),
        "inputs": {"video": str(raw_video)},
        "outputs": {
            "normalized_video": str(normalized_video),
            "metadata_json": str(metadata_json),
            "quality_csv": str(quality_csv),
            "report_json": str(report_json),
        },
        "normalization": {
            "normalized": normalization.normalized,
            "skipped_reencode": normalization.skipped_reencode,
            "ffmpeg_command": normalization.ffmpeg_command,
        },
        "metadata": {
            "duration_sec": metadata.get("duration_sec"),
            "fps": metadata.get("fps"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "frame_count": metadata.get("frame_count"),
            "codec": metadata.get("codec"),
            "pix_fmt": metadata.get("pix_fmt"),
            "rotation_degrees": metadata.get("rotation_degrees"),
        },
        "quality": {
            "sampled_frame_count": quality_summary.sampled_frame_count,
            "rejected_frame_count": quality_summary.rejected_frame_count,
            "columns": list(quality_summary.columns),
            "sampling_stride": quality_summary.sampling_stride,
            "sampling_method": quality_summary.sampling_method,
        },
    }
    write_json_atomic(report_json, report)
    logger.info("Stage 1 complete in %.3fs", elapsed)
    print(f"STAGE_01_INGEST_OK normalized_video={normalized_video} quality_rows={quality_summary.sampled_frame_count}")
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, root_override=args.root_override)
        run_ingest(cfg, force=args.force, log_level=args.log_level)
    except (ConfigError, FileNotFoundError, RuntimeError) as exc:
        logging.getLogger("pipeline.stage_01_ingest").handlers.clear()
        print(f"STAGE_01_INGEST_FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
