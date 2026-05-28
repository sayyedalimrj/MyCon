"""CLI and implementation for Stage 2: adaptive keyframe selection.

Stage 2 consumes only the Stage 1 file contract:
- normalized CFR video
- frame_quality.csv

It writes keyframe JPEGs, a manifest CSV, a contact sheet, and a JSON summary.
The implementation is intentionally lightweight and explainable: no COLMAP, DA3,
YOLO, SAM, or learned models are used in this stage.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import cv2

from pipeline.common.config import ConfigError, PipelineConfig, load_config
from pipeline.common.logging_utils import setup_logging
from pipeline.common.paths import atomic_output_path, output_path, run_logs_dir, run_reports_dir, write_json_atomic
from pipeline.stage_02_keyframes.contact_sheet import create_contact_sheet
from pipeline.stage_02_keyframes.novelty import (
    SelectionCandidate,
    enforce_global_time_gap,
    limit_keyframes_temporally,
    mark_dense_subset,
    normalized_score,
    pick_best_per_time_gap,
)
from pipeline.stage_02_keyframes.segment_video import (
    assign_segments,
    segment_summaries_to_dicts,
    to_float,
    to_int,
)

REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = (
    "keyframe_id",
    "source_frame_index",
    "timestamp_sec",
    "image_path",
    "segment_id",
    "sharpness_laplacian",
    "exposure_mean",
    "motion_score",
    "novelty_score",
    "quality_score",
    "keep_sparse",
    "keep_dense",
    "selection_reason",
)

OPTIONAL_MANIFEST_COLUMNS: tuple[str, ...] = (
    "exposure_jump",
    "duplicate_similarity",
    "reject_reason",
    "warning_reason",
    "feature_count",
    "feature_density_score",
    "histogram_similarity",
    "rolling_shutter_score",
    "jitter_score",
    "selection_score",
)

MANIFEST_COLUMNS: tuple[str, ...] = REQUIRED_MANIFEST_COLUMNS + OPTIONAL_MANIFEST_COLUMNS


class Stage2Error(RuntimeError):
    """Raised when Stage 2 cannot satisfy its file contract."""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2: adaptive keyframe selection.")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config, e.g. configs/site01.yaml")
    parser.add_argument("--force", action="store_true", help="Clear and rewrite Stage 2 outputs safely.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--root-override", type=Path, default=None, help="Optional project root override for tests/local runs.")
    return parser


def run_keyframe_selection(cfg: PipelineConfig, *, force: bool = False, log_level: str = "INFO") -> dict[str, Any]:
    logs_dir = run_logs_dir(cfg)
    reports_dir = run_reports_dir(cfg)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(name="pipeline.stage_02_keyframes", level=log_level, log_file=logs_dir / "stage_02_keyframes.log")
    start = time.perf_counter()

    normalized_video = output_path(cfg, "normalized_video")
    quality_csv = output_path(cfg, "quality_csv")
    keyframes_dir = output_path(cfg, "keyframes_dir")
    manifest_csv = output_path(cfg, "manifest_csv")
    contact_sheet_path = output_path(cfg, "contact_sheet")
    report_json = reports_dir / "keyframe_summary.json"

    logger.info("Starting Stage 2 for project=%s run_id=%s", cfg.project_name, cfg.run_id)
    _validate_inputs(normalized_video, quality_csv)
    _prepare_outputs(keyframes_dir, manifest_csv, contact_sheet_path, report_json, force=force)

    video_stats = _probe_video(normalized_video)
    rows = _read_quality_rows(quality_csv)
    if not rows:
        raise Stage2Error(f"Stage 1 quality CSV has no rows: {quality_csv}")
    alignment_stats = _validate_quality_video_alignment(rows, video_stats, cfg, logger)

    strict_segment_ids, strict_segments = assign_segments(rows, cfg, relaxed=False)
    for idx, segment_id in enumerate(strict_segment_ids):
        rows[idx]["segment_id"] = segment_id
    strict_indices = [idx for idx, segment_id in enumerate(strict_segment_ids) if segment_id >= 0]
    logger.info("Strict Stage 2 candidates: %d rows across %d segments", len(strict_indices), len(strict_segments))

    selected = _select_candidates(rows, strict_indices, cfg, reason="strict_quality_novelty")
    selected, fallback_used, fallback_segments = _maybe_apply_fallback(rows, selected, cfg, logger)

    selected = _postprocess_candidates(selected, cfg)
    selected, supplemental_count, emergency_used = _ensure_minimum_keyframes(rows, selected, cfg, logger)
    selected = _postprocess_candidates(selected, cfg)

    if not selected and bool(cfg.require("keyframes.emergency_fallback_if_no_keyframes")):
        selected = _emergency_best_available(rows, cfg, max_count=1)
        emergency_used = True
    if not selected:
        raise Stage2Error("No keyframes selected. Inspect Stage 1 frame_quality.csv and keyframe thresholds.")

    selected = sorted(selected, key=lambda item: item.timestamp_sec)
    image_paths = _extract_keyframes(normalized_video, selected, rows, keyframes_dir, cfg, video_stats, logger=logger)
    manifest_rows = _build_manifest_rows(rows, selected, image_paths, cfg)
    _write_manifest_csv(manifest_csv, manifest_rows)
    contact_summary = create_contact_sheet(
        image_paths,
        contact_sheet_path,
        thumb_width=int(cfg.require("keyframes.contact_sheet_thumb_width")),
        max_images=int(cfg.require("keyframes.contact_sheet_max_images")),
        label=True,
    )

    elapsed = time.perf_counter() - start
    report: dict[str, Any] = {
        "stage": "stage_02_keyframes",
        "project": cfg.project_name,
        "run_id": cfg.run_id,
        "elapsed_sec": round(elapsed, 6),
        "inputs": {
            "normalized_video": str(normalized_video),
            "quality_csv": str(quality_csv),
        },
        "outputs": {
            "keyframes_dir": str(keyframes_dir),
            "manifest_csv": str(manifest_csv),
            "contact_sheet": str(contact_sheet_path),
            "report_json": str(report_json),
        },
        "video_probe": video_stats,
        "quality_alignment": alignment_stats,
        "quality_rows": len(rows),
        "strict_candidate_count": len(strict_indices),
        "selected_keyframe_count": len(selected),
        "fallback_used": fallback_used,
        "supplemental_fallback_count": supplemental_count,
        "emergency_fallback_used": emergency_used,
        "strict_segments": segment_summaries_to_dicts(strict_segments),
        "fallback_segments": segment_summaries_to_dicts(fallback_segments),
        "contact_sheet": contact_summary,
        "config": {
            "min_time_gap_sec": float(cfg.require("keyframes.min_time_gap_sec")),
            "max_frames_first_run": int(cfg.require("keyframes.max_frames_first_run")),
            "dense_keep_ratio": float(cfg.require("keyframes.dense_keep_ratio")),
            "allow_relaxed_fallback": bool(cfg.require("keyframes.allow_relaxed_fallback")),
            "random_seek_extraction": bool(cfg.require("keyframes.random_seek_extraction")),
            "verify_frame_index_bounds": bool(cfg.require("keyframes.verify_frame_index_bounds")),
            "verify_timestamp_frame_consistency": bool(cfg.require("keyframes.verify_timestamp_frame_consistency")),
        },
    }
    write_json_atomic(report_json, report)
    logger.info("Stage 2 complete in %.3fs; selected=%d manifest=%s", elapsed, len(selected), manifest_csv)
    print(f"STAGE_02_KEYFRAMES_OK keyframes={len(selected)} manifest={manifest_csv}")
    return report


def _validate_inputs(normalized_video: Path, quality_csv: Path) -> None:
    missing = [str(path) for path in (normalized_video, quality_csv) if not path.exists()]
    if missing:
        raise FileNotFoundError("Stage 2 requires completed Stage 1 outputs. Missing: " + ", ".join(missing))
    if normalized_video.stat().st_size <= 0:
        raise Stage2Error(f"Normalized video is empty: {normalized_video}")
    if quality_csv.stat().st_size <= 0:
        raise Stage2Error(f"Frame quality CSV is empty: {quality_csv}")


def _prepare_outputs(keyframes_dir: Path, manifest_csv: Path, contact_sheet: Path, report_json: Path, *, force: bool) -> None:
    existing_outputs = []
    if keyframes_dir.exists() and any(keyframes_dir.glob("*.jpg")):
        existing_outputs.append(str(keyframes_dir))
    for path in (manifest_csv, contact_sheet, report_json):
        if path.exists():
            existing_outputs.append(str(path))
    if existing_outputs and not force:
        raise Stage2Error("Stage 2 outputs already exist. Re-run with --force to rewrite: " + ", ".join(existing_outputs))
    if force:
        if keyframes_dir.exists():
            shutil.rmtree(keyframes_dir)
        for path in (manifest_csv, contact_sheet, report_json):
            if path.exists():
                path.unlink()
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)


def _read_quality_rows(quality_csv: Path) -> list[dict[str, object]]:
    with quality_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    for idx, row in enumerate(rows):
        row.setdefault("frame_index", str(idx))
        row.setdefault("timestamp_sec", "0")
    return sorted(rows, key=lambda item: to_float(item.get("timestamp_sec")))


def _probe_video(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Stage2Error(f"Could not open normalized video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(round(float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
        width = int(round(float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)))
        height = int(round(float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)))
    finally:
        cap.release()
    if fps <= 0:
        raise Stage2Error(f"Could not determine FPS for normalized video: {video_path}")
    if frame_count <= 0:
        raise Stage2Error(f"Could not determine frame count for normalized video: {video_path}")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps if fps > 0 else 0.0,
    }


def _validate_quality_video_alignment(
    rows: Sequence[dict[str, object]],
    video_stats: dict[str, Any],
    cfg: PipelineConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    fps = float(video_stats["fps"])
    frame_count = int(video_stats["frame_count"])
    frame_indices = [to_int(row.get("frame_index"), default=-1) for row in rows]
    min_frame = min(frame_indices) if frame_indices else -1
    max_frame = max(frame_indices) if frame_indices else -1
    if bool(cfg.require("keyframes.verify_frame_index_bounds")):
        if min_frame < 0:
            raise Stage2Error("Stage 1 quality CSV contains negative or invalid frame_index values.")
        if max_frame >= frame_count:
            raise Stage2Error(
                f"Stage 1 quality CSV references frame_index={max_frame}, but normalized video has "
                f"only {frame_count} frames. Re-run Stage 1 normalization and quality scoring."
            )
    max_drift = 0.0
    if bool(cfg.require("keyframes.verify_timestamp_frame_consistency")):
        tolerance = float(cfg.require("keyframes.max_timestamp_frame_index_drift_sec"))
        for row, frame_index in zip(rows, frame_indices):
            timestamp = to_float(row.get("timestamp_sec"))
            expected = frame_index / fps
            max_drift = max(max_drift, abs(timestamp - expected))
        if max_drift > tolerance:
            raise Stage2Error(
                f"Stage 1 timestamp/frame_index drift is too high: {max_drift:.6f}s > {tolerance:.6f}s. "
                "This usually indicates a VFR/CFR mismatch. Re-run Stage 1 with CFR verification enabled."
            )
    logger.info(
        "Stage 1/2 alignment check: fps=%.3f frame_count=%d max_csv_frame=%d max_timestamp_drift=%.6fs",
        fps,
        frame_count,
        max_frame,
        max_drift,
    )
    return {
        "min_frame_index": min_frame,
        "max_frame_index": max_frame,
        "video_frame_count": frame_count,
        "video_fps": fps,
        "max_timestamp_frame_index_drift_sec": round(max_drift, 6),
    }


def _select_candidates(
    rows: list[dict[str, object]],
    row_indices: Sequence[int],
    cfg: PipelineConfig,
    *,
    reason: str,
) -> list[SelectionCandidate]:
    by_segment: dict[int, list[int]] = {}
    for idx in row_indices:
        segment_id = int(rows[idx].get("segment_id", -1))
        if segment_id < 0:
            continue
        by_segment.setdefault(segment_id, []).append(idx)
    selected: list[SelectionCandidate] = []
    for segment_id in sorted(by_segment):
        selected.extend(pick_best_per_time_gap(rows, by_segment[segment_id], cfg, reason=reason))
    return sorted(selected, key=lambda item: item.timestamp_sec)


def _maybe_apply_fallback(
    rows: list[dict[str, object]],
    selected: list[SelectionCandidate],
    cfg: PipelineConfig,
    logger: logging.Logger,
) -> tuple[list[SelectionCandidate], bool, list[Any]]:
    fallback_min = int(cfg.require("keyframes.fallback_min_keyframes"))
    if len(selected) >= fallback_min or not bool(cfg.require("keyframes.allow_relaxed_fallback")):
        return selected, False, []
    relaxed_ids, relaxed_segments = assign_segments(rows, cfg, relaxed=True)
    for idx, segment_id in enumerate(relaxed_ids):
        rows[idx]["relaxed_segment_id"] = segment_id
    relaxed_indices = [idx for idx, segment_id in enumerate(relaxed_ids) if segment_id >= 0]
    for idx in relaxed_indices:
        if int(rows[idx].get("segment_id", -1)) < 0:
            rows[idx]["segment_id"] = rows[idx].get("relaxed_segment_id", -1)
    logger.warning(
        "Strict selection produced %d keyframes; applying controlled fallback with %d relaxed rows",
        len(selected),
        len(relaxed_indices),
    )
    fallback_selected = _select_candidates(rows, relaxed_indices, cfg, reason="relaxed_fallback_for_sequence_continuity")
    merged = _merge_candidates(selected, fallback_selected)
    return merged, True, relaxed_segments


def _postprocess_candidates(candidates: Sequence[SelectionCandidate], cfg: PipelineConfig) -> list[SelectionCandidate]:
    min_gap = float(cfg.require("keyframes.min_time_gap_sec"))
    max_count = int(cfg.require("keyframes.max_frames_first_run"))
    selected = enforce_global_time_gap(candidates, min_gap)
    selected = limit_keyframes_temporally(selected, max_count)
    selected = enforce_global_time_gap(selected, min_gap)
    return sorted(selected, key=lambda item: item.timestamp_sec)


def _ensure_minimum_keyframes(
    rows: list[dict[str, object]],
    selected: list[SelectionCandidate],
    cfg: PipelineConfig,
    logger: logging.Logger,
) -> tuple[list[SelectionCandidate], int, bool]:
    fallback_min = int(cfg.require("keyframes.fallback_min_keyframes"))
    max_count = int(cfg.require("keyframes.max_frames_first_run"))
    if len(selected) >= min(fallback_min, max_count):
        return selected, 0, False
    if not bool(cfg.require("keyframes.allow_relaxed_fallback")):
        return selected, 0, False

    relaxed_ids, _ = assign_segments(rows, cfg, relaxed=True)
    relaxed_indices = [idx for idx, segment_id in enumerate(relaxed_ids) if segment_id >= 0]
    pool_indices = relaxed_indices if relaxed_indices else list(range(len(rows)))
    pool = _candidate_pool_from_indices(
        rows,
        pool_indices,
        cfg,
        reason="supplemental_fallback_to_reach_minimum" if relaxed_indices else "emergency_best_available_no_valid_segment",
    )

    before = len(selected)
    selected = _add_candidates_preserving_gap(selected, pool, cfg, target_count=min(fallback_min, max_count))
    supplemental_count = max(0, len(selected) - before)
    emergency_used = not relaxed_indices and supplemental_count > 0

    if len(selected) < min(fallback_min, max_count) and bool(cfg.require("keyframes.emergency_fallback_if_no_keyframes")):
        emergency_pool = _candidate_pool_from_indices(rows, range(len(rows)), cfg, reason="emergency_best_available_to_avoid_empty_manifest")
        before_emergency = len(selected)
        selected = _add_candidates_preserving_gap(selected, emergency_pool, cfg, target_count=min(fallback_min, max_count))
        supplemental_count += max(0, len(selected) - before_emergency)
        emergency_used = emergency_used or len(selected) > before_emergency

    if supplemental_count:
        logger.warning("Stage 2 added %d supplemental fallback keyframes; final candidate count=%d", supplemental_count, len(selected))
    return sorted(selected, key=lambda item: item.timestamp_sec), supplemental_count, emergency_used


def _candidate_pool_from_indices(
    rows: Sequence[dict[str, object]],
    indices: Sequence[int] | range,
    cfg: PipelineConfig,
    *,
    reason: str,
) -> list[SelectionCandidate]:
    candidates: list[SelectionCandidate] = []
    for idx in indices:
        row = rows[int(idx)]
        segment_id = int(row.get("segment_id", row.get("relaxed_segment_id", -1)) or -1)
        candidates.append(
            SelectionCandidate(
                row_index=int(idx),
                score=normalized_score(row, cfg),
                timestamp_sec=to_float(row.get("timestamp_sec")),
                segment_id=segment_id,
                selection_reason=reason,
            )
        )
    return candidates


def _add_candidates_preserving_gap(
    selected: Sequence[SelectionCandidate],
    pool: Sequence[SelectionCandidate],
    cfg: PipelineConfig,
    *,
    target_count: int,
) -> list[SelectionCandidate]:
    min_gap = float(cfg.require("keyframes.min_time_gap_sec"))
    max_count = int(cfg.require("keyframes.max_frames_first_run"))
    target = min(target_count, max_count)
    by_row = {item.row_index: item for item in selected}
    kept = list(selected)
    for candidate in sorted(pool, key=lambda item: (-item.score, item.timestamp_sec)):
        if len(kept) >= target:
            break
        if candidate.row_index in by_row:
            continue
        if all(abs(candidate.timestamp_sec - existing.timestamp_sec) >= min_gap for existing in kept):
            kept.append(candidate)
            by_row[candidate.row_index] = candidate
    if not kept and pool:
        best = max(pool, key=lambda item: item.score)
        kept.append(best)
    return sorted(kept, key=lambda item: item.timestamp_sec)


def _emergency_best_available(rows: Sequence[dict[str, object]], cfg: PipelineConfig, *, max_count: int) -> list[SelectionCandidate]:
    pool = _candidate_pool_from_indices(rows, range(len(rows)), cfg, reason="emergency_best_available_no_selected_rows")
    return _add_candidates_preserving_gap([], pool, cfg, target_count=max_count)


def _merge_candidates(primary: Sequence[SelectionCandidate], fallback: Sequence[SelectionCandidate]) -> list[SelectionCandidate]:
    by_frame: dict[int, SelectionCandidate] = {}
    for item in list(fallback) + list(primary):
        row_index = item.row_index
        previous = by_frame.get(row_index)
        if previous is None or item.score >= previous.score:
            by_frame[row_index] = item
    return sorted(by_frame.values(), key=lambda item: item.timestamp_sec)


def _extract_keyframes(
    video_path: Path,
    selected: Sequence[SelectionCandidate],
    rows: Sequence[dict[str, object]],
    keyframes_dir: Path,
    cfg: PipelineConfig,
    video_stats: dict[str, Any],
    *,
    logger: logging.Logger,
) -> list[Path]:
    if bool(cfg.require("keyframes.random_seek_extraction")):
        return _extract_keyframes_random_seek(video_path, selected, rows, keyframes_dir, cfg, video_stats, logger=logger)
    return _extract_keyframes_sequential(video_path, selected, rows, keyframes_dir, cfg, video_stats, logger=logger)


def _extract_keyframes_sequential(
    video_path: Path,
    selected: Sequence[SelectionCandidate],
    rows: Sequence[dict[str, object]],
    keyframes_dir: Path,
    cfg: PipelineConfig,
    video_stats: dict[str, Any],
    *,
    logger: logging.Logger,
) -> list[Path]:
    targets = _target_frame_map(selected, rows)
    if not targets:
        return []
    max_target = max(targets)
    frame_count = int(video_stats.get("frame_count", 0))
    if frame_count > 0 and max_target >= frame_count:
        raise Stage2Error(f"Requested keyframe frame_index={max_target}, but video has only {frame_count} frames.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Stage2Error(f"Could not open normalized video for keyframe extraction: {video_path}")
    image_paths: list[Path] = []
    frame_idx = 0
    jpeg_quality = int(cfg.require("keyframes.jpeg_quality"))
    logger.info("Extracting %d keyframes by sequential decode up to frame %d", len(targets), max_target)
    try:
        while frame_idx <= max_target:
            ok, frame = cap.read()
            if not ok:
                break
            candidates = targets.get(frame_idx)
            if candidates:
                for _candidate in candidates:
                    path = _keyframe_path(keyframes_dir, cfg.project_name, len(image_paths) + 1, frame_idx)
                    _write_jpeg_atomic(path, frame, jpeg_quality)
                    image_paths.append(path)
            frame_idx += 1
    finally:
        cap.release()
    if len(image_paths) != len(selected):
        raise Stage2Error(
            f"Extracted {len(image_paths)} keyframes, expected {len(selected)}. "
            "Check source_frame_index values and normalized video frame count."
        )
    return image_paths


def _extract_keyframes_random_seek(
    video_path: Path,
    selected: Sequence[SelectionCandidate],
    rows: Sequence[dict[str, object]],
    keyframes_dir: Path,
    cfg: PipelineConfig,
    video_stats: dict[str, Any],
    *,
    logger: logging.Logger,
) -> list[Path]:
    frame_count = int(video_stats.get("frame_count", 0))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Stage2Error(f"Could not open normalized video for keyframe extraction: {video_path}")
    image_paths: list[Path] = []
    jpeg_quality = int(cfg.require("keyframes.jpeg_quality"))
    logger.warning("Using random seek extraction because keyframes.random_seek_extraction=true")
    try:
        for candidate in selected:
            frame_idx = to_int(rows[candidate.row_index].get("frame_index"))
            if frame_count > 0 and frame_idx >= frame_count:
                raise Stage2Error(f"Requested frame_index={frame_idx}, but video has only {frame_count} frames.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                raise Stage2Error(f"Failed to extract frame_index={frame_idx} from {video_path}")
            path = _keyframe_path(keyframes_dir, cfg.project_name, len(image_paths) + 1, frame_idx)
            _write_jpeg_atomic(path, frame, jpeg_quality)
            image_paths.append(path)
    finally:
        cap.release()
    return image_paths


def _target_frame_map(selected: Sequence[SelectionCandidate], rows: Sequence[dict[str, object]]) -> dict[int, list[SelectionCandidate]]:
    targets: dict[int, list[SelectionCandidate]] = {}
    for candidate in selected:
        frame_idx = to_int(rows[candidate.row_index].get("frame_index"))
        targets.setdefault(frame_idx, []).append(candidate)
    return targets


def _keyframe_path(keyframes_dir: Path, project_name: str, keyframe_id: int, source_frame_index: int) -> Path:
    safe_project = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in project_name)
    return keyframes_dir / f"{safe_project}_kf_{keyframe_id:05d}_f{source_frame_index:06d}.jpg"


def _write_jpeg_atomic(path: Path, frame: Any, jpeg_quality: int) -> None:
    with atomic_output_path(path) as tmp_path:
        ok = cv2.imwrite(str(tmp_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise Stage2Error(f"Failed to write keyframe image: {tmp_path}")


def _build_manifest_rows(
    rows: Sequence[dict[str, object]],
    selected: Sequence[SelectionCandidate],
    image_paths: Sequence[Path],
    cfg: PipelineConfig,
) -> list[dict[str, object]]:
    dense_flags = mark_dense_subset(len(selected), float(cfg.require("keyframes.dense_keep_ratio")))
    manifest_rows: list[dict[str, object]] = []
    for idx, (candidate, image_path) in enumerate(zip(selected, image_paths), start=1):
        row = rows[candidate.row_index]
        manifest_rows.append(
            {
                "keyframe_id": f"kf_{idx:05d}",
                "source_frame_index": to_int(row.get("frame_index")),
                "timestamp_sec": f"{to_float(row.get('timestamp_sec')):.6f}",
                "image_path": _relative_to_root(image_path, cfg),
                "segment_id": int(row.get("segment_id", row.get("relaxed_segment_id", -1)) or -1),
                "sharpness_laplacian": _format_float(row.get("sharpness_laplacian")),
                "exposure_mean": _format_float(row.get("exposure_mean")),
                "motion_score": _format_float(row.get("motion_score")),
                "novelty_score": _format_float(row.get("novelty_score")),
                "quality_score": _format_float(row.get("quality_score")),
                "keep_sparse": "true",
                "keep_dense": "true" if dense_flags[idx - 1] else "false",
                "selection_reason": candidate.selection_reason,
                "exposure_jump": _format_float(row.get("exposure_jump")),
                "duplicate_similarity": _format_float(row.get("duplicate_similarity")),
                "reject_reason": str(row.get("reject_reason") or ""),
                "warning_reason": str(row.get("warning_reason") or ""),
                "feature_count": str(to_int(row.get("feature_count"))),
                "feature_density_score": _format_float(row.get("feature_density_score")),
                "histogram_similarity": _format_float(row.get("histogram_similarity")),
                "rolling_shutter_score": _format_float(row.get("rolling_shutter_score")),
                "jitter_score": _format_float(row.get("jitter_score")),
                "selection_score": f"{candidate.score:.6f}",
            }
        )
    return manifest_rows


def _write_manifest_csv(manifest_csv: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise Stage2Error("Cannot write an empty keyframe manifest.")
    with atomic_output_path(manifest_csv) as tmp_path:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def _relative_to_root(path: Path, cfg: PipelineConfig) -> str:
    try:
        return path.relative_to(cfg.project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _format_float(value: object) -> str:
    return f"{to_float(value):.6f}"


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, root_override=args.root_override)
        run_keyframe_selection(cfg, force=args.force, log_level=args.log_level)
    except (ConfigError, FileNotFoundError, Stage2Error, RuntimeError, ValueError) as exc:
        logging.getLogger("pipeline.stage_02_keyframes").handlers.clear()
        print(f"STAGE_02_KEYFRAMES_FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
