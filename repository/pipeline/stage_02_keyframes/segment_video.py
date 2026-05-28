"""Segment frame-quality rows into stable subsequences for keyframe selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from pipeline.common.config import PipelineConfig


@dataclass(frozen=True)
class SegmentSummary:
    segment_id: int
    start_timestamp_sec: float
    end_timestamp_sec: float
    frame_count: int
    duration_sec: float
    strict: bool


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def rejection_tokens(value: object) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    return {part.strip() for part in text.split("|") if part.strip()}


def is_strict_quality_row(row: dict[str, object], cfg: PipelineConfig) -> tuple[bool, list[str]]:
    """Return whether a Stage 1 quality row is acceptable under strict Stage 2 gates."""
    reasons: list[str] = []
    sharpness = to_float(row.get("sharpness_laplacian"))
    exposure_jump = to_float(row.get("exposure_jump"))
    motion_score = to_float(row.get("motion_score"))
    duplicate_similarity = to_float(row.get("duplicate_similarity"))
    min_blur = float(cfg.require("video_quality.min_blur_laplacian"))
    max_exposure_jump = float(cfg.require("video_quality.max_exposure_jump"))
    min_motion = float(cfg.require("video_quality.min_motion_score"))
    max_motion = float(cfg.require("video_quality.max_motion_score"))
    max_duplicate = float(cfg.require("video_quality.max_duplicate_similarity"))

    reject_tokens = rejection_tokens(row.get("reject_reason"))
    if sharpness < min_blur or "blur" in reject_tokens:
        reasons.append("blur")
    if exposure_jump > max_exposure_jump or "exposure_jump" in reject_tokens:
        reasons.append("exposure_jump")
    if duplicate_similarity > max_duplicate or "duplicate" in reject_tokens:
        reasons.append("duplicate")
    if motion_score < min_motion or motion_score > max_motion or "motion" in reject_tokens:
        reasons.append("motion")

    if bool(cfg.get("keyframes.reject_stage1_warnings", False)):
        warning = str(row.get("warning_reason") or "")
        if warning:
            reasons.append("stage1_warning")

    if bool(cfg.get("keyframes.reject_low_feature_density", False)):
        feature_score = to_float(row.get("feature_density_score"), default=1.0)
        min_feature = float(cfg.get("video_quality.min_feature_density_score", 0.0))
        if feature_score < min_feature:
            reasons.append("low_feature_density")

    return len(reasons) == 0, reasons


def is_relaxed_quality_row(row: dict[str, object], cfg: PipelineConfig) -> tuple[bool, list[str]]:
    """Return whether a row is acceptable for controlled fallback selection.

    Fallback relaxes duplicate and Stage 1 aggregate reject reasons first, but it still
    rejects severe blur, severe exposure jumps, and extreme motion because those are
    dangerous for downstream SfM.
    """
    reasons: list[str] = []
    sharpness = to_float(row.get("sharpness_laplacian"))
    exposure_jump = to_float(row.get("exposure_jump"))
    motion_score = to_float(row.get("motion_score"))
    min_blur = float(cfg.require("video_quality.min_blur_laplacian")) * float(cfg.get("keyframes.fallback_blur_ratio", 0.65))
    max_exposure_jump = min(1.0, float(cfg.require("video_quality.max_exposure_jump")) * float(cfg.get("keyframes.fallback_exposure_multiplier", 1.5)))
    min_motion = max(0.0, float(cfg.require("video_quality.min_motion_score")) * 0.5)
    max_motion = min(1.0, float(cfg.require("video_quality.max_motion_score")) * float(cfg.get("keyframes.fallback_motion_multiplier", 1.15)))

    if sharpness < min_blur:
        reasons.append("fallback_blur")
    if exposure_jump > max_exposure_jump:
        reasons.append("fallback_exposure_jump")
    if motion_score < min_motion or motion_score > max_motion:
        reasons.append("fallback_motion")
    return len(reasons) == 0, reasons


def assign_segments(rows: Sequence[dict[str, object]], cfg: PipelineConfig, *, relaxed: bool = False) -> tuple[list[int], list[SegmentSummary]]:
    """Assign stable segment IDs to rows.

    Invalid rows receive segment_id = -1. Valid consecutive rows are grouped into
    segments; groups shorter than the configured minimum duration or sample count are
    discarded to avoid fragile single-frame islands.
    """
    if not rows:
        return [], []
    min_segment_frames = int(cfg.require("keyframes.min_segment_frames"))
    min_segment_duration_sec = float(cfg.require("keyframes.min_segment_duration_sec"))
    max_gap_sec = float(cfg.require("keyframes.max_segment_gap_sec"))
    validity: list[bool] = []
    for row in rows:
        ok, _ = is_relaxed_quality_row(row, cfg) if relaxed else is_strict_quality_row(row, cfg)
        validity.append(ok)

    segment_ids = [-1 for _ in rows]
    summaries: list[SegmentSummary] = []
    current: list[int] = []
    segment_id = 0
    last_ts: float | None = None

    def close_segment(indices: list[int]) -> None:
        nonlocal segment_id
        if not indices:
            return
        start_ts = to_float(rows[indices[0]].get("timestamp_sec"))
        end_ts = to_float(rows[indices[-1]].get("timestamp_sec"))
        duration = max(0.0, end_ts - start_ts)
        if len(indices) >= min_segment_frames and duration >= min_segment_duration_sec:
            for idx in indices:
                segment_ids[idx] = segment_id
            summaries.append(
                SegmentSummary(
                    segment_id=segment_id,
                    start_timestamp_sec=start_ts,
                    end_timestamp_sec=end_ts,
                    frame_count=len(indices),
                    duration_sec=duration,
                    strict=not relaxed,
                )
            )
            segment_id += 1

    for idx, row in enumerate(rows):
        ts = to_float(row.get("timestamp_sec"))
        if not validity[idx]:
            close_segment(current)
            current = []
            last_ts = None
            continue
        if last_ts is not None and ts - last_ts > max_gap_sec:
            close_segment(current)
            current = []
        current.append(idx)
        last_ts = ts
    close_segment(current)
    return segment_ids, summaries


def segment_summaries_to_dicts(summaries: Iterable[SegmentSummary]) -> list[dict[str, object]]:
    return [
        {
            "segment_id": item.segment_id,
            "start_timestamp_sec": item.start_timestamp_sec,
            "end_timestamp_sec": item.end_timestamp_sec,
            "frame_count": item.frame_count,
            "duration_sec": item.duration_sec,
            "strict": item.strict,
        }
        for item in summaries
    ]
