"""Scoring and diversity helpers for adaptive keyframe selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pipeline.common.config import PipelineConfig
from pipeline.stage_02_keyframes.segment_video import to_float


@dataclass(frozen=True)
class SelectionCandidate:
    row_index: int
    score: float
    timestamp_sec: float
    segment_id: int
    selection_reason: str


def normalized_score(row: dict[str, object], cfg: PipelineConfig) -> float:
    quality_weight = float(cfg.require("keyframes.selection_quality_weight"))
    novelty_weight = float(cfg.require("keyframes.selection_novelty_weight"))
    feature_weight = float(cfg.require("keyframes.selection_feature_weight"))
    quality = _clip01(to_float(row.get("quality_score")))
    novelty = _clip01(to_float(row.get("novelty_score")))
    feature = _clip01(to_float(row.get("feature_density_score"), default=0.5))
    total = quality_weight + novelty_weight + feature_weight
    if total <= 0:
        return quality
    return (quality_weight * quality + novelty_weight * novelty + feature_weight * feature) / total


def pick_best_per_time_gap(
    rows: Sequence[dict[str, object]],
    row_indices: Sequence[int],
    cfg: PipelineConfig,
    *,
    reason: str,
) -> list[SelectionCandidate]:
    """Select locally best rows while enforcing a minimum time gap.

    The method partitions each continuous candidate sequence into temporal buckets of
    approximately min_time_gap_sec and picks the best-scoring frame from each bucket.
    This preserves chronological coverage while rejecting redundant frames.
    """
    min_gap = float(cfg.require("keyframes.min_time_gap_sec"))
    if min_gap <= 0:
        raise ValueError("keyframes.min_time_gap_sec must be positive")
    if not row_indices:
        return []
    sorted_indices = sorted(row_indices, key=lambda idx: to_float(rows[idx].get("timestamp_sec")))
    selected: list[SelectionCandidate] = []
    bucket: list[int] = []
    bucket_start: float | None = None

    def close_bucket(items: list[int]) -> None:
        if not items:
            return
        best_idx = max(items, key=lambda idx: normalized_score(rows[idx], cfg))
        selected.append(
            SelectionCandidate(
                row_index=best_idx,
                score=normalized_score(rows[best_idx], cfg),
                timestamp_sec=to_float(rows[best_idx].get("timestamp_sec")),
                segment_id=int(rows[best_idx].get("segment_id", -1)),
                selection_reason=reason,
            )
        )

    for row_idx in sorted_indices:
        ts = to_float(rows[row_idx].get("timestamp_sec"))
        if bucket_start is None:
            bucket_start = ts
            bucket = [row_idx]
            continue
        if ts - bucket_start < min_gap:
            bucket.append(row_idx)
        else:
            close_bucket(bucket)
            bucket_start = ts
            bucket = [row_idx]
    close_bucket(bucket)
    return sorted(selected, key=lambda item: item.timestamp_sec)


def limit_keyframes_temporally(candidates: Sequence[SelectionCandidate], max_count: int) -> list[SelectionCandidate]:
    """Limit candidates to max_count while preserving temporal coverage and score."""
    if max_count <= 0:
        raise ValueError("max_count must be positive")
    ordered = sorted(candidates, key=lambda item: item.timestamp_sec)
    if len(ordered) <= max_count:
        return ordered
    if max_count == 1:
        return [max(ordered, key=lambda item: item.score)]
    bins: list[list[SelectionCandidate]] = [[] for _ in range(max_count)]
    for pos, candidate in enumerate(ordered):
        bin_idx = min(max_count - 1, int(pos * max_count / len(ordered)))
        bins[bin_idx].append(candidate)
    limited: list[SelectionCandidate] = []
    for bucket in bins:
        if bucket:
            limited.append(max(bucket, key=lambda item: item.score))
    return sorted(limited, key=lambda item: item.timestamp_sec)


def mark_dense_subset(total_count: int, dense_keep_ratio: float) -> list[bool]:
    """Return deterministic keep_dense flags for a chronological keyframe list."""
    if total_count <= 0:
        return []
    ratio = max(0.0, min(1.0, dense_keep_ratio))
    if ratio >= 0.999:
        return [True] * total_count
    target = max(1, int(round(total_count * ratio)))
    if target >= total_count:
        return [True] * total_count
    flags = [False] * total_count
    if target == 1:
        flags[total_count // 2] = True
        return flags
    for i in range(target):
        idx = round(i * (total_count - 1) / (target - 1))
        flags[int(idx)] = True
    return flags


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def enforce_global_time_gap(candidates: Sequence[SelectionCandidate], min_gap_sec: float) -> list[SelectionCandidate]:
    """Enforce a global minimum time gap after merging segments/fallbacks.

    Adjacent segments can otherwise contribute frames that are too close in time.
    When two candidates collide, the higher scoring frame is retained.
    """
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: item.timestamp_sec)
    kept: list[SelectionCandidate] = []
    for candidate in ordered:
        if not kept:
            kept.append(candidate)
            continue
        if candidate.timestamp_sec - kept[-1].timestamp_sec >= min_gap_sec:
            kept.append(candidate)
            continue
        if candidate.score > kept[-1].score:
            kept[-1] = candidate
    return sorted(kept, key=lambda item: item.timestamp_sec)
