"""Streaming frame-quality scoring for Stage 1.

This module intentionally stays lightweight: it uses OpenCV-only signals that are
safe to run before COLMAP. The output CSV is the file contract consumed by Stage 2.

Important engineering choices:
- Sequential decoding is the default because mobile H.264/H.265 Long-GOP videos
  can make frame-accurate OpenCV seeking unreliable. A fast-seek mode remains
  available for experiments, but it is not the baseline.
- Quality metrics are computed on a uniformly downscaled scoring frame so blur,
  histogram, and feature-density signals are less resolution-dependent.
- Rolling-shutter and jitter heuristics are recorded as warnings by default, not
  hard rejection causes, because they are approximate proxies.
"""

from __future__ import annotations

import csv
import logging
import statistics
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from pipeline.common.config import PipelineConfig
from pipeline.common.paths import atomic_output_path


REQUIRED_QUALITY_COLUMNS: tuple[str, ...] = (
    "frame_index", "timestamp_sec", "sharpness_laplacian", "exposure_mean", "exposure_std",
    "exposure_jump", "motion_score", "duplicate_similarity", "novelty_score",
    "quality_score", "reject_reason",
)

EXTRA_QUALITY_COLUMNS: tuple[str, ...] = (
    "histogram_similarity", "feature_count", "feature_density_score",
    "adaptive_blur_threshold", "rolling_shutter_score", "jitter_score",
    "warning_reason", "sampling_method", "scoring_width", "scoring_height",
)

QUALITY_COLUMNS: tuple[str, ...] = REQUIRED_QUALITY_COLUMNS + EXTRA_QUALITY_COLUMNS


@dataclass(frozen=True)
class FrameQualitySummary:
    video_path: str
    quality_csv: str
    sampled_frame_count: int
    rejected_frame_count: int
    columns: tuple[str, ...]
    sampling_stride: int
    sampling_method: str


@dataclass
class _PreviousFrameState:
    gray_small: np.ndarray | None = None
    histogram: np.ndarray | None = None
    exposure_mean: float | None = None


@dataclass(frozen=True)
class _FeatureDensityEvaluator:
    enabled: bool
    detector_name: str
    detector: Any | None
    max_features: int
    target_feature_count: int


def compute_frame_quality_table(
    video_path: Path,
    quality_csv: Path,
    cfg: PipelineConfig,
    *,
    logger: logging.Logger,
) -> FrameQualitySummary:
    """Compute the Stage 1 frame-quality CSV in streaming mode.

    The CSV is written atomically. Feature detectors are instantiated once per
    video, not once per frame, to avoid CPU/memory churn on long site videos.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Normalized video does not exist: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video for quality scoring: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if fps <= 0:
        raise RuntimeError(f"Could not determine FPS for quality scoring: {video_path}")

    sample_fps = float(cfg.require("video.sample_fps_for_quality"))
    if sample_fps <= 0:
        raise ValueError("video.sample_fps_for_quality must be greater than zero")
    stride = max(1, int(round(fps / sample_fps)))
    fast_seek = bool(cfg.get("video_quality.fast_seek", False))
    feature_evaluator = _create_feature_density_evaluator(cfg)

    logger.info(
        "Computing frame quality: fps=%.3f frame_count=%d sample_fps=%.3f stride=%d fast_seek=%s feature_detector=%s",
        fps, frame_count, sample_fps, stride, fast_seek, feature_evaluator.detector_name,
    )

    rows_written = 0
    rejected = 0
    previous = _PreviousFrameState()
    sharpness_window: deque[float] = deque(maxlen=max(1, int(cfg.get("video_quality.adaptive_blur_window", 25))))

    with atomic_output_path(quality_csv) as tmp_csv:
        with tmp_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(QUALITY_COLUMNS))
            writer.writeheader()
            for frame_index, timestamp_sec, frame, method in _iter_sampled_frames(
                video_path, fps=fps, frame_count=frame_count, stride=stride, fast_seek=fast_seek, logger=logger,
            ):
                row = _score_frame(
                    frame=frame,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    cfg=cfg,
                    previous=previous,
                    sharpness_window=sharpness_window,
                    feature_evaluator=feature_evaluator,
                    sampling_method=method,
                )
                writer.writerow(row)
                rows_written += 1
                if row["reject_reason"]:
                    rejected += 1

    if rows_written == 0:
        raise RuntimeError(f"No frames were sampled from video: {video_path}")
    logger.info("Frame quality complete: rows=%d rejected=%d csv=%s", rows_written, rejected, quality_csv)
    return FrameQualitySummary(
        str(video_path), str(quality_csv), rows_written, rejected, QUALITY_COLUMNS, stride,
        "fast_seek" if fast_seek else "sequential_decode",
    )


def _iter_sampled_frames(
    video_path: Path,
    *,
    fps: float,
    frame_count: int,
    stride: int,
    fast_seek: bool,
    logger: logging.Logger,
) -> Iterator[tuple[int, float, np.ndarray, str]]:
    if fast_seek:
        logger.warning(
            "video_quality.fast_seek=true is experimental for mobile Long-GOP videos; "
            "sequential_decode is the recommended baseline."
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")
        last_position = -1
        try:
            for frame_index in range(0, max(frame_count, 1), stride):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    logger.warning("Fast seek failed at frame %d; stopping sampled read.", frame_index)
                    break
                actual_position = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or frame_index + 1) - 1
                if actual_position <= last_position:
                    logger.warning(
                        "Fast seek returned a non-increasing frame index at requested=%d actual=%d; "
                        "continuing with requested index in the CSV.",
                        frame_index,
                        actual_position,
                    )
                last_position = max(last_position, actual_position)
                yield frame_index, frame_index / fps, frame, "fast_seek"
        finally:
            capture.release()
        return

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % stride == 0:
                yield frame_index, frame_index / fps, frame, "sequential_decode"
            frame_index += 1
    finally:
        capture.release()


def _score_frame(
    *,
    frame: np.ndarray,
    frame_index: int,
    timestamp_sec: float,
    cfg: PipelineConfig,
    previous: _PreviousFrameState,
    sharpness_window: deque[float],
    feature_evaluator: _FeatureDensityEvaluator,
    sampling_method: str,
) -> dict[str, str | int | float]:
    scoring_frame = _resize_for_scoring(frame, cfg)
    gray = cv2.cvtColor(scoring_frame, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32) / 255.0
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    exposure_mean = float(gray_float.mean())
    exposure_std = float(gray_float.std())
    exposure_jump = abs(exposure_mean - previous.exposure_mean) if previous.exposure_mean is not None else 0.0

    gray_small = cv2.resize(gray_float, (64, 64), interpolation=cv2.INTER_AREA)
    if previous.gray_small is None:
        motion = 0.0
        pixel_similarity = 1.0
        rolling_shutter_score = 0.0
    else:
        diff = np.abs(gray_small - previous.gray_small)
        motion = float(diff.mean())
        pixel_similarity = float(max(0.0, 1.0 - motion))
        rolling_shutter_score = _rolling_shutter_proxy(gray_small, previous.gray_small)

    histogram = _gray_histogram(gray, bins=int(cfg.get("video_quality.histogram_bins", 32)))
    histogram_similarity = 1.0 if previous.histogram is None else _cosine_similarity(histogram, previous.histogram)
    duplicate_similarity = max(pixel_similarity, histogram_similarity)
    novelty = max(0.0, min(1.0, 1.0 - duplicate_similarity))
    feature_count, feature_density_score = _feature_density(gray, feature_evaluator)
    adaptive_threshold = _adaptive_blur_threshold(cfg, sharpness_window)
    jitter_score = _jitter_proxy(motion_score=motion, novelty_score=novelty)

    reject_reasons = _reject_reasons(
        cfg=cfg,
        sharpness=sharpness,
        adaptive_blur_threshold=adaptive_threshold,
        exposure_jump=exposure_jump,
        motion_score=motion,
        duplicate_similarity=duplicate_similarity,
        feature_density_score=feature_density_score,
        rolling_shutter_score=rolling_shutter_score,
        jitter_score=jitter_score,
    )
    warning_reasons = _warning_reasons(
        cfg=cfg,
        rolling_shutter_score=rolling_shutter_score,
        jitter_score=jitter_score,
    )
    quality = _quality_score(
        cfg=cfg,
        sharpness=sharpness,
        adaptive_blur_threshold=adaptive_threshold,
        exposure_jump=exposure_jump,
        motion_score=motion,
        novelty_score=novelty,
        feature_density_score=feature_density_score,
        rolling_shutter_score=rolling_shutter_score,
        jitter_score=jitter_score,
    )

    if _should_update_sharpness_window(sharpness=sharpness, threshold=adaptive_threshold, cfg=cfg, window=sharpness_window):
        sharpness_window.append(sharpness)

    previous.gray_small = gray_small
    previous.histogram = histogram
    previous.exposure_mean = exposure_mean
    return {
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp_sec, 6),
        "sharpness_laplacian": round(sharpness, 6),
        "exposure_mean": round(exposure_mean, 6),
        "exposure_std": round(exposure_std, 6),
        "exposure_jump": round(exposure_jump, 6),
        "motion_score": round(motion, 6),
        "duplicate_similarity": round(duplicate_similarity, 6),
        "novelty_score": round(novelty, 6),
        "quality_score": round(quality, 6),
        "reject_reason": ",".join(reject_reasons),
        "histogram_similarity": round(histogram_similarity, 6),
        "feature_count": feature_count,
        "feature_density_score": round(feature_density_score, 6),
        "adaptive_blur_threshold": round(adaptive_threshold, 6),
        "rolling_shutter_score": round(rolling_shutter_score, 6),
        "jitter_score": round(jitter_score, 6),
        "warning_reason": ",".join(warning_reasons),
        "sampling_method": sampling_method,
        "scoring_width": int(scoring_frame.shape[1]),
        "scoring_height": int(scoring_frame.shape[0]),
    }


def _resize_for_scoring(frame: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    max_width = int(cfg.get("video_quality.scoring_max_width", 1280))
    max_height = int(cfg.get("video_quality.scoring_max_height", 720))
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Cannot score an empty frame")
    scale = min(max_width / float(width), max_height / float(height), 1.0)
    if scale >= 0.999:
        return frame
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _gray_histogram(gray: np.ndarray, *, bins: int) -> np.ndarray:
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).astype(np.float32).flatten()
    norm = float(np.linalg.norm(hist))
    return hist if norm <= 1e-12 else hist / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denominator = norm_a * norm_b
    if denominator <= 1e-12:
        return 1.0 if norm_a <= 1e-12 and norm_b <= 1e-12 else 0.0
    result = float(np.dot(a, b) / denominator)
    return float(np.clip(result, -1.0, 1.0))


def _create_feature_density_evaluator(cfg: PipelineConfig) -> _FeatureDensityEvaluator:
    enabled = bool(cfg.get("video_quality.use_feature_density", True))
    detector_name = str(cfg.get("video_quality.feature_detector", "ORB")).upper()
    max_features = max(1, int(cfg.get("video_quality.feature_max_keypoints", 1000)))
    target = max(1, int(cfg.get("video_quality.target_feature_count", 450)))
    if not enabled:
        return _FeatureDensityEvaluator(False, "DISABLED", None, max_features, target)
    if detector_name == "FAST":
        detector = cv2.FastFeatureDetector_create()
    elif detector_name == "ORB":
        detector = cv2.ORB_create(nfeatures=max_features)
    else:
        raise ValueError("video_quality.feature_detector must be ORB or FAST")
    return _FeatureDensityEvaluator(True, detector_name, detector, max_features, target)


def _feature_density(gray: np.ndarray, evaluator: _FeatureDensityEvaluator) -> tuple[int, float]:
    if not evaluator.enabled or evaluator.detector is None:
        return 0, 1.0
    keypoints = evaluator.detector.detect(gray, None)
    count = min(len(keypoints), evaluator.max_features)
    return int(count), float(max(0.0, min(1.0, count / float(evaluator.target_feature_count))))


def _adaptive_blur_threshold(cfg: PipelineConfig, window: deque[float]) -> float:
    base = float(cfg.require("video_quality.min_blur_laplacian"))
    if not bool(cfg.get("video_quality.adaptive_blur_enabled", True)) or not window:
        return base
    multiplier = float(cfg.get("video_quality.adaptive_blur_multiplier", 0.60))
    floor = float(cfg.get("video_quality.adaptive_blur_floor", 25.0))
    median_value = float(statistics.median(window))
    return max(floor, min(base, median_value * multiplier))


def _should_update_sharpness_window(
    *,
    sharpness: float,
    threshold: float,
    cfg: PipelineConfig,
    window: deque[float] | None = None,
) -> bool:
    floor = float(cfg.get("video_quality.adaptive_blur_floor", 25.0))
    min_samples = int(cfg.get("video_quality.adaptive_blur_min_window_samples", 5))
    if sharpness < floor:
        return False
    if window is not None and len(window) < max(1, min_samples):
        return True
    update_min_ratio = float(cfg.get("video_quality.adaptive_blur_update_min_ratio", 0.75))
    return sharpness >= max(floor, threshold * update_min_ratio)


def _rolling_shutter_proxy(current_small: np.ndarray, previous_small: np.ndarray) -> float:
    row_diff = np.mean(np.abs(current_small - previous_small), axis=1)
    mean = float(row_diff.mean())
    if mean <= 1e-6:
        return 0.0
    return float(max(0.0, min(1.0, (float(row_diff.std()) / mean) / 3.0)))


def _jitter_proxy(*, motion_score: float, novelty_score: float) -> float:
    return float(max(0.0, min(1.0, motion_score - novelty_score)))


def _reject_reasons(**kwargs: object) -> list[str]:
    cfg = kwargs["cfg"]
    assert isinstance(cfg, PipelineConfig)
    reasons: list[str] = []
    if float(kwargs["sharpness"]) < float(kwargs["adaptive_blur_threshold"]):
        reasons.append("blur")
    if float(kwargs["exposure_jump"]) > float(cfg.require("video_quality.max_exposure_jump")):
        reasons.append("exposure_jump")
    if float(kwargs["duplicate_similarity"]) > float(cfg.require("video_quality.max_duplicate_similarity")):
        reasons.append("duplicate")
    motion = float(kwargs["motion_score"])
    if motion < float(cfg.require("video_quality.min_motion_score")):
        reasons.append("low_motion")
    if motion > float(cfg.require("video_quality.max_motion_score")):
        reasons.append("excessive_motion")
    if bool(cfg.get("video_quality.reject_low_feature_density", False)):
        if float(kwargs["feature_density_score"]) < float(cfg.get("video_quality.min_feature_density_score", 0.0)):
            reasons.append("low_feature_density")
    if bool(cfg.get("video_quality.reject_rolling_shutter_warning", False)):
        if float(kwargs["rolling_shutter_score"]) > float(cfg.get("video_quality.rolling_shutter_warning_threshold", 0.65)):
            reasons.append("rolling_shutter")
    if bool(cfg.get("video_quality.reject_jitter_warning", False)):
        if float(kwargs["jitter_score"]) > float(cfg.get("video_quality.jitter_warning_threshold", 0.35)):
            reasons.append("jitter")
    return reasons


def _warning_reasons(**kwargs: object) -> list[str]:
    cfg = kwargs["cfg"]
    assert isinstance(cfg, PipelineConfig)
    warnings: list[str] = []
    if float(kwargs["rolling_shutter_score"]) > float(cfg.get("video_quality.rolling_shutter_warning_threshold", 0.65)):
        warnings.append("rolling_shutter_warning")
    if float(kwargs["jitter_score"]) > float(cfg.get("video_quality.jitter_warning_threshold", 0.35)):
        warnings.append("jitter_warning")
    return warnings


def _quality_score(
    *,
    cfg: PipelineConfig,
    sharpness: float,
    adaptive_blur_threshold: float,
    exposure_jump: float,
    motion_score: float,
    novelty_score: float,
    feature_density_score: float,
    rolling_shutter_score: float,
    jitter_score: float,
) -> float:
    weights = cfg.require("video_quality.quality_weights")
    sharpness_component = max(0.0, min(1.0, sharpness / max(adaptive_blur_threshold, 1e-6)))
    exposure_component = max(0.0, 1.0 - exposure_jump / max(float(cfg.require("video_quality.max_exposure_jump")), 1e-6))
    motion_component = _motion_component(
        motion_score=motion_score,
        min_motion=float(cfg.require("video_quality.min_motion_score")),
        max_motion=float(cfg.require("video_quality.max_motion_score")),
    )
    score = (
        float(weights["sharpness"]) * sharpness_component
        + float(weights["exposure_stability"]) * exposure_component
        + float(weights["motion"]) * motion_component
        + float(weights["novelty"]) * novelty_score
        + float(weights["feature_density"]) * feature_density_score
    )
    score -= 0.15 * rolling_shutter_score + 0.20 * jitter_score
    return float(max(0.0, min(1.0, score)))


def _motion_component(*, motion_score: float, min_motion: float, max_motion: float) -> float:
    if motion_score < min_motion:
        return max(0.0, motion_score / max(min_motion, 1e-6))
    if motion_score > max_motion:
        return max(0.0, 1.0 - (motion_score - max_motion) / max(1e-6, 1.0 - max_motion))
    return 1.0
