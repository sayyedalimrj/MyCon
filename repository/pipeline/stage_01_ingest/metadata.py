"""Video metadata extraction for Stage 1."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2

from pipeline.common.paths import write_json_atomic
from pipeline.stage_01_ingest.normalize_video import ffprobe_json, _parse_fps


class MetadataError(RuntimeError):
    """Raised when metadata extraction fails."""


def extract_video_metadata(video_path: Path, *, logger: logging.Logger | None = None) -> dict[str, Any]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    probe = ffprobe_json(video_path)
    stream = _first_video_stream(probe) or {}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise MetadataError(f"OpenCV could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    finally:
        capture.release()

    if fps <= 0:
        fps = _parse_fps(str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"))
    if width <= 0:
        width = _safe_int(stream.get("width")) or 0
    if height <= 0:
        height = _safe_int(stream.get("height")) or 0
    if frame_count <= 0:
        frame_count = _safe_int(stream.get("nb_frames")) or 0

    format_duration = _safe_float((probe.get("format") or {}).get("duration"))
    if fps > 0 and frame_count > 0:
        duration = frame_count / fps
    elif format_duration is not None:
        duration = format_duration
    else:
        duration = 0.0

    if fps <= 0:
        raise MetadataError(f"Could not determine a positive FPS for normalized video: {video_path}")

    fourcc = _fourcc_to_string(fourcc_int)
    rotation = _extract_rotation(stream)
    metadata = {
        "video_path": str(video_path),
        "duration_sec": float(duration),
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "codec": stream.get("codec_name") or fourcc or None,
        "opencv_fourcc": fourcc,
        "pix_fmt": stream.get("pix_fmt"),
        "rotation_degrees": rotation,
        "ffprobe": {
            "format_name": (probe.get("format") or {}).get("format_name"),
            "format_duration_sec": format_duration,
            "format_size_bytes": _safe_int((probe.get("format") or {}).get("size")),
            "video_stream": {
                "codec_name": stream.get("codec_name"),
                "profile": stream.get("profile"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "avg_frame_rate": stream.get("avg_frame_rate"),
                "r_frame_rate": stream.get("r_frame_rate"),
                "pix_fmt": stream.get("pix_fmt"),
                "rotation_degrees": rotation,
            },
        },
    }
    if logger is not None:
        logger.info("Metadata: fps=%.3f size=%dx%d frames=%d duration=%.3fs", fps, width, height, frame_count, duration)
    return metadata


def write_metadata_json(metadata: dict[str, Any], output_path: Path) -> None:
    write_json_atomic(output_path, metadata)


def _first_video_stream(probe: dict[str, Any]) -> dict[str, Any] | None:
    for stream in probe.get("streams", []):
        if isinstance(stream, dict) and stream.get("codec_type") == "video":
            return stream
    return None


def _extract_rotation(stream: dict[str, Any]) -> float | None:
    tags = stream.get("tags")
    if isinstance(tags, dict) and "rotate" in tags:
        return _safe_float(tags.get("rotate"))
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if not isinstance(item, dict):
                continue
            if "rotation" in item:
                return _safe_float(item.get("rotation"))
    return None


def _fourcc_to_string(value: int) -> str:
    if value <= 0:
        return ""
    return "".join(chr((value >> 8 * i) & 0xFF) for i in range(4)).strip()


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return None if value is None else int(float(value))
    except (TypeError, ValueError):
        return None
