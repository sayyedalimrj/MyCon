"""FFmpeg-based video normalization with CFR enforcement and atomic writes.

The Stage 1 baseline favors reliability across FFmpeg versions over newer-only
options. The default CFR mode is the widely available legacy ``-vsync 1`` plus
an explicit ``fps=...`` filter. If a deployment is known to support it, the YAML
can select ``video.cfr_option: fps_mode`` to use ``-fps_mode cfr`` instead.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pipeline.common.config import PipelineConfig
from pipeline.common.paths import atomic_output_path


FFPROBE_TIMEOUT_SEC = 60
FFMPEG_TIMEOUT_SEC = 21600


@dataclass(frozen=True)
class NormalizationResult:
    input_video: str
    output_video: str
    normalized: bool
    skipped_reencode: bool
    ffmpeg_command: list[str]
    ffprobe_before: dict[str, Any]
    ffprobe_after: dict[str, Any]


class VideoNormalizationError(RuntimeError):
    """Raised when ffmpeg or ffprobe fails."""


def normalize_video(
    input_video: Path,
    output_video: Path,
    cfg: PipelineConfig,
    *,
    logger: logging.Logger,
    force: bool = False,
) -> NormalizationResult:
    """Normalize a mobile video into the Stage 1 video contract.

    Mobile videos are often variable-frame-rate even when ffprobe reports an
    average FPS close to the target. The default config therefore re-encodes to
    constant-frame-rate (CFR) instead of copying apparently compliant files.
    """
    if not input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    overwrite = bool(cfg.get("video.overwrite", True))
    if output_video.exists() and not overwrite and not force:
        after = ffprobe_json(output_video)
        _assert_normalized_video_contract(after, cfg, output_video)
        return NormalizationResult(str(input_video), str(output_video), True, True, [], ffprobe_json(input_video), after)

    before = ffprobe_json(input_video)
    skip_reencode = bool(cfg.get("video.skip_reencode_if_compliant", False))
    if skip_reencode and not force and _is_compliant(before, cfg):
        logger.warning(
            "skip_reencode_if_compliant=true is not recommended for mobile SfM. "
            "Copying only because the config explicitly requested it."
        )
        with atomic_output_path(output_video) as tmp_path:
            shutil.copy2(input_video, tmp_path)
        after = ffprobe_json(output_video)
        _assert_normalized_video_contract(after, cfg, output_video)
        return NormalizationResult(str(input_video), str(output_video), True, True, [], before, after)

    codec = str(cfg.get("video.codec", "libx264"))
    fallback = str(cfg.get("video.codec_fallback", "libx264"))
    command = _build_ffmpeg_command(input_video, output_video, cfg, codec)
    try:
        after = _run_ffmpeg_atomic(command, output_video, logger)
    except VideoNormalizationError:
        if codec == fallback:
            raise
        logger.warning("ffmpeg failed with codec=%s; retrying with fallback=%s", codec, fallback)
        command = _build_ffmpeg_command(input_video, output_video, cfg, fallback)
        after = _run_ffmpeg_atomic(command, output_video, logger)
    _assert_normalized_video_contract(after, cfg, output_video)
    return NormalizationResult(str(input_video), str(output_video), True, False, command, before, after)


def ffprobe_json(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=FFPROBE_TIMEOUT_SEC)
    except subprocess.TimeoutExpired as exc:
        raise VideoNormalizationError(f"ffprobe timed out for {video_path} after {FFPROBE_TIMEOUT_SEC} seconds") from exc
    if result.returncode != 0:
        raise VideoNormalizationError(
            f"ffprobe failed for {video_path} with exit code {result.returncode}: {result.stderr.strip()}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoNormalizationError(f"ffprobe returned invalid JSON for {video_path}") from exc


def _build_ffmpeg_command(input_video: Path, output_video: Path, cfg: PipelineConfig, codec: str) -> list[str]:
    fps = float(cfg.require("video.normalize_fps"))
    if fps <= 0:
        raise ValueError("video.normalize_fps must be greater than zero")
    pix_fmt = str(cfg.get("video.output_pix_fmt", "yuv420p"))
    loglevel = str(cfg.get("video.ffmpeg_loglevel", "error"))
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", loglevel, "-y", "-i", str(input_video),
        "-map", "0:v:0", "-vf", f"fps={fps:g}",
    ]
    if bool(cfg.get("video.force_constant_frame_rate", True)):
        command.extend(_cfr_output_args(cfg))
    command.extend(["-c:v", codec, "-pix_fmt", pix_fmt])
    if codec in {"libx264", "libx265"}:
        command.extend(["-crf", str(cfg.get("video.crf", 18)), "-preset", str(cfg.get("video.preset", "medium"))])
    if bool(cfg.get("video.preserve_audio", False)):
        command.extend(["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"])
    else:
        command.append("-an")
    if bool(cfg.get("video.clear_rotation_metadata", True)):
        command.extend(["-metadata:s:v:0", "rotate=0"])
    command.extend(["-movflags", "+faststart", str(output_video)])
    return command


def _cfr_output_args(cfg: PipelineConfig) -> list[str]:
    """Return FFmpeg output arguments for constant-frame-rate output.

    ``auto`` intentionally maps to ``-vsync 1`` because the user's current core
    image does not support ``-fps_mode`` and ``ffmpeg -h full`` probing is noisy
    and expensive. Advanced deployments can request ``fps_mode`` explicitly.
    """
    requested = str(cfg.get("video.cfr_option", "vsync")).strip().lower()
    if requested in {"", "auto", "vsync"}:
        return ["-vsync", "1"]
    if requested == "fps_mode":
        return ["-fps_mode", "cfr"]
    raise ValueError("video.cfr_option must be one of: auto, fps_mode, vsync")


def _run_ffmpeg_atomic(command: Sequence[str], output_video: Path, logger: logging.Logger) -> dict[str, Any]:
    with atomic_output_path(output_video) as tmp_path:
        tmp_command = list(command)
        tmp_command[-1] = str(tmp_path)
        logger.info("Running ffmpeg normalization with CFR enforcement")
        logger.debug("Command: %s", " ".join(tmp_command))
        try:
            result = subprocess.run(tmp_command, capture_output=True, text=True, check=False, timeout=FFMPEG_TIMEOUT_SEC)
        except subprocess.TimeoutExpired as exc:
            raise VideoNormalizationError(f"ffmpeg timed out after {FFMPEG_TIMEOUT_SEC} seconds while normalizing {output_video}") from exc
        if result.returncode != 0:
            raise VideoNormalizationError(
                "ffmpeg normalization failed with exit code "
                f"{result.returncode}\nCOMMAND:\n{' '.join(tmp_command)}"
                f"\nSTDOUT:\n{result.stdout.strip()}\nSTDERR:\n{result.stderr.strip()}"
            )
    return ffprobe_json(output_video)


def _assert_normalized_video_contract(probe: dict[str, Any], cfg: PipelineConfig, output_video: Path) -> None:
    if not bool(cfg.get("video.verify_cfr_after_normalization", True)):
        return
    stream = _first_video_stream(probe)
    if stream is None:
        raise VideoNormalizationError(f"Normalized output has no video stream: {output_video}")
    tolerance = float(cfg.get("video.cfr_tolerance_fps", 0.05))
    target_fps = float(cfg.require("video.normalize_fps"))
    avg_fps = _parse_fps(str(stream.get("avg_frame_rate") or "0/1"))
    real_fps = _parse_fps(str(stream.get("r_frame_rate") or "0/1"))
    if avg_fps <= 0 and real_fps <= 0:
        raise VideoNormalizationError(f"Could not verify FPS for normalized output: {output_video}")
    observed = avg_fps if avg_fps > 0 else real_fps
    if abs(observed - target_fps) > tolerance:
        raise VideoNormalizationError(
            f"Normalized output FPS {observed:.6g} differs from target {target_fps:.6g} by more than {tolerance}: {output_video}"
        )
    if avg_fps > 0 and real_fps > 0 and abs(avg_fps - real_fps) > tolerance:
        raise VideoNormalizationError(
            f"Normalized output still appears non-CFR: avg_frame_rate={avg_fps:.6g}, "
            f"r_frame_rate={real_fps:.6g}, file={output_video}"
        )


def _is_compliant(probe: dict[str, Any], cfg: PipelineConfig) -> bool:
    """Return whether a video is safe to copy instead of re-encoding.

    This function is intentionally strict. A file with variable-frame-rate
    indicators is not compliant even if its average FPS equals the target.
    """
    stream = _first_video_stream(probe)
    if stream is None:
        return False
    if _has_variable_frame_rate_indicators(stream, tolerance=float(cfg.get("video.cfr_tolerance_fps", 0.05))):
        return False
    target_fps = float(cfg.require("video.normalize_fps"))
    target_pix_fmt = str(cfg.get("video.output_pix_fmt", "yuv420p"))
    target_codec = str(cfg.get("video.codec", "libx264"))
    codec = str(stream.get("codec_name", ""))
    pix_fmt = str(stream.get("pix_fmt", ""))
    fps = _parse_fps(str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"))
    codec_ok = (target_codec == "libx264" and codec == "h264") or codec == target_codec
    return codec_ok and abs(fps - target_fps) <= 0.02 and pix_fmt == target_pix_fmt


def _has_variable_frame_rate_indicators(stream: dict[str, Any], *, tolerance: float = 0.05) -> bool:
    avg_fps = _parse_fps(str(stream.get("avg_frame_rate") or "0/1"))
    real_fps = _parse_fps(str(stream.get("r_frame_rate") or "0/1"))
    if avg_fps > 0 and real_fps > 0 and abs(avg_fps - real_fps) > tolerance:
        return True
    return False


def _first_video_stream(probe: dict[str, Any]) -> dict[str, Any] | None:
    for stream in probe.get("streams", []):
        if isinstance(stream, dict) and stream.get("codec_type") == "video":
            return stream
    return None


def _parse_fps(value: str) -> float:
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            denominator = float(den)
            return float(num) / denominator if denominator != 0 else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0
