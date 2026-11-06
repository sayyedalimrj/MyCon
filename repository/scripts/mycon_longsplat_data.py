#!/usr/bin/env python3
"""Prepare a separate, read-only-source LongSplat candidate image dataset.

This stage EXTRACTS FRAMES ONLY; it never installs MASt3R/CUDA, trains, deletes
Recorder sidecars, merges models, alters prior NVIDIA checkpoints or claims BIM
metric scale. LongSplat custom-video inputs use a ./data/SCENE/images directory.
"""
from __future__ import annotations

import json
from pathlib import Path

from mycon_audit_runtime import atomic_json, quarantine_derived, sha256
from mycon_colmap_gsplat_env import run_logged

FPS = 8
WIDTH = 512
FRAME_LIMIT = 1200


def _manifest(images: Path, expected_duration_s: float) -> dict:
    frames = sorted(images.glob("frame_*.jpg"))
    expected = round(expected_duration_s * FPS)
    if not 50 <= expected <= FRAME_LIMIT:
        raise RuntimeError("MYCON_LONGSPLAT_SOURCE_DURATION_INVALID")
    if (abs(len(frames) - expected) > max(8, round(expected * .03))
        or len(frames) > FRAME_LIMIT):
        raise RuntimeError(f"MYCON_LONGSPLAT_8FPS_FRAME_COUNT_INVALID:"
                           f" got={len(frames)} expected={expected}")
    files = {}
    for number, path in enumerate(frames, 1):
        if path.name != f"frame_{number:06d}.jpg" or path.stat().st_size < 1024:
            raise RuntimeError("MYCON_LONGSPLAT_FRAME_SEQUENCE_INCOMPLETE:" +
                               path.name)
        files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size,
                            "nominal_time_s": round((number - 1) / FPS, 3)}
    return {"frame_count": len(frames), "files": files}


def prepare_longsplat_frames(video: Path, dataset: Path, log: Path, *,
                             capture_sha256: str, duration_s: float) -> dict:
    """Non-destructive, content-verified frames with safe interrupted replay."""
    video, dataset, log = Path(video), Path(dataset), Path(log)
    if not video.is_file() or video.stat().st_size < 1024:
        raise RuntimeError("MYCON_VERIFIED_LOCAL_RECORDER_VIDEO_REQUIRED")
    if dataset == video or dataset in video.parents or video in dataset.parents:
        raise RuntimeError("MYCON_LONGSPLAT_SOURCE_DESTINATION_OVERLAP")
    actual_video_hash = sha256(video)
    receipt = dataset / "dataset_receipt.json"
    if receipt.is_file():
        row = json.loads(receipt.read_text())
        if (row.get("source_video_sha256") != actual_video_hash
            or row.get("source_recorder_archive_sha256") != capture_sha256
            or row.get("fps") != FPS or row.get("width") != WIDTH
            or abs(float(row.get("source_duration_s", -1)) - duration_s) > .01):
            raise RuntimeError("MYCON_LONGSPLAT_EXISTING_DATASET_SOURCE_CHANGED")
        observed = _manifest(dataset / "images", duration_s)
        if observed != row.get("manifest"):
            raise RuntimeError("MYCON_LONGSPLAT_FRAME_HASH_INTEGRITY_FAILED")
        return {"status": "LONGSPLAT_8FPS_DATASET_VERIFIED_NO_TRAINING",
                "dataset_images": str(dataset / "images"),
                "frame_count": observed["frame_count"],
                "dataset_receipt": str(receipt),
                "full_gpu_training_executed": False}
    if dataset.exists():
        # An interrupted, DERIVED extraction is kept for diagnosis, never
        # recursively deleted; this function does not touch source archives.
        quarantine_derived(dataset, "partial_longsplat_8fps")
    images = dataset / "images"
    images.mkdir(parents=True, exist_ok=False)
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
           "-noautorotate", "-i", str(video),
           "-vf", f"transpose=clock,fps={FPS},scale={WIDTH}:-2:flags=lanczos",
           "-q:v", "2", str(images / "frame_%06d.jpg")]
    run_logged(cmd, log, timeout_s=3600, idle_timeout_s=1200)
    manifest = _manifest(images, duration_s)
    row = {
        "schema": "mycon.longsplat.custom_video_derived_frames.v1",
        "source_recorder_archive_sha256": capture_sha256,
        "source_video_sha256": actual_video_hash,
        "source_duration_s": duration_s,
        "fps": FPS, "width": WIDTH,
        "rotation_applied_to_pixels": "CW_90_ONCE_WITH_NOAUTOROTATE",
        "native_pose_csv_modified": False,
        "timestamps": "NOMINAL_FFMPEG_RESAMPLED_NOT_RECORDER_POSE_TIMESTAMP",
        "original_recorder_and_checkpoints_mutated": False,
        "training_executed": False,
        "metric_scale_verified": False,
        "data_use": "OPTIONAL_NVIDIA_LONGSPLAT_INPUT_ONLY_NOT_COLMAP_BASELINE",
        "manifest": manifest,
    }
    atomic_json(receipt, row)
    return {"status": "LONGSPLAT_8FPS_DATASET_PREPARED_NO_TRAINING",
            "dataset_images": str(images), "frame_count": manifest["frame_count"],
            "dataset_receipt": str(receipt),
            "full_gpu_training_executed": False}
