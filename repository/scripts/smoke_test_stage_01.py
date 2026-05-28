#!/usr/bin/env python3
"""Smoke test for Stage 1 using a synthetic tiny video."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import yaml

from pipeline.common.config import load_config
from pipeline.stage_01_ingest.run_ingest import run_ingest


def main() -> int:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("STAGE_01_SMOKE_FAILED: ffmpeg and ffprobe must be on PATH", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="stage01_smoke_") as tmp:
        root = Path(tmp)
        raw_dir = root / "data" / "raw"
        raw_dir.mkdir(parents=True)
        video_path = raw_dir / "site01.mp4"
        _write_synthetic_video(video_path)
        cfg_path = root / "configs" / "site01.yaml"
        cfg_path.parent.mkdir(parents=True)
        cfg_path.write_text(yaml.safe_dump(_config(root), sort_keys=False), encoding="utf-8")
        cfg = load_config(cfg_path)
        run_ingest(cfg, force=True, log_level="WARNING")
        expected = [
            root / "data" / "normalized" / "site01_normalized.mp4",
            root / "data" / "normalized" / "site01_metadata.json",
            root / "data" / "normalized" / "site01_frame_quality.csv",
            root / "runs" / "smoke_stage01" / "reports" / "stage_01_ingest_report.json",
        ]
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            print(f"STAGE_01_SMOKE_FAILED: missing outputs: {missing}", file=sys.stderr)
            return 2
    print("STAGE_01_SMOKE_OK")
    return 0


def _write_synthetic_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create synthetic mp4 video.")
    try:
        for idx in range(30):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.rectangle(frame, (10 + idx, 20), (70 + idx, 80), (255, 255, 255), -1)
            cv2.line(frame, (0, idx * 3 % 120), (159, (idx * 3 + 40) % 120), (80, 180, 255), 2)
            writer.write(frame)
    finally:
        writer.release()


def _config(root: Path) -> dict:
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "site01.yaml").read_text(encoding="utf-8"))
    cfg["project"]["root"] = str(root)
    cfg["project"]["run_id"] = "smoke_stage01"
    cfg["inputs"]["video"] = "data/raw/site01.mp4"
    cfg["video"]["normalize_fps"] = 10
    cfg["video"]["sample_fps_for_quality"] = 2
    cfg["video"]["crf"] = 23
    cfg["video"]["preset"] = "veryfast"
    cfg["video"]["skip_reencode_if_compliant"] = False
    cfg["video"]["cfr_option"] = "vsync"
    cfg["video"]["verify_cfr_after_normalization"] = True
    cfg["video"]["cfr_tolerance_fps"] = 0.05
    cfg["video_quality"]["min_blur_laplacian"] = 10.0
    cfg["video_quality"]["adaptive_blur_floor"] = 5.0
    cfg["video_quality"]["adaptive_blur_min_window_samples"] = 3
    cfg["video_quality"]["adaptive_blur_window"] = 5
    cfg["video_quality"]["max_duplicate_similarity"] = 0.995
    cfg["video_quality"]["max_exposure_jump"] = 0.60
    cfg["video_quality"]["min_motion_score"] = 0.0
    cfg["video_quality"]["max_motion_score"] = 0.95
    cfg["video_quality"]["fast_seek"] = False
    cfg["video_quality"]["histogram_bins"] = 16
    cfg["video_quality"]["feature_max_keypoints"] = 500
    cfg["video_quality"]["target_feature_count"] = 50
    cfg["video_quality"]["scoring_max_width"] = 1280
    cfg["video_quality"]["scoring_max_height"] = 720
    return cfg


if __name__ == "__main__":
    raise SystemExit(main())
