#!/usr/bin/env python3
"""Smoke test for Stage 2 adaptive keyframe selection."""

from __future__ import annotations

import csv
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
from pipeline.stage_02_keyframes.select_keyframes import run_keyframe_selection


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage02_smoke_") as tmp:
        root = Path(tmp)
        _make_dirs(root)
        video_path = root / "data" / "normalized" / "site01_normalized.mp4"
        quality_csv = root / "data" / "normalized" / "site01_frame_quality.csv"
        _make_video(video_path)
        _make_quality_csv(quality_csv)
        cfg_path = _write_config(root)
        cfg = load_config(cfg_path)
        report = run_keyframe_selection(cfg, force=True, log_level="WARNING")
        manifest = Path(report["outputs"]["manifest_csv"])
        contact_sheet = Path(report["outputs"]["contact_sheet"])
        keyframes_dir = Path(report["outputs"]["keyframes_dir"])
        if not manifest.exists():
            raise SystemExit("STAGE_02_SMOKE_FAILED: manifest missing")
        if not contact_sheet.exists():
            raise SystemExit("STAGE_02_SMOKE_FAILED: contact sheet missing")
        jpgs = sorted(keyframes_dir.glob("*.jpg"))
        if not jpgs:
            raise SystemExit("STAGE_02_SMOKE_FAILED: no JPG keyframes")
        print(f"STAGE_02_SMOKE_OK keyframes={len(jpgs)} manifest={manifest}")
        return 0


def _make_dirs(root: Path) -> None:
    for relative in [
        "data/normalized", "data/frames/key", "data/raw", "data/bim/design", "data/bim/aligned", "data/bim/metrics",
        "data/clean", "data/da3", "data/dense", "data/sfm", "data/sparse", "data/sparse_refined", "exports",
    ]:
        (root / relative).mkdir(parents=True, exist_ok=True)


def _make_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 96))
    if not writer.isOpened():
        raise RuntimeError("Could not create synthetic video")
    for idx in range(40):
        frame = np.zeros((96, 128, 3), dtype=np.uint8)
        cv2.rectangle(frame, (5 + idx, 20), (45 + idx, 70), (80, 180, 255), -1)
        cv2.circle(frame, (90, 30 + idx % 30), 12, (255, 255, 255), -1)
        cv2.line(frame, (0, (idx * 3) % 96), (127, (idx * 3 + 25) % 96), (200, 200, 50), 2)
        writer.write(frame)
    writer.release()


def _make_quality_csv(path: Path) -> None:
    fieldnames = [
        "frame_index", "timestamp_sec", "sharpness_laplacian", "exposure_mean", "exposure_std", "exposure_jump",
        "motion_score", "duplicate_similarity", "novelty_score", "quality_score", "reject_reason", "warning_reason",
        "feature_count", "feature_density_score", "histogram_similarity", "rolling_shutter_score", "jitter_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_idx, frame_idx in enumerate(range(0, 40, 2)):
            bad_duplicate = sample_idx in {5, 6}
            writer.writerow(
                {
                    "frame_index": frame_idx,
                    "timestamp_sec": frame_idx / 10.0,
                    "sharpness_laplacian": 120.0,
                    "exposure_mean": 120.0,
                    "exposure_std": 25.0,
                    "exposure_jump": 0.03,
                    "motion_score": 0.20,
                    "duplicate_similarity": 0.99 if bad_duplicate else 0.50,
                    "novelty_score": 0.20 + sample_idx * 0.02,
                    "quality_score": 0.75,
                    "reject_reason": "duplicate" if bad_duplicate else "",
                    "warning_reason": "",
                    "feature_count": 80,
                    "feature_density_score": 0.65,
                    "histogram_similarity": 0.50,
                    "rolling_shutter_score": 0.10,
                    "jitter_score": 0.10,
                }
            )


def _write_config(root: Path) -> Path:
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "site01.yaml").read_text(encoding="utf-8"))
    cfg["project"]["root"] = str(root)
    cfg["project"]["run_id"] = "stage02_smoke"
    cfg["keyframes"]["max_frames_first_run"] = 8
    cfg["keyframes"]["fallback_min_keyframes"] = 3
    cfg["keyframes"]["min_time_gap_sec"] = 0.4
    cfg["keyframes"]["min_segment_duration_sec"] = 0.5
    cfg["keyframes"]["min_segment_frames"] = 2
    cfg["keyframes"]["contact_sheet_max_images"] = 8
    cfg["keyframes"]["contact_sheet_thumb_width"] = 120
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


if __name__ == "__main__":
    raise SystemExit(main())
