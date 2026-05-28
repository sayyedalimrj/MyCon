from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from pipeline.common.config import load_config
from pipeline.stage_02_keyframes.novelty import limit_keyframes_temporally, mark_dense_subset, normalized_score
from pipeline.stage_02_keyframes.segment_video import assign_segments
from pipeline.stage_02_keyframes.select_keyframes import REQUIRED_MANIFEST_COLUMNS, Stage2Error, run_keyframe_selection


@pytest.fixture()
def tiny_stage2_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    _make_dirs(tmp_path)
    video_path = tmp_path / "data" / "normalized" / "site01_normalized.mp4"
    quality_csv = tmp_path / "data" / "normalized" / "site01_frame_quality.csv"
    _make_video(video_path, fps=10.0, frames=50)
    _make_quality_csv(quality_csv, fps=10.0, frames=50)
    cfg_path = _write_config(tmp_path)
    return cfg_path, video_path, quality_csv


def test_segment_video_groups_valid_subsequences(tiny_stage2_inputs: tuple[Path, Path, Path]) -> None:
    cfg_path, _, quality_csv = tiny_stage2_inputs
    cfg = load_config(cfg_path)
    rows = _read_rows(quality_csv)
    segment_ids, summaries = assign_segments(rows, cfg)
    assert summaries
    assert any(segment_id >= 0 for segment_id in segment_ids)
    assert all(summary.frame_count >= cfg.require("keyframes.min_segment_frames") for summary in summaries)


def test_selection_score_prefers_quality_novelty_and_features(tiny_stage2_inputs: tuple[Path, Path, Path]) -> None:
    cfg_path, _, _ = tiny_stage2_inputs
    cfg = load_config(cfg_path)
    low = {"quality_score": "0.2", "novelty_score": "0.2", "feature_density_score": "0.2"}
    high = {"quality_score": "0.8", "novelty_score": "0.9", "feature_density_score": "0.7"}
    assert normalized_score(high, cfg) > normalized_score(low, cfg)


def test_limit_keyframes_temporally_respects_max_count() -> None:
    from pipeline.stage_02_keyframes.novelty import SelectionCandidate
    candidates = [SelectionCandidate(i, score=float(i % 3), timestamp_sec=float(i), segment_id=0, selection_reason="test") for i in range(20)]
    limited = limit_keyframes_temporally(candidates, 5)
    assert len(limited) == 5
    assert limited == sorted(limited, key=lambda item: item.timestamp_sec)


def test_dense_subset_flags_are_deterministic() -> None:
    assert mark_dense_subset(5, 1.0) == [True, True, True, True, True]
    assert sum(mark_dense_subset(10, 0.3)) == 3
    assert mark_dense_subset(0, 1.0) == []


def test_stage_02_integration_contract(tiny_stage2_inputs: tuple[Path, Path, Path]) -> None:
    cfg_path, _, _ = tiny_stage2_inputs
    cfg = load_config(cfg_path)
    report = run_keyframe_selection(cfg, force=True, log_level="ERROR")
    manifest = Path(report["outputs"]["manifest_csv"])
    contact_sheet = Path(report["outputs"]["contact_sheet"])
    keyframes_dir = Path(report["outputs"]["keyframes_dir"])
    summary = Path(report["outputs"]["report_json"])
    assert manifest.exists()
    assert contact_sheet.exists()
    assert summary.exists()
    jpgs = sorted(keyframes_dir.glob("*.jpg"))
    assert jpgs
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows
    assert set(REQUIRED_MANIFEST_COLUMNS).issubset(set(reader.fieldnames or []))
    assert len(rows) <= int(cfg.require("keyframes.max_frames_first_run"))
    assert len(rows) == len(jpgs)
    timestamps = [float(row["timestamp_sec"]) for row in rows]
    assert timestamps == sorted(timestamps)
    min_gap = float(cfg.require("keyframes.min_time_gap_sec"))
    if len(timestamps) > 1:
        assert min((b - a) for a, b in zip(timestamps, timestamps[1:])) >= min_gap - 1e-6


def test_stage_02_force_required_for_existing_outputs(tiny_stage2_inputs: tuple[Path, Path, Path]) -> None:
    cfg_path, _, _ = tiny_stage2_inputs
    cfg = load_config(cfg_path)
    run_keyframe_selection(cfg, force=True, log_level="ERROR")
    with pytest.raises(RuntimeError, match="--force"):
        run_keyframe_selection(cfg, force=False, log_level="ERROR")


def test_stage_02_detects_timestamp_frame_drift(tiny_stage2_inputs: tuple[Path, Path, Path]) -> None:
    cfg_path, _, quality_csv = tiny_stage2_inputs
    rows = _read_rows(quality_csv)
    with quality_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            row["timestamp_sec"] = f"{float(row['timestamp_sec']) + 5.0:.6f}"
            writer.writerow(row)
    cfg = load_config(cfg_path)
    with pytest.raises(Stage2Error, match="drift"):
        run_keyframe_selection(cfg, force=True, log_level="ERROR")


def test_stage_02_emergency_fallback_avoids_empty_manifest(tiny_stage2_inputs: tuple[Path, Path, Path]) -> None:
    cfg_path, _, quality_csv = tiny_stage2_inputs
    rows = _read_rows(quality_csv)
    with quality_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            row["sharpness_laplacian"] = "1.0"
            row["exposure_jump"] = "0.95"
            row["motion_score"] = "0.99"
            row["duplicate_similarity"] = "0.99"
            row["quality_score"] = "0.05"
            row["novelty_score"] = "0.10"
            row["reject_reason"] = "blur|exposure_jump|duplicate|motion"
            writer.writerow(row)
    cfg = load_config(cfg_path)
    report = run_keyframe_selection(cfg, force=True, log_level="ERROR")
    assert report["selected_keyframe_count"] >= 1
    assert report["emergency_fallback_used"] is True
    manifest = Path(report["outputs"]["manifest_csv"])
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert manifest_rows
    assert any("emergency" in row["selection_reason"] for row in manifest_rows)


def _make_dirs(root: Path) -> None:
    for relative in [
        "data/normalized", "data/frames/key", "data/raw", "data/bim/design", "data/bim/aligned", "data/bim/metrics",
        "data/clean", "data/da3", "data/dense", "data/sfm", "data/sparse", "data/sparse_refined", "exports",
    ]:
        (root / relative).mkdir(parents=True, exist_ok=True)


def _make_video(path: Path, *, fps: float, frames: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (128, 96))
    assert writer.isOpened()
    for idx in range(frames):
        frame = np.zeros((96, 128, 3), dtype=np.uint8)
        cv2.rectangle(frame, (5 + idx % 70, 18), (45 + idx % 70, 72), (80, 180, 255), -1)
        cv2.circle(frame, (88, 25 + idx % 45), 12, (255, 255, 255), -1)
        cv2.line(frame, (0, (idx * 4) % 96), (127, (idx * 4 + 25) % 96), (200, 200, 50), 2)
        writer.write(frame)
    writer.release()


def _make_quality_csv(path: Path, *, fps: float, frames: int) -> None:
    fieldnames = [
        "frame_index", "timestamp_sec", "sharpness_laplacian", "exposure_mean", "exposure_std", "exposure_jump",
        "motion_score", "duplicate_similarity", "novelty_score", "quality_score", "reject_reason", "warning_reason",
        "feature_count", "feature_density_score", "histogram_similarity", "rolling_shutter_score", "jitter_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_idx, frame_idx in enumerate(range(0, frames, 2)):
            # Insert a short invalid duplicate island to test segmentation.
            duplicate = sample_idx in {6, 7}
            blur = sample_idx == 15
            writer.writerow(
                {
                    "frame_index": frame_idx,
                    "timestamp_sec": frame_idx / fps,
                    "sharpness_laplacian": 3.0 if blur else 120.0,
                    "exposure_mean": 120.0,
                    "exposure_std": 20.0,
                    "exposure_jump": 0.02,
                    "motion_score": 0.20,
                    "duplicate_similarity": 0.99 if duplicate else 0.45,
                    "novelty_score": min(1.0, 0.25 + sample_idx * 0.03),
                    "quality_score": 0.80,
                    "reject_reason": "duplicate" if duplicate else ("blur" if blur else ""),
                    "warning_reason": "",
                    "feature_count": 80,
                    "feature_density_score": 0.65,
                    "histogram_similarity": 0.50,
                    "rolling_shutter_score": 0.10,
                    "jitter_score": 0.10,
                }
            )


def _write_config(root: Path) -> Path:
    cfg = yaml.safe_load(Path("configs/site01.yaml").read_text(encoding="utf-8"))
    cfg["project"]["root"] = str(root)
    cfg["project"]["run_id"] = "pytest_stage02"
    cfg["keyframes"]["max_frames_first_run"] = 10
    cfg["keyframes"]["fallback_min_keyframes"] = 4
    cfg["keyframes"]["min_time_gap_sec"] = 0.4
    cfg["keyframes"]["min_segment_duration_sec"] = 0.5
    cfg["keyframes"]["min_segment_frames"] = 2
    cfg["keyframes"]["contact_sheet_max_images"] = 10
    cfg["keyframes"]["contact_sheet_thumb_width"] = 120
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
