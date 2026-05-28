from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest
import yaml

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from pipeline.common.config import load_config
from pipeline.stage_01_ingest.frame_quality import QUALITY_COLUMNS, compute_frame_quality_table
from pipeline.stage_01_ingest.run_ingest import run_ingest


@pytest.fixture()
def tiny_video(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (96, 72))
    assert writer.isOpened()
    for idx in range(16):
        frame = np.zeros((72, 96, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + idx * 2, 36), 12, (255, 255, 255), -1)
        cv2.line(frame, (0, idx * 4 % 72), (95, (idx * 4 + 20) % 72), (120, 200, 255), 2)
        writer.write(frame)
    writer.release()
    return path


def test_frame_quality_contract_columns(tmp_path: Path, tiny_video: Path) -> None:
    cfg_path = _write_config(tmp_path, tiny_video)
    cfg = load_config(cfg_path)
    quality_csv = tmp_path / "quality.csv"
    summary = compute_frame_quality_table(tiny_video, quality_csv, cfg, logger=_NullLogger())
    assert summary.sampled_frame_count > 0
    with quality_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(QUALITY_COLUMNS)
        rows = list(reader)
    assert rows
    assert "feature_density_score" in rows[0]
    assert "histogram_similarity" in rows[0]
    assert "warning_reason" in rows[0]
    assert "scoring_width" in rows[0]


def test_stage_01_integration_smoke(tmp_path: Path, tiny_video: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not installed in this environment")
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    raw_video = raw_dir / "site01.mp4"
    shutil.copy2(tiny_video, raw_video)
    cfg_path = _write_config(tmp_path, raw_video)
    cfg = load_config(cfg_path)
    report = run_ingest(cfg, force=True, log_level="ERROR")
    assert Path(report["outputs"]["normalized_video"]).exists()
    assert Path(report["outputs"]["metadata_json"]).exists()
    assert Path(report["outputs"]["quality_csv"]).exists()
    assert report["quality"]["sampled_frame_count"] > 0


def _write_config(root: Path, video_path: Path) -> Path:
    relative_video = video_path.relative_to(root) if video_path.is_relative_to(root) else video_path
    cfg = yaml.safe_load(Path("configs/site01.yaml").read_text(encoding="utf-8"))
    cfg["project"]["root"] = str(root)
    cfg["project"]["run_id"] = "pytest_stage01"
    cfg["inputs"]["video"] = str(relative_video)
    cfg["video"]["normalize_fps"] = 8
    cfg["video"]["sample_fps_for_quality"] = 2
    cfg["video"]["preset"] = "veryfast"
    cfg["video"]["skip_reencode_if_compliant"] = False
    cfg["video_quality"]["min_blur_laplacian"] = 5.0
    cfg["video_quality"]["adaptive_blur_floor"] = 2.0
    cfg["video_quality"]["max_duplicate_similarity"] = 0.999
    cfg["video_quality"]["max_exposure_jump"] = 0.90
    cfg["video_quality"]["min_motion_score"] = 0.0
    cfg["video_quality"]["max_motion_score"] = 0.99
    cfg["video_quality"]["target_feature_count"] = 20
    cfg["video_quality"]["fast_seek"] = False
    cfg["video_quality"]["scoring_max_width"] = 1280
    cfg["video_quality"]["scoring_max_height"] = 720
    cfg["video_quality"]["adaptive_blur_min_window_samples"] = 3
    cfg["video"]["cfr_option"] = "vsync"
    cfg["video"]["verify_cfr_after_normalization"] = True
    cfg["video"]["cfr_tolerance_fps"] = 0.05
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


class _NullLogger:
    def info(self, *args, **kwargs) -> None:
        return None
    def warning(self, *args, **kwargs) -> None:
        return None
    def debug(self, *args, **kwargs) -> None:
        return None


def test_cosine_similarity_is_clipped() -> None:
    from pipeline.stage_01_ingest.frame_quality import _cosine_similarity
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0 + 1e-7, 0.0, 0.0], dtype=np.float32)
    value = _cosine_similarity(a, b)
    assert -1.0 <= value <= 1.0


def test_adaptive_blur_window_ignores_severe_blur(tmp_path: Path, tiny_video: Path) -> None:
    from collections import deque
    from pipeline.stage_01_ingest.frame_quality import _adaptive_blur_threshold, _should_update_sharpness_window
    cfg_path = _write_config(tmp_path, tiny_video)
    cfg = load_config(cfg_path)
    window = deque([100.0, 105.0, 110.0], maxlen=5)
    threshold_before = _adaptive_blur_threshold(cfg, window)
    assert not _should_update_sharpness_window(sharpness=1.0, threshold=threshold_before, cfg=cfg, window=window)
    assert _should_update_sharpness_window(sharpness=threshold_before, threshold=threshold_before, cfg=cfg, window=window)


def test_ffmpeg_command_enforces_cfr(tmp_path: Path, tiny_video: Path) -> None:
    from pipeline.stage_01_ingest.normalize_video import _build_ffmpeg_command
    cfg_path = _write_config(tmp_path, tiny_video)
    cfg = load_config(cfg_path)
    command = _build_ffmpeg_command(tiny_video, tmp_path / "out.mp4", cfg, "libx264")
    assert "-vf" in command
    vf_index = command.index("-vf")
    expected_fps = float(cfg.require("video.normalize_fps"))
    assert command[vf_index + 1] == f"fps={expected_fps:g}"
    has_fps_mode = "-fps_mode" in command and "cfr" in command
    has_legacy_vsync = "-vsync" in command and "1" in command
    assert has_fps_mode or has_legacy_vsync


def test_adaptive_blur_window_warmup_accepts_moderate_blur(tmp_path: Path, tiny_video: Path) -> None:
    from collections import deque
    from pipeline.stage_01_ingest.frame_quality import _should_update_sharpness_window
    cfg_path = _write_config(tmp_path, tiny_video)
    cfg = load_config(cfg_path)
    window = deque([], maxlen=5)
    assert _should_update_sharpness_window(sharpness=3.0, threshold=80.0, cfg=cfg, window=window)


def test_cosine_similarity_zero_vector_handling() -> None:
    from pipeline.stage_01_ingest.frame_quality import _cosine_similarity
    zero = np.zeros(3, dtype=np.float32)
    one = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert _cosine_similarity(zero, one) == 0.0
    assert _cosine_similarity(zero, zero) == 1.0
