"""CPU-only artifact and protection tests for Geometry-A."""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.stage_04_5_cams_gs.research_gsplat_audit import (
    inspect_gsplat_artifacts, run_audit,
)
from pipeline.stage_04_5_cams_gs.run_cams_gs_prepare import (
    Stage45CamsGsError, run_cams_gs_prepare,
)


def test_gsplat_inspection_observes_existing_files_only(tmp_path: Path) -> None:
    originals = tmp_path / "originals"
    originals.mkdir()
    ckpt, ply, log, video = (
        originals / "ckpt_6999_rank0.pt", originals / "point_cloud_6999.ply",
        originals / "final_canonical_eval.log", originals / "traj_6999.mp4",
    )
    ckpt.write_bytes(b"fake_ckpt")
    ply.write_bytes(b"fake_ply")
    video.write_bytes(b"fake_video")
    log.write_text(
        "[Parser] 192 images, taken by 1 cameras.\n"
        "PSNR: 20.736, SSIM: 0.6176, LPIPS: 0.522 Time: 0.085s/image Number of GS: 1477725\n"
        "Rendering trajectory: 100%|████| 181/181 [00:23]\n"
        "Video saved to /somewhere/traj_6999.mp4\n",
        encoding="utf-8",
    )
    before = {p.name: p.read_bytes() for p in originals.iterdir()}
    result = run_audit(project_root=tmp_path, run_id="geom-audit-test", checkpoint=ckpt,
                       splat_ply=ply, evaluation_log=log, video=video)
    assert result["reported_from_log_only"]["psnr_db"] == pytest.approx(20.736)
    assert result["reported_from_log_only"]["ssim"] == pytest.approx(.6176)
    assert result["trajectory_frames_reported"] == (181, 181)
    assert not result["checkpoint_content_verified"]
    assert not result["novel_views_independent_of_train_verified"]
    assert {p.name: p.read_bytes() for p in originals.iterdir()} == before
    assert not (tmp_path / "runs").exists()
    written = run_audit(project_root=tmp_path, run_id="geom-audit-test",
                        checkpoint=ckpt, splat_ply=ply, evaluation_log=log,
                        video=video, write=True)
    assert Path(written["audit_path"]).exists()
    with pytest.raises(FileExistsError):
        run_audit(project_root=tmp_path, run_id="geom-audit-test",
                  checkpoint=ckpt, splat_ply=ply, evaluation_log=log,
                  video=video, write=True)


def test_missing_render_is_not_evidence(tmp_path: Path) -> None:
    for name in ("x.pt", "y.ply", "e.log"):
        (tmp_path / name).write_bytes(b"x")
    result = inspect_gsplat_artifacts(
        tmp_path / "x.pt", tmp_path / "y.ply", tmp_path / "e.log"
    )
    assert result["video_file_verified_nonempty"] is False
    assert result["reported_from_log_only"] is None
    with pytest.raises(FileNotFoundError):
        inspect_gsplat_artifacts(tmp_path / "missing", tmp_path / "y.ply", tmp_path / "e.log")


def test_stage45_force_cannot_erase_original_artifacts(tmp_path: Path) -> None:
    existing = tmp_path / "data/cams_gs/site01"
    existing.mkdir(parents=True)
    important = existing / "original_checkpoint.pt"
    important.write_bytes(b"retain original")
    cfg = {
        "project": {"root": str(tmp_path), "run_id": "geom-test", "name": "site01"},
        "cams_gs": {"output_dir": "data/cams_gs/site01", "execute_training": False},
    }
    with pytest.raises(Stage45CamsGsError, match="Refusing destructive"):
        run_cams_gs_prepare(cfg, force=True, log_level="ERROR")
    assert important.read_bytes() == b"retain original"


pytestmark = pytest.mark.geometry
