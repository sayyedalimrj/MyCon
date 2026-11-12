"""CPU-only tests: no model inference, GPU training or checkpoint rewriting."""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.stage_04_5_cams_gs.research_training import (
    GeometryTrainingError, command_for_resume, run_resume, validate_inputs,
)
from pipeline.stage_04_5_cams_gs.research_backends import (
    colmap_contract, prepare_backend_command, run_backend,
)


def _inputs(root: Path) -> tuple[Path, Path, Path]:
    images = root / "original_images"
    sparse = root / "original_sparse"
    ckpts = root / "original_checkpoints"
    images.mkdir()
    sparse.mkdir()
    ckpts.mkdir()
    for name in ("a.jpg", "b.jpg"):
        (images / name).write_bytes(b"no_real_image_data")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).write_bytes(b"fake_model")
    (ckpts / "step-000001.ckpt").write_bytes(b"fake_checkpoint")
    return images, sparse, ckpts


def test_resume_inspection_is_read_only(tmp_path: Path) -> None:
    images, sparse, ckpts = _inputs(tmp_path)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    result = run_resume(project_root=tmp_path, run_id="geom-probe-01",
                        images_dir=images, sparse_dir=sparse, resume_dir=ckpts)
    assert result["status"] == "inspected_only"
    assert result["novel_view_validated"] is False
    assert result["input_evidence"]["image_count"] == 2
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def test_checkpoint_must_be_nerfstudio_and_nonempty(tmp_path: Path) -> None:
    images, sparse, ckpts = _inputs(tmp_path)
    (ckpts / "step-000001.ckpt").unlink()
    with pytest.raises(GeometryTrainingError, match="Resume-only"):
        validate_inputs(images, sparse, ckpts)
    with pytest.raises(GeometryTrainingError, match="Run ID"):
        run_resume(project_root=tmp_path, run_id="../unsafe", images_dir=images,
                   sparse_dir=sparse, resume_dir=ckpts)


def test_command_has_real_colmap_parser_and_resume(tmp_path: Path) -> None:
    command = command_for_resume(tmp_path / "data", tmp_path / "new", tmp_path / "old", 1000)
    assert command[:2] == ["ns-train", "splatfacto"]
    assert command[command.index("--load-dir") + 1] == str(tmp_path / "old")
    assert "colmap" in command
    assert command[command.index("--auto-scale-poses") + 1] == "False"
    with pytest.raises(GeometryTrainingError):
        command_for_resume(tmp_path, tmp_path, tmp_path, 0)


def test_new_run_only_and_no_false_render_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images, sparse, ckpts = _inputs(tmp_path)
    monkeypatch.setattr("pipeline.stage_04_5_cams_gs.research_training.shutil.which",
                        lambda name: "/fake/ns-train" if name == "ns-train" else None)
    def fake_exec(argv, log):
        out = Path(argv[argv.index("--output-dir") + 1]) / "mock" / "nerfstudio_models"
        out.mkdir(parents=True)
        (out.parent / "config.yml").write_text("fake", encoding="utf-8")
        (out / "step-000002.ckpt").write_bytes(b"fake_ckpt")
        log.write_text("mock command only", encoding="utf-8")
        return 0
    monkeypatch.setattr("pipeline.stage_04_5_cams_gs.research_training._run_logged", fake_exec)
    evidence = run_resume(project_root=tmp_path, run_id="geom-mock-01", images_dir=images,
                          sparse_dir=sparse, resume_dir=ckpts, execute=True)
    assert evidence["status"] == "trained_checkpoint_present_render_unverified"
    assert evidence["novel_view_validated"] is False
    assert (tmp_path / "runs/geom-mock-01/geometry/training_evidence.json").exists()
    assert (ckpts / "step-000001.ckpt").read_bytes() == b"fake_checkpoint"
    with pytest.raises(FileExistsError):
        run_resume(project_root=tmp_path, run_id="geom-mock-01", images_dir=images,
                   sparse_dir=sparse, resume_dir=ckpts, execute=True)


def test_backend_inspection_and_missing_export(tmp_path: Path) -> None:
    images, _, _ = _inputs(tmp_path)
    vggt = tmp_path / "vggt"
    vggt.mkdir()
    (vggt / "demo_colmap.py").write_text("# mock", encoding="utf-8")
    plan = run_backend(backend="vggt", images=images, project_root=tmp_path,
                       run_id="geom-vggt-probe", backend_root=vggt)
    assert plan["status"] == "inspected_only"
    assert not (tmp_path / "runs").exists()
    assert colmap_contract(tmp_path / "missing")["valid"] is False
    with pytest.raises(ValueError, match="DUSt3R"):
        prepare_backend_command("dust3r", tmp_path, tmp_path)


pytestmark = pytest.mark.lightweight
