#!/usr/bin/env python3
"""Fast Stage 4 smoke test using a fake COLMAP executable."""
from __future__ import annotations

import stat
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from pipeline.common.config import load_config
from pipeline.stage_04_refinement.run_refinement import run_refinement


def _write_fake_colmap(path: Path) -> None:
    script = r'''#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def arg_value(name: str, default: str | None = None) -> str | None:
    if name not in sys.argv:
        return default
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]

cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

if cmd in {"help", "-h", "--help"}:
    print("COLMAP fake 4.0.3")
    print("bundle_adjuster")
    print("model_converter")
    print("model_analyzer")
    raise SystemExit(0)

if cmd == "bundle_adjuster" and any(arg in {"-h", "--help"} for arg in sys.argv):
    print("  --input_path arg")
    print("  --output_path arg")
    print("  --BundleAdjustmentCeres.max_num_iterations arg (=100)")
    print("  --BundleAdjustmentCeres.max_linear_solver_iterations arg (=200)")
    print("  --BundleAdjustmentCeres.function_tolerance arg (=0)")
    print("  --BundleAdjustmentCeres.gradient_tolerance arg (=0)")
    print("  --BundleAdjustmentCeres.parameter_tolerance arg (=0)")
    print("  --BundleAdjustment.refine_focal_length arg (=1)")
    print("  --BundleAdjustment.refine_principal_point arg (=0)")
    print("  --BundleAdjustment.refine_extra_params arg (=1)")
    raise SystemExit(0)

if cmd == "bundle_adjuster":
    src = Path(arg_value("--input_path", "/tmp/input"))
    out = Path(arg_value("--output_path", "/tmp/output"))
    if not out.exists() or not out.is_dir():
        print("`output_path` is not a directory", file=sys.stderr)
        raise SystemExit(1)
    shutil.rmtree(out)
    shutil.copytree(src, out)
    # Make the output detectably non-empty and stable.
    with (out / "points3D.bin").open("ab") as handle:
        handle.write(b"refined")
    print("Bundle adjustment complete")
    raise SystemExit(0)

if cmd == "model_converter":
    out = Path(arg_value("--output_path", "/tmp/fake_text"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "cameras.txt").write_text("# cameras\n1 SIMPLE_RADIAL 60 40 50 30 20 0.01\n", encoding="utf-8")
    (out / "images.txt").write_text(
        "# images\n"
        "1 1 0 0 0 0 0 0 1 site01_kf_00001.jpg\n\n"
        "2 1 0 0 0 1 0 0 1 site01_kf_00002.jpg\n\n"
        "3 1 0 0 0 2 0 0 1 site01_kf_00003.jpg\n\n"
        "4 1 0 0 0 3 0 0 1 site01_kf_00004.jpg\n\n",
        encoding="utf-8",
    )
    (out / "points3D.txt").write_text("# points\n1 0 0 0 255 255 255 0.1\n2 1 0 0 255 255 255 0.1\n", encoding="utf-8")
    print("Model conversion complete")
    raise SystemExit(0)

if cmd == "model_analyzer":
    print("Registered images: 4")
    print("Points: 2")
    raise SystemExit(0)

print(f"Unsupported fake COLMAP command: {cmd}", file=sys.stderr)
raise SystemExit(2)
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_sparse_model(root: Path) -> None:
    model = root / "data" / "sparse" / "site01" / "0"
    model.mkdir(parents=True, exist_ok=True)
    (model / "cameras.bin").write_bytes(b"camera-binary")
    (model / "images.bin").write_bytes(b"image-binary" * 8)
    (model / "points3D.bin").write_bytes(b"points-binary" * 16)


def _write_config(root: Path, fake_colmap: Path) -> Path:
    template_path = PROJECT_ROOT / "configs" / "site01.yaml"
    cfg = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    cfg["project"]["name"] = "site01"
    cfg["project"]["run_id"] = "smoke_stage_04"
    cfg["project"]["root"] = str(root)
    cfg.setdefault("paths", {})
    cfg["paths"].update({"sparse_dir": "data/sparse/site01", "sparse_refined_dir": "data/sparse_refined/site01"})
    cfg.setdefault("colmap", {})
    cfg["colmap"]["executable"] = str(fake_colmap)
    cfg["colmap"]["qt_qpa_platform"] = "offscreen"
    cfg.setdefault("refinement", {})
    cfg["refinement"].update(
        {
            "enabled": True,
            "input_sparse_dir": "data/sparse/site01/0",
            "output_sparse_dir": "data/sparse_refined/site01/0",
            "work_dir": "data/sparse_refined/site01/_work",
            "report_json": "runs/smoke_stage_04/reports/refinement_stats.json",
            "command_history_json": "data/sparse_refined/site01/command_history.json",
            "validate_binary_before_refinement": True,
            "ba_max_num_iterations": 5,
            "ba_max_linear_solver_iterations": 10,
            "refine_focal_length": True,
            "refine_principal_point": False,
            "refine_extra_params": True,
            "quality_gate_min_registered_images": 2,
            "quality_gate_min_points": 1,
            "ba_rounds": 1,
            "ba_num_threads": -1,
            "quality_gate_max_point_loss_ratio": 0.40,
            "quality_gate_fail_on_point_loss": True,
            "quality_gate_max_reprojection_error_increase_ratio": 0.10,
            "quality_gate_max_reprojection_error_increase_abs_px": 0.25,
            "quality_gate_fail_on_reprojection_error_increase": True,
            "fail_on_quality_gate": True,
            "pixsfm_enabled": False,
            "pixsfm_allow_missing": True,
        }
    )
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage04_smoke_") as tmp:
        root = Path(tmp)
        fake_colmap = root / "fake_colmap.py"
        _write_fake_colmap(fake_colmap)
        _write_sparse_model(root)
        cfg_path = _write_config(root, fake_colmap)
        cfg = load_config(cfg_path)
        report = run_refinement(cfg, force=True, log_level="ERROR")
        refined = Path(report["refined_sparse_model_dir"])
        if not refined.exists():
            raise SystemExit("STAGE_04_SMOKE_FAILED missing refined model")
        if not report["quality_gate"]["passed"]:
            raise SystemExit("STAGE_04_SMOKE_FAILED quality gate failed")
        if report.get("pycolmap_in_process_used") is not False:
            raise SystemExit("STAGE_04_SMOKE_FAILED in-process pycolmap should not be used")
        print(f"STAGE_04_SMOKE_OK refined={refined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
