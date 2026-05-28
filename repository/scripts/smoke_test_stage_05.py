#!/usr/bin/env python3
"""Fast Stage 5 smoke test using a fake COLMAP executable."""
from __future__ import annotations

import stat
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
from pipeline.stage_05_dense.run_dense import run_dense


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
    print("COLMAP fake 4.0.3 (with CUDA)")
    print("image_undistorter")
    print("patch_match_stereo")
    print("stereo_fusion")
    raise SystemExit(0)

if cmd == "image_undistorter" and "-h" in sys.argv:
    print("  --image_path arg")
    print("  --input_path arg")
    print("  --output_path arg")
    print("  --output_type arg (=COLMAP)")
    print("  --max_image_size arg (=-1)")
    raise SystemExit(0)

if cmd == "patch_match_stereo" and "-h" in sys.argv:
    print("  --workspace_path arg")
    print("  --workspace_format arg (=COLMAP)")
    print("  --PatchMatchStereo.max_image_size arg (=-1)")
    print("  --PatchMatchStereo.gpu_index arg (=-1)")
    print("  --PatchMatchStereo.window_radius arg (=5)")
    print("  --PatchMatchStereo.window_step arg (=1)")
    print("  --PatchMatchStereo.num_samples arg (=15)")
    print("  --PatchMatchStereo.num_iterations arg (=5)")
    print("  --PatchMatchStereo.geom_consistency arg (=1)")
    print("  --PatchMatchStereo.geom_consistency_regularizer arg (=0.3)")
    print("  --PatchMatchStereo.geom_consistency_max_cost arg (=3)")
    print("  --PatchMatchStereo.filter arg (=1)")
    print("  --PatchMatchStereo.filter_min_ncc arg (=0.1)")
    print("  --PatchMatchStereo.filter_min_triangulation_angle arg (=3)")
    print("  --PatchMatchStereo.filter_min_num_consistent arg (=2)")
    print("  --PatchMatchStereo.filter_geom_consistency_max_cost arg (=1)")
    print("  --PatchMatchStereo.cache_size arg (=32)")
    print("  --PatchMatchStereo.num_threads arg (=-1)")
    raise SystemExit(0)

if cmd == "stereo_fusion" and "-h" in sys.argv:
    print("  --workspace_path arg")
    print("  --workspace_format arg (=COLMAP)")
    print("  --input_type arg (=geometric)")
    print("  --output_path arg")
    print("  --StereoFusion.max_image_size arg (=-1)")
    print("  --StereoFusion.min_num_pixels arg (=5)")
    print("  --StereoFusion.max_reproj_error arg (=2)")
    print("  --StereoFusion.max_depth_error arg (=0.01)")
    raise SystemExit(0)

if cmd == "image_undistorter":
    image_path = Path(arg_value("--image_path", "/tmp/images"))
    output = Path(arg_value("--output_path", "/tmp/dense"))
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "sparse").mkdir(parents=True, exist_ok=True)
    for img in image_path.glob("*.jpg"):
        shutil.copy2(img, output / "images" / img.name)
    (output / "sparse" / "cameras.txt").write_text("# cameras\n", encoding="utf-8")
    (output / "sparse" / "images.txt").write_text("# images\n", encoding="utf-8")
    (output / "sparse" / "points3D.txt").write_text("# points\n", encoding="utf-8")
    print("Image undistortion complete")
    raise SystemExit(0)

if cmd == "patch_match_stereo":
    workspace = Path(arg_value("--workspace_path", "/tmp/dense"))
    stereo = workspace / "stereo"
    depth = stereo / "depth_maps"
    normal = stereo / "normal_maps"
    consistency = stereo / "consistency_graphs"
    depth.mkdir(parents=True, exist_ok=True)
    normal.mkdir(parents=True, exist_ok=True)
    consistency.mkdir(parents=True, exist_ok=True)
    for img in (workspace / "images").glob("*.jpg"):
        (depth / f"{img.name}.geometric.bin").write_bytes(b"depth")
        (normal / f"{img.name}.geometric.bin").write_bytes(b"normal")
        (consistency / f"{img.name}.geometric.bin").write_bytes(b"graph")
    (stereo / "patch-match.cfg").write_text("__auto__, 5\n", encoding="utf-8")
    (stereo / "fusion.cfg").write_text("__auto__\n", encoding="utf-8")
    print("PatchMatch complete")
    raise SystemExit(0)

if cmd == "stereo_fusion":
    output = Path(arg_value("--output_path", "/tmp/dense/fused.ply"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "ply\nformat ascii 1.0\nelement vertex 42\nproperty float x\nproperty float y\nproperty float z\nend_header\n" +
        "\n".join("0 0 0" for _ in range(42)) + "\n",
        encoding="utf-8",
    )
    print("Stereo fusion complete")
    raise SystemExit(0)

print(f"Unsupported fake COLMAP command: {cmd}", file=sys.stderr)
raise SystemExit(2)
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_inputs(root: Path, count: int = 4) -> None:
    images = root / "data" / "sfm" / "site01" / "images"
    images.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        arr = np.full((50, 70, 3), 35 + idx * 40, dtype=np.uint8)
        if not cv2.imwrite(str(images / f"site01_kf_{idx + 1:05d}.jpg"), arr):
            raise RuntimeError("Failed to write smoke image")
    sparse = root / "data" / "sparse_refined" / "site01" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    (sparse / "cameras.bin").write_bytes(b"camera")
    (sparse / "images.bin").write_bytes(b"images" * 10)
    (sparse / "points3D.bin").write_bytes(b"points" * 20)


def _write_config(root: Path, fake_colmap: Path) -> Path:
    template = PROJECT_ROOT / "configs" / "site01.yaml"
    cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
    cfg["project"]["name"] = "site01"
    cfg["project"]["run_id"] = "smoke_stage_05"
    cfg["project"]["root"] = str(root)
    cfg.setdefault("colmap", {})["executable"] = str(fake_colmap)
    cfg["colmap"]["qt_qpa_platform"] = "offscreen"
    cfg.setdefault("dense", {}).update(
        {
            "input_sparse_refined_dir": "data/sparse_refined/site01/0",
            "input_images_dir": "data/sfm/site01/images",
            "workspace_dir": "data/dense/site01",
            "fused_ply": "data/dense/site01/fused.ply",
            "report_json": "runs/smoke_stage_05/reports/dense_summary.json",
            "command_history_json": "data/dense/site01/command_history.json",
            "min_input_images": 2,
            "require_cuda": True,
            "cuda_preflight": True,
            "gpu_preflight": False,
            "adaptive_gpu_profile": False,
            "patch_match_gpu_index": "-1",
            "max_image_size": 800,
            "patch_match_max_image_size": 800,
            "patch_window_radius": 3,
            "patch_num_iterations": 1,
            "fusion_min_num_pixels": 1,
            "quality_min_fused_points": 10,
            "quality_min_fused_points_per_image": 5.0,
            "quality_min_depth_map_ratio_warning": 0.5,
            "fail_on_quality_gate": True,
        }
    )
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage05_smoke_") as tmp:
        root = Path(tmp)
        fake_colmap = root / "fake_colmap.py"
        _write_fake_colmap(fake_colmap)
        _write_inputs(root)
        cfg_path = _write_config(root, fake_colmap)
        cfg = load_config(cfg_path)
        report = run_dense(cfg, force=True, log_level="ERROR")
        stats = report["dense_stats"]
        if stats.get("fused_vertex_count") != 42:
            raise SystemExit(f"STAGE_05_SMOKE_FAILED expected 42 vertices, got {stats.get('fused_vertex_count')}")
        if not Path(report["fused_ply"]).exists():
            raise SystemExit("STAGE_05_SMOKE_FAILED missing fused.ply")
        print(f"STAGE_05_SMOKE_OK vertices={stats.get('fused_vertex_count')} fused={report['fused_ply']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
