"""Read-only novel-view rasterizer for the *official gsplat* canonical .pt format.

Input must be a trusted gsplat checkpoint with dict['splats'] containing tensors
means, scales, quats, opacities, sh0, shN. No optimizer/training is instantiated.
No input file is ever changed. GPU work occurs only with --execute.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np

SAFE_RUN = re.compile(r"^geom-[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_camera_path(data: dict) -> list[dict]:
    if data.get("frame_convention") != "colmap_opencv_world_to_camera":
        raise ValueError("Invalid or missing frame_convention: expected colmap_opencv_world_to_camera")
    cameras = data.get("cameras")
    if not isinstance(cameras, list) or not (1 <= len(cameras) <= 60):
        raise ValueError("Provide 1 to 60 novel camera poses")
    validated = []
    for i, camera in enumerate(cameras):
        if not isinstance(camera, dict):
            raise ValueError(f"Invalid camera entry {i}")
        height, width = int(camera["height"]), int(camera["width"])
        if not (16 <= height <= 4096 and 16 <= width <= 4096):
            raise ValueError("Camera render dimensions out of allowed range")
        fx, fy, cx, cy = [float(camera[k]) for k in ("fx", "fy", "cx", "cy")]
        if not all(math.isfinite(k) for k in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
            raise ValueError("Invalid intrinsic matrix")
        viewmat = np.asarray(camera["world_to_camera_4x4"], dtype=np.float64)
        if viewmat.shape != (4, 4) or not np.isfinite(viewmat).all():
            raise ValueError("Camera viewmat must be finite 4x4")
        if not np.allclose(viewmat[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("Invalid camera homogeneous last row")
        rot = viewmat[:3, :3]
        if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-3) or np.linalg.det(rot) < 0.999:
            raise ValueError("Camera rotation must be a proper rigid rotation")
        validated.append({
            "width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "world_to_camera_4x4": viewmat.tolist(),
            "view_id": str(camera.get("view_id", i)),
            "is_training_image": camera.get("is_training_image"),
        })
    return validated


def inspect_render_command(project_root: Path, run_id: str, checkpoint: Path, camera_json: Path) -> dict:
    if not SAFE_RUN.fullmatch(run_id):
        raise ValueError("Use a safe geom-* run ID")
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0 or checkpoint.suffix.lower() != ".pt":
        raise FileNotFoundError("Expected existing nonempty official gsplat canonical .pt checkpoint")
    if not camera_json.is_file():
        raise FileNotFoundError(camera_json)
    cameras = validate_camera_path(json.loads(camera_json.read_text(encoding="utf-8")))
    target = project_root.expanduser().resolve() / "runs" / run_id / "geometry" / "gsplat_render"
    return {
        "schema": "mycon.geometry.gsplat_render.v1",
        "run_id": run_id,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "camera_spec_path": str(camera_json.resolve()),
        "camera_count": len(cameras),
        "output_dir": str(target),
        "frame_convention": "colmap_opencv_world_to_camera",
        "status": "inspection_only",
        "trained_model": "existing_official_gsplat_checkpoint",
        "training_executed": False,
        "render_executed": False,
        "novel_view_holdout_verified": False,
        "metric_scale_verified": False,
        "bim_geometry_verified": False,
    }


def render_from_checkpoint(
    *, project_root: Path, run_id: str, checkpoint: Path,
    camera_json: Path, execute: bool = False,
) -> dict:
    """Render RGB and a depth channel; outputs are unverified COLMAP frame units."""
    report = inspect_render_command(project_root, run_id, checkpoint, camera_json)
    if not execute:
        return report
    target = Path(report["output_dir"])
    if target.exists():
        raise FileExistsError(f"Refusing existing output directory: {target}")
    # Check runtime before creating anything. Optional GPU libraries are imported
    # lazily, so inspect and lightweight tests need no CUDA or Torch packages.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: no render attempted")
    from gsplat.rendering import rasterization
    from PIL import Image

    cameras = validate_camera_path(json.loads(camera_json.read_text(encoding="utf-8")))
    start = time.monotonic()
    target.mkdir(parents=True, exist_ok=False)
    report_path = target / "render_report.json"
    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["status"] = "render_in_progress"
    save()
    try:
        # The canonical checkpoint stores only plain tensors; weights_only=True
        # prevents arbitrary pickled Python objects from executing.
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict) or not isinstance(raw.get("splats"), dict):
            raise ValueError("Not an official canonical gsplat {'splats': tensors} file")
        splats = raw["splats"]
        needed = ("means", "scales", "quats", "opacities", "sh0", "shN")
        if not all(isinstance(splats.get(k), torch.Tensor) for k in needed):
            raise ValueError(f"Required tensor fields missing: {needed}")
        count = int(splats["means"].shape[0])
        if count == 0 or splats["means"].shape != (count, 3):
            raise ValueError("Invalid Gaussian mean tensor shape")
        if splats["scales"].shape != (count, 3) or splats["quats"].shape != (count, 4):
            raise ValueError("Invalid scale/quaternion tensor shapes")
        if splats["opacities"].numel() != count:
            raise ValueError("Invalid opacity tensor shape")
        if splats["sh0"].ndim != 3 or splats["shN"].ndim != 3:
            raise ValueError("Invalid SH color tensor ranks")
        if splats["sh0"].shape[0] != count or splats["shN"].shape[0] != count:
            raise ValueError("Mismatched SH color tensors")
        if not all(bool(torch.isfinite(splats[k]).all().item()) for k in needed):
            raise ValueError("Nonfinite canonical splat tensor")
        dev = torch.device("cuda")
        means = splats["means"].to(dev).float()
        scales = torch.exp(splats["scales"].to(dev).float())
        quats = torch.nn.functional.normalize(splats["quats"].to(dev).float(), dim=-1)
        opacities = torch.sigmoid(splats["opacities"].to(dev).float().reshape(-1))
        colors = torch.cat([splats["sh0"].to(dev).float(),
                            splats["shN"].to(dev).float()], dim=1)
        if colors.shape[2] != 3:
            raise ValueError("Expected RGB spherical-harmonic coefficients")
        degree = round(math.sqrt(colors.shape[1]) - 1)
        if (degree + 1) ** 2 != colors.shape[1] or degree < 0 or degree > 4:
            raise ValueError("Invalid spherical-harmonic coefficient count")
        report["gaussian_count"] = count
        report["sh_degree"] = degree
        report["checkpoint_step"] = raw.get("step") if isinstance(raw.get("step"), int) else None
        frames = []
        for index, camera in enumerate(cameras):
            K = torch.tensor(
                [[[camera["fx"], 0, camera["cx"]],
                  [0, camera["fy"], camera["cy"]],
                  [0, 0, 1]]],
                dtype=torch.float32, device=dev,
            )
            viewmats = torch.tensor([camera["world_to_camera_4x4"]],
                                    dtype=torch.float32, device=dev)
            with torch.no_grad():
                rendered, alpha, _ = rasterization(
                    means=means, quats=quats, scales=scales, opacities=opacities,
                    colors=colors, viewmats=viewmats, Ks=K,
                    width=camera["width"], height=camera["height"],
                    sh_degree=degree, packed=True, render_mode="RGB+D",
                )
            if not torch.isfinite(rendered).all():
                raise ValueError(f"Nonfinite image at camera {index}")
            frame = rendered[0].detach().cpu().numpy()
            rgb = np.uint8(np.clip(frame[..., :3], 0, 1) * 255)
            image_name = f"view_{index:04d}.png"
            Image.fromarray(rgb, mode="RGB").save(target / image_name)
            depth_name = f"depth_{index:04d}_unscaled.npy"
            np.save(target / depth_name, frame[..., 3].astype(np.float32), allow_pickle=False)
            frames.append({"view_id": camera["view_id"], "rgb_png": image_name,
                           "depth_npy": depth_name,
                           "is_training_image": camera["is_training_image"]})
        report["frames"] = frames
        report["render_executed"] = True
        report["status"] = "rendered_unverified_camera_holdout_and_scale"
        report["wall_time_sec"] = round(time.monotonic() - start, 3)
    except Exception as exc:
        report["status"] = "render_failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        save()
        raise
    save()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Existing gsplat .pt checkpoint to candidate novel-view images")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cameras-json", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Opt in to GPU rasterization ONLY; never trains")
    args = parser.parse_args(argv)
    try:
        result = render_from_checkpoint(project_root=args.project_root, run_id=args.run_id,
                                        checkpoint=args.checkpoint,
                                        camera_json=args.cameras_json, execute=args.execute)
    except (RuntimeError, OSError, ValueError, ImportError, KeyError) as exc:
        print(json.dumps({"status": "blocked_or_failed", "reason": str(exc)}))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
