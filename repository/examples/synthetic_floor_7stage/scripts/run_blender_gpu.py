#!/usr/bin/env python3
"""Host-side orchestrator for the GPU Blender renderer.

This is the GPU/Blender twin of ``run_generate.py``. It:

1. Resolves the path to the Blender executable (``--blender`` or
   the ``BLENDER`` environment variable).
2. Generates the BIM + mesh + sidecar JSON for each stage (via the
   existing pipeline) if not already present.
3. Calls ``blender_gpu_renderer.py`` once per stage as a subprocess,
   feeding it the right input mesh and CLI flags.
4. Encodes the rendered RGB frames into ``stage_NN.mp4`` using
   ``imageio-ffmpeg`` (already a dependency of the repo) so the GPU
   pipeline output matches the existing CSV/manifest format.

Run from the repository root, in Colab or locally::

    PYTHONPATH=examples/synthetic_floor_7stage/src \\
        python3 examples/synthetic_floor_7stage/scripts/run_blender_gpu.py \\
            --blender /content/blender/blender \\
            --stages 1 4 7 \\
            --resolution 1280 720 \\
            --samples 128

If you want to render every stage::

    PYTHONPATH=examples/synthetic_floor_7stage/src \\
        python3 examples/synthetic_floor_7stage/scripts/run_blender_gpu.py \\
            --blender /content/blender/blender
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE_ROOT = HERE.parent
SRC = EXAMPLE_ROOT / "src"
RENDERER_SCRIPT = SRC / "synthetic_floor" / "blender_gpu_renderer.py"

sys.path.insert(0, str(SRC))

# Reuse the existing CPU pipeline modules to prepare BIM/mesh/elements
# JSON for each stage. We never call the CPU renderer itself.
from synthetic_floor.scene_spec import load_scene_spec  # noqa: E402
from synthetic_floor.layout import build_layout, elements_by_category  # noqa: E402
from synthetic_floor.stage_controller import select_for_stage  # noqa: E402
from synthetic_floor.materials import build_material_library  # noqa: E402
from synthetic_floor.ifc_builder import write_stage_ifc  # noqa: E402
from synthetic_floor.mesh_builder import write_stage_meshes  # noqa: E402
from synthetic_floor.metadata_exporter import (  # noqa: E402
    write_dataset_schedule,
    write_bim_schedule_mapping,
    write_element_metrics_csv,
    write_manifest,
)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


def setup_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run_blender_gpu.log"
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger("synthetic_floor.gpu")


# ---------------------------------------------------------------------
# Blender executable resolution
# ---------------------------------------------------------------------


def _resolve_blender(arg: str | None) -> str:
    """Find a Blender executable, raising a clear error if missing."""
    candidates: list[str] = []
    if arg:
        candidates.append(arg)
    env_var = os.environ.get("BLENDER")
    if env_var:
        candidates.append(env_var)
    candidates.extend([
        "/content/blender/blender",  # the Colab default install location
        "/usr/local/bin/blender",
        "/usr/bin/blender",
        "blender",
    ])
    for c in candidates:
        if Path(c).is_file() and os.access(c, os.X_OK):
            return c
        # Try resolving from PATH
        which = shutil.which(c)
        if which:
            return which
    raise FileNotFoundError(
        "Could not find the 'blender' executable. Pass --blender PATH, set the "
        "BLENDER env var, or run scripts/setup_colab_blender.sh first."
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path,
                   default=EXAMPLE_ROOT / "config" / "scene.yaml")
    p.add_argument("--blender", type=str, default=None,
                   help="Path to blender executable; default: /content/blender/blender")
    p.add_argument("--stages", type=int, nargs="*", default=None,
                   help="Subset of stage ids to render (1..7). Default: all.")
    p.add_argument("--resolution", nargs=2, type=int, default=[1280, 720],
                   metavar=("W", "H"))
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--frames", type=int, default=None,
                   help="Override frames per stage (default: derived from "
                        "config duration_per_stage_sec * fps).")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", default="OPTIX",
                   choices=("OPTIX", "CUDA", "CPU"))
    p.add_argument("--no-motion-blur", action="store_true")
    p.add_argument("--sun-elevation", type=float, default=38.0)
    p.add_argument("--sun-azimuth", type=float, default=135.0)
    p.add_argument("--skip-prepare", action="store_true",
                   help="Skip generating BIM/mesh/sidecar (assume present).")
    p.add_argument("--skip-encode", action="store_true",
                   help="Skip MP4 encoding step.")
    p.add_argument("--quick", action="store_true",
                   help="Tiny smoke test: 480x270, 30 frames, 32 samples.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


# ---------------------------------------------------------------------
# Stage preparation (BIM + mesh + sidecar)
# ---------------------------------------------------------------------


def prepare_stage_assets(spec, stage_ids: list[int], log: logging.Logger) -> dict[int, dict]:
    """Generate IFC, GLB, and elements sidecar for the requested stages.

    Returns ``{stage_id: {"ifc": Path, "glb": Path, "elements_json": Path, "csv": Path}}``.
    """
    elements = build_layout(spec)
    by_cat = elements_by_category(elements)
    library = build_material_library(seed=spec.random_seed)

    out: dict[int, dict] = {}
    for sid in stage_ids:
        stage = spec.stages[sid - 1]
        staged = select_for_stage(stage, by_cat)
        ifc_path = spec.output.bim / f"stage_{sid:02d}.ifc"
        if not ifc_path.exists():
            try:
                write_stage_ifc(spec, sid, staged, ifc_path)
            except Exception as e:
                log.warning("IFC export failed for stage %d: %s", sid, e)

        meshes = write_stage_meshes(spec, sid, staged, library, spec.output.mesh)

        # Per-stage element_metrics CSV (Stage-9 contract) -- the GPU
        # renderer doesn't need it, but the rest of the pipeline does.
        csv_path = spec.output.manifests / f"stage_{sid:02d}_element_metrics.csv"
        write_element_metrics_csv(staged, csv_path)

        out[sid] = {
            "ifc": ifc_path,
            "glb": meshes["glb"],
            "obj": meshes["obj"],
            "elements_json": meshes["elements_json"],
            "element_metrics_csv": csv_path,
        }
        log.info("[stage %d] prepared GLB=%s elements=%s",
                 sid, meshes["glb"].name, meshes["elements_json"].name)
    return out


# ---------------------------------------------------------------------
# Blender invocation
# ---------------------------------------------------------------------


def render_stage_with_blender(
    blender_exe: str,
    stage_id: int,
    glb_path: Path,
    elements_json: Path,
    output_dir: Path,
    args: argparse.Namespace,
    fps: int,
    frames: int,
    log: logging.Logger,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        blender_exe, "-b",
        "--python", str(RENDERER_SCRIPT),
        "--",
        "--input-mesh", str(glb_path),
        "--elements-json", str(elements_json),
        "--output-dir", str(output_dir),
        "--stage-id", str(stage_id),
        "--frames", str(frames),
        "--fps", str(fps),
        "--samples", str(args.samples),
        "--resolution", str(args.resolution[0]), str(args.resolution[1]),
        "--sun-elevation", str(args.sun_elevation),
        "--sun-azimuth", str(args.sun_azimuth),
        "--device", args.device,
    ]
    if args.no_motion_blur:
        cmd.append("--no-motion-blur")

    log.info("[stage %d] launching: %s", stage_id, " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    log.info("[stage %d] blender exited rc=%d in %.1fs", stage_id, proc.returncode, dt)

    # Always persist Blender's stdout/stderr alongside the outputs
    (output_dir / "blender_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "blender_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        # Surface the last few lines for quick inspection
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
        log.error("[stage %d] blender failed:\n%s", stage_id, tail)
    return proc.returncode


# ---------------------------------------------------------------------
# MP4 encoding
# ---------------------------------------------------------------------


def encode_mp4(frames_dir: Path, out_path: Path, fps: int, log: logging.Logger) -> Path | None:
    """Encode PNGs to MP4. Robust: handles mixed sizes, corrupt frames, progress."""
    pngs = sorted(frames_dir.glob("frame_*.png"))
    if not pngs:
        log.warning("no PNGs found in %s; skipping MP4 encode", frames_dir)
        return None
    try:
        import imageio.v3 as iio
        import numpy as np
        from PIL import Image
    except Exception as e:
        log.warning("imageio/Pillow not available (%s); skipping MP4 encode", e)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load frames, normalising to a common size + RGB mode.
    # Skip corrupt or unreadable files instead of crashing.
    valid_frames: list = []
    target_size: tuple | None = None
    for i, p in enumerate(pngs):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            log.warning("  skipping corrupt frame %s: %s", p.name, e)
            continue
        if target_size is None:
            target_size = img.size  # (W, H)
        elif img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)
        valid_frames.append(np.asarray(img))
        # Progress indicator every 20% of frames
        if (i + 1) % max(1, len(pngs) // 5) == 0:
            log.info("  encoding: loaded %d/%d frames ...", i + 1, len(pngs))

    if not valid_frames:
        log.error("no valid frames in %s; MP4 not written.", frames_dir)
        return None

    arrays = np.stack(valid_frames, axis=0)
    iio.imwrite(out_path, arrays, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    log.info("wrote %s (%d frames, %dx%d)", out_path, len(valid_frames),
             target_size[0] if target_size else 0, target_size[1] if target_size else 0)
    return out_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    spec = load_scene_spec(args.config)

    # Apply --quick / --frames overrides on the camera spec (frames per stage
    # is what matters for Blender; resolution is passed via CLI).
    cam_kwargs: dict = {}
    if args.quick:
        cam_kwargs.update({"width_px": 480, "height_px": 270, "duration_per_stage_sec": 1.0})
        args.resolution = [480, 270]
        args.samples = 32
    if args.frames is None:
        if cam_kwargs:
            duration = cam_kwargs["duration_per_stage_sec"]
        else:
            duration = spec.camera.duration_per_stage_sec
        frames = max(2, int(round(duration * args.fps)))
    else:
        frames = int(args.frames)

    spec.output.ensure()
    log = setup_logging(spec.output.logs, args.log_level)

    log.info("=== run_blender_gpu ===")
    log.info("config       : %s", spec.config_path)
    log.info("output root  : %s", spec.output.root)
    log.info("resolution   : %dx%d", args.resolution[0], args.resolution[1])
    log.info("samples      : %d", args.samples)
    log.info("frames       : %d @ %d fps  (%.1fs/stage)", frames, args.fps, frames / args.fps)
    log.info("device       : %s", args.device)

    stage_ids = args.stages or [s.id for s in spec.stages]
    log.info("stages       : %s", stage_ids)

    if not args.skip_prepare:
        log.info("preparing stage assets (BIM + mesh + sidecar) ...")
        prepared = prepare_stage_assets(spec, stage_ids, log)
    else:
        prepared = {}
        for sid in stage_ids:
            prepared[sid] = {
                "glb": spec.output.mesh / f"stage_{sid:02d}.glb",
                "elements_json": spec.output.mesh / f"stage_{sid:02d}_elements.json",
            }

    # Dataset-level CSVs (schedule + mapping). These are deterministic
    # and small, regenerate every run.
    elements = build_layout(spec)
    write_dataset_schedule(spec, elements, spec.output.manifests / "schedule.csv")
    write_bim_schedule_mapping(spec, elements, spec.output.manifests / "bim_schedule_mapping.csv")

    # Resolve Blender AFTER preparation so a dry-run that just wants the
    # GLB + sidecar JSON outputs (e.g. for CI / asset inspection) still
    # gets useful artefacts even on a machine without Blender installed.
    try:
        blender_exe = _resolve_blender(args.blender)
        log.info("blender      : %s", blender_exe)
    except FileNotFoundError as e:
        log.error("%s", e)
        log.info("preparation finished; rendering skipped because Blender is not available.")
        return 0

    blender_renders = spec.output.root / "blender_renders"
    blender_renders.mkdir(parents=True, exist_ok=True)

    # Import the progress bar (non-critical; fallback to plain logging)
    try:
        from synthetic_floor.progress import ProgressBar
    except ImportError:
        ProgressBar = None  # type: ignore[assignment,misc]

    files_by_stage: dict[int, dict] = {}
    if ProgressBar:
        pb = ProgressBar(total=len(stage_ids), label="stages")
    for sid in stage_ids:
        info = prepared[sid]
        if not Path(info["glb"]).exists():
            log.error("[stage %d] GLB missing: %s", sid, info["glb"])
            continue
        stage_dir = blender_renders / f"stage_{sid:02d}"
        rc = render_stage_with_blender(
            blender_exe=blender_exe,
            stage_id=sid,
            glb_path=Path(info["glb"]),
            elements_json=Path(info["elements_json"]),
            output_dir=stage_dir,
            args=args,
            fps=args.fps,
            frames=frames,
            log=log,
        )
        if rc != 0:
            log.error("stage %d failed; continuing with the rest.", sid)
            continue

        video_path = None
        if not args.skip_encode:
            video_path = encode_mp4(
                stage_dir / "rgb",
                spec.output.video / f"stage_{sid:02d}_blender.mp4",
                fps=args.fps,
                log=log,
            )

        files_by_stage[sid] = {
            "glb": Path(info["glb"]),
            "elements_json": Path(info["elements_json"]),
            "blender_dir": stage_dir,
            "rgb_dir": stage_dir / "rgb",
            "depth_dir": stage_dir / "depth",
            "seg_dir": stage_dir / "seg",
            "camera_path": stage_dir / "camera_path.json",
            "video": video_path,
            "element_metrics_csv": info.get("element_metrics_csv"),
        }
        if ProgressBar:
            pb.update(1)

    # Refresh manifest so the new files show up
    extras = {
        "schedule_csv": spec.output.manifests / "schedule.csv",
        "bim_schedule_mapping_csv": spec.output.manifests / "bim_schedule_mapping.csv",
    }
    manifest_path = spec.output.manifests / "manifest_blender_gpu.json"
    write_manifest(spec, files_by_stage, extras, manifest_path)
    log.info("manifest -> %s", manifest_path)
    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
