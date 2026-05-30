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
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE_ROOT = HERE.parent
SRC = EXAMPLE_ROOT / "src"
RENDERER_SCRIPT = SRC / "synthetic_floor" / "blender_gpu_renderer.py"

sys.path.insert(0, str(SRC))

# Reuse the existing CPU pipeline modules to prepare BIM/mesh/elements
# JSON for each stage. We never call the CPU renderer itself.
from synthetic_floor.scene_spec import load_scene_spec  # noqa: E402
from synthetic_floor.presets import (  # noqa: E402
    PRESET_NAMES, DEFAULT_PRESET, apply_gpu_preset, describe_preset,
)
from synthetic_floor import paths as P  # noqa: E402
from synthetic_floor.checkpoint import (  # noqa: E402
    RESUME_MODES, DEFAULT_RESUME_MODE, filter_stages, write_done_marker,
    write_run_state,
)
from synthetic_floor import colab_sync as CS  # noqa: E402
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
    p.add_argument("--preset", choices=PRESET_NAMES, default=None,
                   help="Quality preset: debug, balanced (default), or hq. "
                        "Explicit --resolution/--samples/--frames override.")
    p.add_argument("--resolution", nargs=2, type=int, default=None,
                   metavar=("W", "H"))
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--frames", type=int, default=None,
                   help="Override frames per stage.")
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--device", default="OPTIX",
                   choices=("OPTIX", "CUDA", "CPU"))
    p.add_argument("--no-motion-blur", action="store_true")
    p.add_argument("--motion-blur", action="store_true",
                   help="Force motion blur ON (overrides preset).")
    p.add_argument("--sun-elevation", type=float, default=38.0)
    p.add_argument("--sun-azimuth", type=float, default=135.0)
    p.add_argument("--skip-prepare", action="store_true",
                   help="Skip generating BIM/mesh/sidecar (assume present).")
    p.add_argument("--skip-encode", action="store_true",
                   help="Skip MP4 encoding step.")
    p.add_argument("--quick", action="store_true",
                   help="Alias for --preset debug.")
    p.add_argument("--resume", action="store_true",
                   help="Skip stages whose Blender outputs already exist (alias for --mode resume).")
    p.add_argument("--force", action="store_true",
                   help="Delete prior outputs and rerun (alias for --mode force).")
    p.add_argument("--mode", choices=RESUME_MODES, default=None,
                   help="Resume policy. Default: 'run' (always rerun).")
    p.add_argument("--strict-render", action="store_true",
                   help="Fail loudly if Blender does not produce the expected "
                        "number of frames or any of the required outputs.")
    p.add_argument("--save-blend", action="store_true",
                   help="Save a self-contained .blend per stage and zip it under "
                        "output/blend/ for download (open in Blender on Windows).")
    # --- Google Drive persistence / resume ---
    p.add_argument("--drive-root", type=Path, default=None,
                   help="Drive folder to mirror outputs into (e.g. "
                        "/content/drive/MyDrive/MyCon_Colab/synthetic_floor_7stage/<run>). "
                        "Implies Drive sync; enables cross-session/device resume.")
    p.add_argument("--mount-drive", action="store_true",
                   help="Mount Google Drive and auto-resolve --drive-root from the run_id "
                        "(Colab). Outputs are synced to Drive after every stage.")
    p.add_argument("--no-drive", action="store_true",
                   help="Disable all Drive syncing even on Colab.")
    p.add_argument("--drive-sync-interval", type=float, default=60.0,
                   help="Background Drive sync period in seconds (default 60).")
    p.add_argument("--start-stage", type=int, default=None, metavar="N",
                   help="Render stages N..7 (e.g. --start-stage 5). Ignored if "
                        "--stages is given. Combine with --resume to continue.")
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
        ifc_path = P.stage_ifc_path(spec, sid)
        if not ifc_path.exists():
            try:
                write_stage_ifc(spec, sid, staged, ifc_path)
            except Exception as e:
                log.warning("IFC export failed for stage %d: %s", sid, e)

        meshes = write_stage_meshes(spec, sid, staged, library, spec.output.mesh)

        # Per-stage element_metrics CSV (Stage-9 contract) -- the GPU
        # renderer doesn't need it, but the rest of the pipeline does.
        csv_path = P.stage_element_metrics_csv(spec, sid)
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
    width: int,
    height: int,
    samples: int,
    motion_blur: bool,
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
        "--samples", str(samples),
        "--resolution", str(width), str(height),
        "--sun-elevation", str(args.sun_elevation),
        "--sun-azimuth", str(args.sun_azimuth),
        "--device", args.device,
    ]
    if not motion_blur:
        cmd.append("--no-motion-blur")
    if getattr(args, "save_blend", False):
        cmd.append("--save-blend")

    log.info("[stage %d] launching: %s", stage_id, " ".join(cmd))
    t0 = time.time()

    # Live intra-stage progress: the renderer writes render_progress.json
    # after every frame. Tail it so the user sees frame-level progress (and
    # an ETA) instead of a silent multi-hour stage.
    progress_file = output_dir / "render_progress.json"
    stop = threading.Event()

    def _monitor() -> None:
        import json as _json
        last = -1
        while not stop.wait(15.0):
            try:
                d = _json.loads(progress_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            done = d.get("frames_done", 0)
            if done != last:
                last = done
                log.info("[stage %d] progress: %s/%s frames (%.1f%%) eta=%.1fmin status=%s",
                         stage_id, done, d.get("frames_total", "?"),
                         d.get("percent", 0.0), d.get("eta_sec", 0.0) / 60.0,
                         d.get("status", "?"))

    mon = threading.Thread(target=_monitor, name=f"progress-{stage_id}", daemon=True)
    mon.start()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stop.set()
    mon.join(timeout=2)
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
# Frame quality guard (catches "nothing but light" / blank renders)
# ---------------------------------------------------------------------


def check_frame_quality(frames_dir: Path, log: logging.Logger) -> dict | None:
    """Inspect a representative RGB frame for the "all light / blank" failure.

    Returns ``{"mean","std","white_frac","ok"}`` or ``None`` if it could not
    inspect. A frame that is almost entirely near-white (or has near-zero
    spatial variance) means the geometry was not in view / the scene was
    blown out — exactly the bug the alignment + exposure fixes address.
    """
    pngs = sorted(Path(frames_dir).glob("frame_*.png"))
    if not pngs:
        return None
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None
    mid = pngs[len(pngs) // 2]
    try:
        arr = np.asarray(Image.open(mid).convert("RGB"), dtype=np.float32)
    except Exception:
        return None
    lum = arr.mean(axis=2)
    mean = float(lum.mean())
    std = float(lum.std())
    white_frac = float((lum > 250.0).mean())
    # "Blank" if almost all pixels are near-white, or there is essentially no
    # spatial structure (a flat colour field).
    ok = not (white_frac > 0.97 or std < 3.0)
    report = {"frame": mid.name, "mean": round(mean, 1), "std": round(std, 2),
              "white_frac": round(white_frac, 3), "ok": ok}
    if ok:
        log.info("[quality] %s mean=%.1f std=%.2f white=%.1f%% -> OK",
                 mid.name, mean, std, white_frac * 100)
    else:
        log.warning("[quality] %s mean=%.1f std=%.2f white=%.1f%% -> LOOKS BLANK "
                    "(geometry off-frame or over-exposed)",
                    mid.name, mean, std, white_frac * 100)
    return report


def zip_blend_project(spec, stage_id: int, stage_dir: Path, log: logging.Logger) -> Path | None:
    """Zip the per-stage .blend project under output/blend/ for download.

    The .blend is self-contained (procedural materials + packed data), so the
    zip can be downloaded and opened in Blender on Windows/macOS/Linux.
    """
    import zipfile

    blend_path = Path(stage_dir) / f"stage_{stage_id:02d}.blend"
    if not blend_path.exists():
        log.warning("[blend] stage %d: no .blend was produced (%s)", stage_id, blend_path)
        return None
    blend_dir = spec.output.root / "blend"
    blend_dir.mkdir(parents=True, exist_ok=True)
    zip_path = blend_dir / f"{spec.project_name}_stage_{stage_id:02d}.blend.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(blend_path, arcname=blend_path.name)
        # Include a short README so the recipient knows what it is.
        zf.writestr(
            "README.txt",
            f"{spec.project_name} stage {stage_id}\n"
            f"Open {blend_path.name} in Blender 4.2+ and press F12 (or play the\n"
            f"timeline) to re-render. Materials are procedural; no external\n"
            f"textures are required.\n",
        )
    log.info("[blend] stage %d -> %s (%.1f MB)", stage_id, zip_path,
             zip_path.stat().st_size / 1024 / 1024)
    return zip_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    spec = load_scene_spec(args.config)

    # Resolve preset (CLI flags override preset values).
    preset_name = args.preset or ("debug" if args.quick else DEFAULT_PRESET)
    motion_blur: bool | None
    if args.no_motion_blur:
        motion_blur = False
    elif args.motion_blur:
        motion_blur = True
    else:
        motion_blur = None  # inherit from preset
    preset = apply_gpu_preset(
        preset_name,
        resolution=tuple(args.resolution) if args.resolution else None,
        samples=args.samples,
        frames=args.frames,
        fps=args.fps,
        motion_blur=motion_blur,
    )
    # Make the resolved values visible to the rest of main() through
    # familiar variable names.
    width, height = preset.width_px, preset.height_px
    samples = preset.samples
    frames = preset.frames_per_stage
    fps = preset.fps

    spec.output.ensure()
    log = setup_logging(spec.output.logs, args.log_level)

    log.info("=== run_blender_gpu ===")
    log.info("preset       : %s", describe_preset(preset_name, gpu=True))
    log.info("config       : %s", spec.config_path)
    log.info("output root  : %s", spec.output.root)
    log.info("resolution   : %dx%d", width, height)
    log.info("samples      : %d", samples)
    log.info("frames       : %d @ %d fps  (%.1fs/stage)", frames, fps, frames / fps)
    log.info("motion blur  : %s", preset.motion_blur)
    log.info("device       : %s", args.device)

    stage_ids = args.stages or [s.id for s in spec.stages]
    if not args.stages and args.start_stage is not None:
        stage_ids = [s.id for s in spec.stages if s.id >= int(args.start_stage)]
        log.info("start-stage=%d -> rendering stages %s", args.start_stage, stage_ids)
    all_stage_ids = list(stage_ids)
    log.info("stages req   : %s", stage_ids)

    # --- Google Drive persistence -------------------------------------------
    # Mount Drive (Colab) and mirror the whole output tree to Drive so a
    # disconnect/runtime reset never loses work, and a run can be resumed
    # from another device or Drive account. We PULL any prior outputs back
    # BEFORE resume planning so completed stages are detected.
    drive_mirror = None
    if not args.no_drive and (args.drive_root is not None or args.mount_drive):
        if CS.maybe_mount_drive(log=log.info):
            drive_root = args.drive_root or CS.default_drive_root(spec.run_id)
            log.info("drive root   : %s", drive_root)
            drive_mirror = CS.DriveMirror(
                local_root=spec.output.root,
                drive_root=drive_root,
                log=log.info,
                interval=args.drive_sync_interval,
            )
            pulled = drive_mirror.pull()
            log.info("drive pull   : %s", pulled)
            drive_mirror.start()
        else:
            log.warning("Drive requested but not available; continuing local-only.")

    # Resolve resume mode: --force/--resume are convenience aliases for --mode.
    if args.mode is not None:
        resume_mode = args.mode
    elif args.force:
        resume_mode = "force"
    elif args.resume:
        resume_mode = "resume"
    else:
        resume_mode = DEFAULT_RESUME_MODE
    log.info("resume mode  : %s", resume_mode)
    stage_ids, statuses = filter_stages(spec, stage_ids, resume_mode, gpu=True, log=log)
    log.info("stages run   : %s", stage_ids)
    if not stage_ids:
        log.info("nothing to do (all requested stages already complete).")
        write_run_state(spec, all_stage_ids, gpu=True, extra={"in_progress": False})
        if drive_mirror is not None:
            drive_mirror.stop(final_push=True)
        return 0

    if not args.skip_prepare:
        log.info("preparing stage assets (BIM + mesh + sidecar) ...")
        prepared = prepare_stage_assets(spec, stage_ids, log)
    else:
        prepared = {}
        for sid in stage_ids:
            prepared[sid] = {
                "glb": P.stage_glb_path(spec, sid),
                "elements_json": P.stage_elements_json_path(spec, sid),
            }

    # Dataset-level CSVs (schedule + mapping). These are deterministic
    # and small, regenerate every run.
    elements = build_layout(spec)
    write_dataset_schedule(spec, elements, P.dataset_schedule_csv(spec))
    write_bim_schedule_mapping(spec, elements, P.dataset_bim_schedule_mapping_csv(spec))

    # Resolve Blender AFTER preparation so a dry-run that just wants the
    # GLB + sidecar JSON outputs (e.g. for CI / asset inspection) still
    # gets useful artefacts even on a machine without Blender installed.
    try:
        blender_exe = _resolve_blender(args.blender)
        log.info("blender      : %s", blender_exe)
    except FileNotFoundError as e:
        log.error("%s", e)
        log.info("preparation finished; rendering skipped because Blender is not available.")
        write_run_state(spec, all_stage_ids, gpu=True, extra={"in_progress": False})
        if drive_mirror is not None:
            drive_mirror.stop(final_push=True)
        return 0

    blender_renders = P.blender_renders_root(spec)
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
        stage_dir = P.stage_blender_render_dir(spec, sid)
        rc = render_stage_with_blender(
            blender_exe=blender_exe,
            stage_id=sid,
            glb_path=Path(info["glb"]),
            elements_json=Path(info["elements_json"]),
            output_dir=stage_dir,
            args=args,
            fps=fps,
            frames=frames,
            width=width,
            height=height,
            samples=samples,
            motion_blur=preset.motion_blur,
            log=log,
        )
        if rc != 0:
            log.error("stage %d failed; continuing with the rest.", sid)
            if args.strict_render:
                log.error("[strict] aborting because Blender returned non-zero rc.")
                return 7
            continue

        # Strict-render guard: count actual frames written by Blender.
        sub = P.stage_blender_subdirs(spec, sid)
        actual_rgb = len(list(sub["rgb"].glob("frame_*.png"))) if sub["rgb"].exists() else 0
        if actual_rgb < frames:
            msg = (f"stage {sid}: Blender wrote {actual_rgb} RGB frames, "
                   f"expected {frames}")
            if args.strict_render:
                log.error("[strict] %s", msg)
                return 7
            log.warning("[non-strict] %s", msg)

        video_path = None
        if not args.skip_encode:
            video_path = encode_mp4(
                stage_dir / "rgb",
                P.stage_video_path(spec, sid, gpu=True),
                fps=fps,
                log=log,
            )
            if args.strict_render and video_path is None:
                log.error("[strict] stage %d: encode_mp4 returned no file", sid)
                return 7

        # Guard against the "nothing but light" failure: a frame that is
        # almost entirely near-white / has no spatial structure means the
        # geometry was off-frame or the scene was blown out.
        quality = check_frame_quality(stage_dir / "rgb", log)
        if args.strict_render and quality is not None and not quality["ok"]:
            log.error("[strict] stage %d: render looks blank (%s)", sid, quality)
            return 7

        # Package the saved .blend project as a downloadable zip.
        blend_zip = None
        if args.save_blend:
            blend_zip = zip_blend_project(spec, sid, stage_dir, log)

        files_by_stage[sid] = {
            "glb": Path(info["glb"]),
            "elements_json": Path(info["elements_json"]),
            "blender_dir": stage_dir,
            "rgb_dir": sub["rgb"],
            "depth_dir": sub["depth"],
            "seg_dir": sub["seg"],
            "camera_path": sub["camera_path"],
            "video": video_path,
            "element_metrics_csv": info.get("element_metrics_csv"),
        }

        # Write the .done marker so future --resume runs skip this stage.
        write_done_marker(spec, sid, payload={
            "preset": preset_name,
            "rgb_frame_count": actual_rgb,
            "expected_frames": frames,
            "video": str(video_path) if video_path else None,
            "samples": samples,
            "resolution": [width, height],
            "motion_blur": preset.motion_blur,
            "strict_render": bool(args.strict_render),
            "frame_quality": quality,
            "blend_zip": str(blend_zip) if blend_zip else None,
        }, gpu=True)

        # Persist progress to Drive immediately (per-stage), plus a portable
        # run-state manifest so the run can be resumed from another device.
        write_run_state(spec, all_stage_ids, gpu=True,
                        extra={"preset": preset_name, "in_progress": True})
        if drive_mirror is not None:
            try:
                drive_mirror.push()
            except Exception as exc:  # pragma: no cover
                log.warning("drive push after stage %d failed: %s", sid, exc)

        if ProgressBar:
            pb.update(1)

    # Refresh manifest so the new files show up
    extras = {
        "schedule_csv": P.dataset_schedule_csv(spec),
        "bim_schedule_mapping_csv": P.dataset_bim_schedule_mapping_csv(spec),
    }
    manifest_path = P.dataset_manifest_path(spec, gpu=True)
    write_manifest(spec, files_by_stage, extras, manifest_path)
    log.info("manifest -> %s", manifest_path)

    # Final portable run-state + Drive flush so everything is on Drive.
    write_run_state(spec, all_stage_ids, gpu=True,
                    extra={"preset": preset_name, "in_progress": False})
    if drive_mirror is not None:
        try:
            drive_mirror.stop(final_push=True)
            log.info("drive: final sync complete -> %s", drive_mirror.drive_root)
        except Exception as exc:  # pragma: no cover
            log.warning("final drive push failed: %s", exc)
    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
