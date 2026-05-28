#!/usr/bin/env python3
"""End-to-end generator for the 7-stage synthetic floor example.

Run from the repository root:

    PYTHONPATH=examples/synthetic_floor_7stage/src \
        python3 examples/synthetic_floor_7stage/scripts/run_generate.py

Useful flags:
    --config PATH            override the default config
    --stages N1 N2 ...       only generate these stage ids
    --skip-render            skip rendering and video (only BIM + meshes)
    --skip-bim               skip IFC export (helpful when ifcopenshell missing)
    --quick                  small render resolution + short clips for smoke tests
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

# Make the package importable regardless of CWD
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from synthetic_floor.scene_spec import load_scene_spec  # noqa: E402
from synthetic_floor.layout import build_layout, elements_by_category  # noqa: E402
from synthetic_floor.stage_controller import select_for_stage, kept_only  # noqa: E402
from synthetic_floor.materials import build_material_library, save_material_samples  # noqa: E402
from synthetic_floor.ifc_builder import write_stage_ifc  # noqa: E402
from synthetic_floor.mesh_builder import write_stage_meshes  # noqa: E402
from synthetic_floor.camera_path import plan_camera_path  # noqa: E402
from synthetic_floor.renderer import render_stage, render_seg_color  # noqa: E402
from synthetic_floor.smartphone_sim import SmartphoneSimulator  # noqa: E402
from synthetic_floor.video_exporter import write_mp4  # noqa: E402
from synthetic_floor.metadata_exporter import (  # noqa: E402
    write_stage_metadata,
    write_camera_path,
    write_element_metrics_csv,
    write_dataset_schedule,
    write_bim_schedule_mapping,
    write_manifest,
)
from synthetic_floor.validate import (  # noqa: E402
    check_camera_path,
    check_geometry_dimensions,
    check_manifest_consistency,
    check_outputs_exist,
    check_stage_progression,
    check_unique_ifc_guids,
)


def setup_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run_generate.log"
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt, handlers=handlers, force=True)
    return logging.getLogger("synthetic_floor")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=HERE.parent / "config" / "scene.yaml")
    p.add_argument("--stages", type=int, nargs="*", default=None,
                   help="Subset of stage ids to generate (1..7). Default: all.")
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-bim", action="store_true")
    p.add_argument("--skip-mesh", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="Use a small render resolution and short clips for fast smoke tests.")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--duration", type=float, default=None,
                   help="Override per-stage clip duration (seconds).")
    p.add_argument("--save-frames", action="store_true",
                   help="Also save per-frame PNGs alongside the video.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    spec = load_scene_spec(args.config)
    spec.output.ensure()
    log = setup_logging(spec.output.logs, args.log_level)

    # Optional camera overrides
    cam_kwargs: dict = {}
    if args.quick:
        cam_kwargs.update({"width_px": 480, "height_px": 270, "duration_per_stage_sec": 2.0})
    if args.width is not None:
        cam_kwargs["width_px"] = args.width
    if args.height is not None:
        cam_kwargs["height_px"] = args.height
    if args.duration is not None:
        cam_kwargs["duration_per_stage_sec"] = args.duration
    if cam_kwargs:
        spec = replace(spec, camera=replace(spec.camera, **cam_kwargs))

    log.info("Project       : %s (run_id=%s)", spec.project_name, spec.run_id)
    log.info("Config        : %s", spec.config_path)
    log.info("Output root   : %s", spec.output.root)
    log.info("Render res    : %dx%d @ %d fps for %.1fs/stage",
             spec.camera.width_px, spec.camera.height_px,
             spec.camera.fps, spec.camera.duration_per_stage_sec)

    # ---- Build the layout once -----------------------------------------------
    elements = build_layout(spec)
    by_cat = elements_by_category(elements)
    log.info("Layout        : %d elements", len(elements))
    for cat, lst in by_cat.items():
        log.info("  %-16s: %d", cat, len(lst))

    ok_g, m_g = check_geometry_dimensions(spec, elements)
    for line in m_g:
        log.info("[geom] %s", line)
    ok_u, m_u = check_unique_ifc_guids(elements)
    for line in m_u:
        log.info("[guid] %s", line)
    if not (ok_g and ok_u):
        log.error("Geometry checks failed; aborting.")
        return 2

    # ---- Materials -----------------------------------------------------------
    library = build_material_library(seed=spec.random_seed)
    if spec.renderer.emit_material_samples:
        sample_dir = spec.output.root / "material_samples"
        save_material_samples(library, sample_dir)
        log.info("Material samples saved to %s", sample_dir)

    # ---- Per-stage generation ------------------------------------------------
    stage_ids = args.stages or [s.id for s in spec.stages]
    stages_to_run = [s for s in spec.stages if s.id in stage_ids]
    log.info("Stages        : %s", [s.id for s in stages_to_run])

    files_by_stage: dict[int, dict[str, Path]] = {}
    staged_per_stage: dict = {}

    for stage in stages_to_run:
        sid = stage.id
        t0 = time.time()
        log.info("=== Stage %d: %s ===", sid, stage.name)
        staged = select_for_stage(stage, by_cat)
        staged_per_stage[sid] = staged
        kept = kept_only(staged)
        log.info("  kept %d / %d elements", len(kept), len(staged))

        files: dict[str, Path] = {}

        # IFC ----------------------------------------------------------------
        if not args.skip_bim:
            ifc_path = spec.output.bim / f"stage_{sid:02d}.ifc"
            try:
                write_stage_ifc(spec, sid, staged, ifc_path)
                log.info("  IFC      -> %s (%d bytes)", ifc_path, ifc_path.stat().st_size)
                files["ifc"] = ifc_path
            except Exception as e:
                log.exception("  IFC export failed: %s", e)

        # Mesh ---------------------------------------------------------------
        if not args.skip_mesh:
            try:
                meshes = write_stage_meshes(spec, sid, staged, library, spec.output.mesh)
                files["mesh_obj"] = meshes["obj"]
                files["mesh_glb"] = meshes["glb"]
                files["mesh_ply"] = meshes["ply"]
                log.info("  Mesh OBJ -> %s", meshes["obj"])
                log.info("  Mesh GLB -> %s", meshes["glb"])
                log.info("  Mesh PLY -> %s", meshes["ply"])
            except Exception as e:
                log.exception("  Mesh export failed: %s", e)

        # Camera path --------------------------------------------------------
        poses = plan_camera_path(spec, stage_id=sid)
        ok_c, m_c = check_camera_path(poses)
        for line in m_c:
            log.info("[cam ] %s", line)
        if not ok_c:
            log.error("Camera path invalid for stage %d", sid)
            return 3
        cam_json = spec.output.camera / f"stage_{sid:02d}_camera_path.json"
        write_camera_path(spec, sid, poses, cam_json)
        files["camera_path"] = cam_json

        # Render + video -----------------------------------------------------
        if not args.skip_render:
            sim = SmartphoneSimulator(spec.camera, seed=spec.random_seed + sid * 17)
            clean_frames: list = []
            sim_frames: list = []
            depth_arrs: list = []
            seg_arrs: list = []
            t_render = time.time()
            for fi, rgb, depth, seg in render_stage(spec, poses, elements, library, staged):
                clean_frames.append(rgb)
                sim_frames.append(sim(rgb))
                # Save depth/seg only every 10 frames to keep size in check
                if fi % 10 == 0:
                    depth_arrs.append((fi, depth))
                    seg_arrs.append((fi, seg))
                if fi % 30 == 0:
                    log.info("    rendered frame %3d/%d (%.1fs elapsed)", fi + 1, len(poses), time.time() - t_render)
            log.info("  render: %d frames in %.1fs", len(clean_frames), time.time() - t_render)

            if not args.skip_video:
                video_clean = spec.output.video / f"stage_{sid:02d}_clean.mp4"
                video_sim = spec.output.video / f"stage_{sid:02d}.mp4"
                write_mp4(clean_frames, video_clean, fps=spec.camera.fps)
                write_mp4(sim_frames, video_sim, fps=spec.camera.fps)
                files["video"] = video_sim
                files["video_clean"] = video_clean
                log.info("  Video    -> %s", video_sim)
                log.info("  Video    -> %s", video_clean)

            if args.save_frames:
                from synthetic_floor.video_exporter import write_frames_dir
                write_frames_dir(sim_frames, spec.output.renders / f"stage_{sid:02d}", prefix=f"frame")

            # Save sparse depth + seg as .npz (compact)
            if depth_arrs:
                depth_npz = spec.output.depth / f"stage_{sid:02d}_depth.npz"
                np.savez_compressed(
                    depth_npz,
                    frames=np.array([fi for fi, _ in depth_arrs]),
                    depth=np.stack([d for _, d in depth_arrs], axis=0),
                )
                files["depth_npz"] = depth_npz
                log.info("  Depth    -> %s", depth_npz)
            if seg_arrs:
                seg_npz = spec.output.segmentation / f"stage_{sid:02d}_seg.npz"
                np.savez_compressed(
                    seg_npz,
                    frames=np.array([fi for fi, _ in seg_arrs]),
                    seg=np.stack([s for _, s in seg_arrs], axis=0),
                )
                files["segmentation_npz"] = seg_npz
                log.info("  Seg      -> %s", seg_npz)

            # Save a single keyframe PNG for quick eyeballing
            keyframe_png = spec.output.renders / f"stage_{sid:02d}_keyframe.png"
            from PIL import Image
            mid = len(sim_frames) // 2
            Image.fromarray(sim_frames[mid]).save(keyframe_png)
            files["keyframe"] = keyframe_png
            log.info("  Key      -> %s", keyframe_png)

        # Per-stage element metrics CSV (Stage 9 contract)
        elem_csv = spec.output.manifests / f"stage_{sid:02d}_element_metrics.csv"
        write_element_metrics_csv(staged, elem_csv)
        files["element_metrics_csv"] = elem_csv

        # Per-stage metadata JSON
        meta_json = spec.output.manifests / f"stage_{sid:02d}_metadata.json"
        write_stage_metadata(spec, sid, staged, files, meta_json)
        files["metadata_json"] = meta_json

        files_by_stage[sid] = files

        ok_o, m_o = check_outputs_exist(files)
        for line in m_o:
            log.info("[out ] %s", line)
        if not ok_o:
            log.error("Some stage %d outputs are missing.", sid)
            return 4

        log.info("  Stage %d done in %.1fs", sid, time.time() - t0)

    # ---- Dataset-level ------------------------------------------------------
    extras: dict[str, Path] = {}
    schedule_csv = spec.output.manifests / "schedule.csv"
    write_dataset_schedule(spec, elements, schedule_csv)
    extras["schedule_csv"] = schedule_csv

    mapping_csv = spec.output.manifests / "bim_schedule_mapping.csv"
    write_bim_schedule_mapping(spec, elements, mapping_csv)
    extras["bim_schedule_mapping_csv"] = mapping_csv

    manifest = spec.output.manifests / "manifest.json"
    write_manifest(spec, files_by_stage, extras, manifest)
    extras["manifest"] = manifest
    log.info("Manifest -> %s", manifest)

    ok_p, m_p = check_stage_progression(staged_per_stage)
    for line in m_p:
        log.info("[prog] %s", line)
    if not ok_p:
        log.error("Stage progression check failed.")
        return 5

    ok_m, m_m = check_manifest_consistency(manifest)
    for line in m_m:
        log.info("[mfst] %s", line)
    if not ok_m:
        log.error("Manifest consistency check failed.")
        return 6

    log.info("All done. Outputs under: %s", spec.output.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
