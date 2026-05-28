"""GPU-accelerated Blender renderer for the 7-stage synthetic floor.

Run this script INSIDE Blender (not as a regular Python script):

    blender -b --python blender_gpu_renderer.py -- \\
        --input-mesh path/to/stage_07.glb \\
        --elements-json path/to/stage_07_elements.json \\
        --output-dir renders/stage_07 \\
        --stage-id 7 \\
        --frames 120 --fps 30 --samples 128 \\
        --resolution 1280 720 \\
        --sun-elevation 38 --sun-azimuth 135

Designed for Google Colab T4/A100 with Cycles + OptiX/CUDA + OpenImageDenoise.
No external texture asset downloads are required: every PBR material is
generated procedurally from Principled BSDF + Noise/Voronoi nodes, and
the environment uses Blender's built-in Nishita Sky Texture.

Outputs (created under ``--output-dir``)::

    rgb/frame_NNNN.png             sRGB rendered images
    depth/frame_NNNN.exr           32-bit float depth in metres (OpenEXR)
    seg/frame_NNNN.png             uint16 segmentation mask (Object Index pass)
    seg_palette.png                colour palette preview of seg classes
    camera_path.json               per-frame intrinsics + 4x4 cam-to-world

The script is intentionally one big file so it can be copy-pasted into
Colab without any package machinery. It is also strict about logging
to stdout (Colab will display it inline).

Notes on the design:

* GPU activation prefers OPTIX (T4/A100 supported in Blender 4.x), then
  CUDA, then CPU as a last resort. Any device of the chosen type is
  enabled before rendering starts.
* The Cycles ``persistent_data`` flag is enabled so per-frame setup is
  amortised across the animation.
* Materials are looked up by element ``category`` (read from the
  sidecar JSON written by the host pipeline), with the per-element
  ``finishing`` mixed in for the late stages (painted, plaster, etc.).
* Segmentation uses the Object Index render pass plus per-category
  ``ID Mask`` File Output nodes. Each category gets a stable integer
  index so the mask is comparable across stages.
* The handheld camera path is a Bezier curve traced through a fixed
  set of waypoints inside the building. The camera follows the curve
  via a Follow Path constraint, looks at a target empty (Track To
  constraint), and is given two layers of Noise F-Curve modifiers on
  rotation for natural micro-jitter and slow human sway.
* Motion blur is enabled on Cycles to mimic a real smartphone shutter.

Schema written to ``camera_path.json`` matches the existing
``synthetic_floor_camera_path.v1`` contract from
``metadata_exporter.py`` so downstream consumers (COLMAP feeders, etc.)
do not need to know which renderer produced the frames.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# These imports are only available inside Blender. We import them lazily
# from main() so that a host-side syntax check (python -m py_compile)
# can still parse the file even if bpy is not installed.
bpy = None  # type: ignore[assignment]
mathutils = None  # type: ignore[assignment]


# ---------------------------------------------------------------------
# Logging helper (Blender swallows logging.* output sometimes)
# ---------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[blender_gpu_renderer] {msg}", flush=True)


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------


@dataclass
class RenderArgs:
    input_mesh: Path
    elements_json: Path | None
    output_dir: Path
    stage_id: int
    frames: int
    fps: int
    samples: int
    width: int
    height: int
    sun_elevation_deg: float
    sun_azimuth_deg: float
    seed: int
    device: str  # OPTIX | CUDA | CPU
    motion_blur: bool
    sky_only_below_stage: int  # stages 1..N use only sky no ceiling


def _parse_args() -> RenderArgs:
    """Parse args from sys.argv after Blender's ``--`` separator."""
    import argparse

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    p = argparse.ArgumentParser(description="GPU Blender renderer for synthetic_floor_7stage")
    p.add_argument("--input-mesh", required=True, type=Path,
                   help="Path to .glb / .gltf / .obj input mesh.")
    p.add_argument("--elements-json", type=Path, default=None,
                   help="Sidecar JSON with element_id -> category mapping.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--stage-id", required=True, type=int, choices=range(1, 8))
    p.add_argument("--frames", type=int, default=120,
                   help="Total frames (default: 120 = 4s @ 30fps).")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--samples", type=int, default=128,
                   help="Cycles samples per pixel (128 fast, 256 high quality).")
    p.add_argument("--resolution", nargs=2, type=int, default=[1280, 720],
                   metavar=("W", "H"))
    p.add_argument("--sun-elevation", type=float, default=38.0)
    p.add_argument("--sun-azimuth", type=float, default=135.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="OPTIX",
                   choices=("OPTIX", "CUDA", "CPU"))
    p.add_argument("--no-motion-blur", action="store_true")
    p.add_argument("--sky-only-below-stage", type=int, default=4,
                   help="For stages strictly below this id we keep the ceiling "
                        "hidden so the open sky is visible (1..7).")
    a = p.parse_args(argv)

    return RenderArgs(
        input_mesh=a.input_mesh,
        elements_json=a.elements_json,
        output_dir=a.output_dir,
        stage_id=a.stage_id,
        frames=a.frames,
        fps=a.fps,
        samples=a.samples,
        width=int(a.resolution[0]),
        height=int(a.resolution[1]),
        sun_elevation_deg=float(a.sun_elevation),
        sun_azimuth_deg=float(a.sun_azimuth),
        seed=int(a.seed),
        device=a.device,
        motion_blur=not a.no_motion_blur,
        sky_only_below_stage=int(a.sky_only_below_stage),
    )


# ---------------------------------------------------------------------
# Scene reset + GPU bootstrap
# ---------------------------------------------------------------------


def reset_scene() -> None:
    """Wipe the default cube/light/camera and start clean."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Drop every collection just in case factory_settings kept something
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)


def configure_gpu(device_pref: str) -> str:
    """Enable Cycles GPU. Returns the actual backend used."""
    prefs = bpy.context.preferences.addons["cycles"].preferences

    order = ["OPTIX", "CUDA", "CPU"] if device_pref == "OPTIX" else [device_pref, "CUDA", "CPU"]
    chosen: str | None = None
    for backend in order:
        try:
            prefs.compute_device_type = backend
        except TypeError:
            # Backend not available in this build (e.g. OPTIX on machines without an NVIDIA card)
            continue
        prefs.get_devices()
        usable = [d for d in prefs.devices if d.type == backend]
        if backend == "CPU":
            chosen = "CPU"
            break
        if usable:
            chosen = backend
            break

    if chosen is None:
        chosen = "CPU"
        prefs.compute_device_type = "CPU"
        prefs.get_devices()

    # Enable everything of the chosen type, plus CPU as helper if a GPU is available.
    enabled_devices: list[str] = []
    for d in prefs.devices:
        if d.type == chosen or (chosen != "CPU" and d.type == "CPU"):
            d.use = True
            enabled_devices.append(f"{d.name} ({d.type})")
        else:
            d.use = False

    bpy.context.scene.cycles.device = "GPU" if chosen != "CPU" else "CPU"
    _log(f"GPU backend: {chosen}")
    for name in enabled_devices:
        _log(f"  enabled device: {name}")
    return chosen


# ---------------------------------------------------------------------
# Mesh import
# ---------------------------------------------------------------------


def import_mesh(path: Path) -> list:
    """Import GLB / GLTF / OBJ. Returns the list of imported mesh objects."""
    before = set(bpy.data.objects)
    suffix = path.suffix.lower()
    if suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path), merge_vertices=False)
    elif suffix == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=str(path))
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=str(path))
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported mesh format: {path}")

    new_objs = [o for o in bpy.data.objects if o not in before]
    mesh_objs = [o for o in new_objs if o.type == "MESH"]
    _log(f"Imported {len(mesh_objs)} mesh objects from {path.name}")
    return mesh_objs


def map_objects_to_elements(mesh_objs: list, elements_json: Path | None) -> dict:
    """Return ``{object_name: element_record}``.

    The host pipeline writes ``elements_json`` next to the GLB. Object
    names in Blender after a GLB import usually match the trimesh
    ``geom_name`` (which we set to the element_id) or ``node_name``
    (which we set to the IFC GlobalId). We try both.
    """
    if elements_json is None or not Path(elements_json).exists():
        _log("WARNING: no elements_json sidecar; using fallback category 'unknown' for all parts.")
        return {o.name: {"category": "unknown", "finishing": "raw_concrete", "element_id": o.name}
                for o in mesh_objs}

    payload = json.loads(Path(elements_json).read_text())
    by_id = {e["element_id"]: e for e in payload["elements"]}
    by_guid = {e["ifc_global_id"]: e for e in payload["elements"]}

    out: dict = {}
    for o in mesh_objs:
        name = o.name
        record = by_id.get(name) or by_guid.get(name)
        if record is None:
            # Blender appends ".001", ".002" etc. for duplicates and may
            # split nested names. Try a stripped match.
            base = name.split(".")[0]
            record = by_id.get(base) or by_guid.get(base)
        if record is None:
            # Final fallback: try to substring-match any element_id.
            for k, v in by_id.items():
                if k in name or name in k:
                    record = v
                    break
        if record is None:
            record = {"category": "unknown", "finishing": "raw_concrete", "element_id": name}
        out[o.name] = record
    matched = sum(1 for v in out.values() if v.get("category") != "unknown")
    _log(f"Element mapping: matched {matched}/{len(mesh_objs)} objects to categories.")
    return out


# ---------------------------------------------------------------------
# Materials (procedural PBR)
# ---------------------------------------------------------------------


# Stable integer index per category, used as IndexOB for segmentation.
CATEGORY_PASS_INDEX = {
    "slab": 1,
    "columns": 2,
    "exterior_walls": 3,
    "interior_walls": 4,
    "windows": 5,
    "doors": 6,
    "ceiling": 7,
    "floor_finish": 8,
    "baseboards": 9,
    "fixtures": 10,
    "unknown": 99,
}


# Per-category PBR presets. `(base_rgba, roughness, metallic, transmission, noise_scale, noise_strength)`
CATEGORY_PRESETS = {
    "slab":           {"color": (0.62, 0.62, 0.62, 1.0), "roughness": 0.92, "metallic": 0.0, "noise_scale": 12.0, "bump": 0.10},
    "columns":        {"color": (0.66, 0.66, 0.66, 1.0), "roughness": 0.90, "metallic": 0.0, "noise_scale": 14.0, "bump": 0.08},
    "exterior_walls": {"color": (0.78, 0.74, 0.70, 1.0), "roughness": 0.88, "metallic": 0.0, "noise_scale": 18.0, "bump": 0.20},
    "interior_walls": {"color": (0.94, 0.92, 0.89, 1.0), "roughness": 0.82, "metallic": 0.0, "noise_scale": 30.0, "bump": 0.05},
    "windows":        {"color": (0.78, 0.86, 0.95, 0.4), "roughness": 0.05, "metallic": 0.0, "noise_scale": 20.0, "bump": 0.0, "transmission": 0.85, "ior": 1.45},
    "doors":          {"color": (0.55, 0.40, 0.30, 1.0), "roughness": 0.55, "metallic": 0.0, "noise_scale": 8.0,  "bump": 0.10},
    "ceiling":        {"color": (0.95, 0.95, 0.93, 1.0), "roughness": 0.85, "metallic": 0.0, "noise_scale": 24.0, "bump": 0.04},
    "floor_finish":   {"color": (0.85, 0.83, 0.78, 1.0), "roughness": 0.30, "metallic": 0.0, "noise_scale": 35.0, "bump": 0.02},
    "baseboards":     {"color": (0.92, 0.92, 0.92, 1.0), "roughness": 0.45, "metallic": 0.0, "noise_scale": 10.0, "bump": 0.04},
    "fixtures":       {"color": (0.95, 0.95, 0.95, 1.0), "roughness": 0.40, "metallic": 0.0, "noise_scale": 5.0,  "bump": 0.02},
    "unknown":        {"color": (0.70, 0.70, 0.70, 1.0), "roughness": 0.80, "metallic": 0.0, "noise_scale": 10.0, "bump": 0.05},
}


# Finishing modifies the preset (late stages -> smoother, more colourful).
FINISHING_MODIFIERS = {
    "raw_concrete":  {"roughness_add": 0.00, "lightness": 1.00},
    "rough_plaster": {"roughness_add": -0.05, "lightness": 1.05},
    "fine_plaster":  {"roughness_add": -0.20, "lightness": 1.10},
    "painted":       {"roughness_add": -0.30, "lightness": 1.12, "color_blend": (0.96, 0.95, 0.92, 1.0), "color_blend_w": 0.7},
    "tile":          {"roughness_add": -0.40, "lightness": 1.05},
    "raw_wood":      {"roughness_add": -0.10, "lightness": 0.95, "color_blend": (0.55, 0.40, 0.30, 1.0), "color_blend_w": 0.5},
    "painted_wood":  {"roughness_add": -0.25, "lightness": 1.08, "color_blend": (0.92, 0.92, 0.92, 1.0), "color_blend_w": 0.7},
}


def _make_material(name: str, category: str, finishing: str, seed: int):
    """Create a Principled-BSDF material with procedural surface variation."""
    preset = dict(CATEGORY_PRESETS.get(category, CATEGORY_PRESETS["unknown"]))
    mod = FINISHING_MODIFIERS.get(finishing, {})

    base_rgba = list(preset["color"])
    blend = mod.get("color_blend")
    if blend is not None:
        w = float(mod.get("color_blend_w", 0.5))
        base_rgba = [(1 - w) * a + w * b for a, b in zip(base_rgba, list(blend))]
    light = float(mod.get("lightness", 1.0))
    base_rgba = [min(1.0, c * light) for c in base_rgba[:3]] + [base_rgba[3]]
    roughness = max(0.02, min(1.0, preset["roughness"] + float(mod.get("roughness_add", 0.0))))

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    # Output + Principled BSDF
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = base_rgba
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = float(preset.get("metallic", 0.0))
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.5
    if "IOR" in bsdf.inputs and "ior" in preset:
        bsdf.inputs["IOR"].default_value = float(preset["ior"])
    if "Transmission Weight" in bsdf.inputs and "transmission" in preset:
        bsdf.inputs["Transmission Weight"].default_value = float(preset["transmission"])

    # Procedural variation: Noise -> ColorRamp -> mix into base color
    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Location"].default_value = (
        ((seed * 13) % 100) * 0.01,
        ((seed * 7) % 100) * 0.01,
        ((seed * 11) % 100) * 0.01,
    )
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = float(preset["noise_scale"])
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.6
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[1].position = 0.65
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = 0.45
    mix.inputs["Color1"].default_value = tuple(base_rgba)

    nt.links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], mix.inputs["Color2"])
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])

    # Surface bump (cheap normal variation without an external normal map)
    bump_strength = float(preset.get("bump", 0.0))
    if bump_strength > 1e-3:
        voronoi = nt.nodes.new("ShaderNodeTexVoronoi")
        voronoi.inputs["Scale"].default_value = max(2.0, float(preset["noise_scale"]) * 0.5)
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = bump_strength
        bump.inputs["Distance"].default_value = 0.05
        nt.links.new(mapping.outputs["Vector"], voronoi.inputs["Vector"])
        nt.links.new(voronoi.outputs["Distance"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Glass: enable alpha blending on the slot
    if category == "windows":
        mat.blend_method = "BLEND" if hasattr(mat, "blend_method") else mat.blend_method
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = 0.4

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def assign_materials(mesh_objs: list, mapping: dict, seed: int) -> dict:
    """Assign one procedural material per (category, finishing) pair.

    Also sets ``pass_index`` on each object for segmentation rendering.
    Returns the cache of materials created (for inspection/debug).
    """
    cache: dict = {}
    for obj in mesh_objs:
        rec = mapping.get(obj.name, {"category": "unknown", "finishing": "raw_concrete"})
        cat = rec.get("category", "unknown")
        fin = rec.get("finishing", "raw_concrete")
        key = f"{cat}__{fin}"
        if key not in cache:
            cache[key] = _make_material(f"mat_{key}", cat, fin, seed=seed + abs(hash(key)) % 1000)
        # Replace any imported materials
        obj.data.materials.clear()
        obj.data.materials.append(cache[key])
        obj.pass_index = CATEGORY_PASS_INDEX.get(cat, 99)
    _log(f"Assigned {len(cache)} unique procedural materials across {len(mesh_objs)} objects.")
    return cache


# ---------------------------------------------------------------------
# Environment (Nishita Sky + Sun light)
# ---------------------------------------------------------------------


def setup_world_sky(sun_elevation_deg: float, sun_azimuth_deg: float) -> None:
    """Configure the World shader with a procedural Nishita Sky.

    For early stages (no ceiling) this gives a realistic open sky.
    """
    world = bpy.data.worlds.new("World") if "World" not in bpy.data.worlds else bpy.data.worlds["World"]
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.0
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(sun_elevation_deg)
    sky.sun_rotation = math.radians(sun_azimuth_deg)
    sky.sun_disc = True
    sky.sun_size = math.radians(0.55)
    sky.sun_intensity = 1.0
    sky.air_density = 1.0
    sky.dust_density = 1.0
    sky.ozone_density = 1.0
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # Add a real Sun light too -- gives crisper shadows than the sky alone.
    sun_data = bpy.data.lights.new("SunLight", type="SUN")
    sun_data.energy = 4.0
    sun_data.angle = math.radians(0.5)
    sun_obj = bpy.data.objects.new("SunLight", sun_data)
    bpy.context.collection.objects.link(sun_obj)
    az = math.radians(sun_azimuth_deg)
    el = math.radians(sun_elevation_deg)
    # Point sun direction along (cos(el)*cos(az), cos(el)*sin(az), sin(el))
    sun_obj.rotation_euler = (math.pi / 2 - el, 0.0, az + math.pi / 2)


def hide_ceiling_for_early_stages(mesh_objs: list, mapping: dict, stage_id: int, sky_only_below_stage: int) -> int:
    """Disable the ceiling in early stages so the open sky is visible.

    Returns the number of hidden objects.
    """
    hidden = 0
    if stage_id < sky_only_below_stage:
        for o in mesh_objs:
            cat = mapping.get(o.name, {}).get("category")
            if cat == "ceiling":
                o.hide_render = True
                hidden += 1
    return hidden


# ---------------------------------------------------------------------
# Camera path
# ---------------------------------------------------------------------


def _scene_bbox(mesh_objs: list) -> tuple:
    """Compute the axis-aligned bbox of all mesh objects (world-space)."""
    bx_min = [math.inf, math.inf, math.inf]
    bx_max = [-math.inf, -math.inf, -math.inf]
    for obj in mesh_objs:
        if obj.type != "MESH":
            continue
        for v in obj.bound_box:
            wp = obj.matrix_world @ mathutils.Vector(v)
            for i in range(3):
                bx_min[i] = min(bx_min[i], wp[i])
                bx_max[i] = max(bx_max[i], wp[i])
    return tuple(bx_min), tuple(bx_max)


def build_camera(mesh_objs: list, frames: int, fps: int, args: RenderArgs):
    """Build a Bezier camera path that sweeps through the room.

    Returns the camera object and the path object so the caller can
    keyframe them if desired.
    """
    (xmn, ymn, zmn), (xmx, ymx, zmx) = _scene_bbox(mesh_objs)
    cx, cy = 0.5 * (xmn + xmx), 0.5 * (ymn + ymx)
    L, W = xmx - xmn, ymx - ymn
    hold_z = zmn + 1.55  # human eye height above floor

    # Bezier waypoints (X, Y, Z). Designed for an 18x12 floor; scales to the actual bbox.
    waypoints = [
        (xmn + L * 0.50, ymn + W * 0.10, hold_z),  # entrance, near south facade
        (xmn + L * 0.50, ymn + W * 0.45, hold_z),
        (xmn + L * 0.18, ymn + W * 0.50, hold_z),  # peek into Office A
        (xmn + L * 0.18, ymn + W * 0.30, hold_z),
        (xmn + L * 0.18, ymn + W * 0.50, hold_z),
        (xmn + L * 0.50, ymn + W * 0.50, hold_z),
        (xmn + L * 0.83, ymn + W * 0.50, hold_z),
        (xmn + L * 0.83, ymn + W * 0.78, hold_z),  # peek into Office E
        (xmn + L * 0.50, ymn + W * 0.50, hold_z),
        (xmn + L * 0.50, ymn + W * 0.10, hold_z),  # back near entrance
    ]

    # --- Bezier curve ----------------------------------------------------
    curve_data = bpy.data.curves.new("CamPath", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.path_duration = frames
    spline = curve_data.splines.new(type="BEZIER")
    spline.bezier_points.add(len(waypoints) - 1)
    for i, pt in enumerate(waypoints):
        bp = spline.bezier_points[i]
        bp.co = pt
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    path_obj = bpy.data.objects.new("CamPath", curve_data)
    bpy.context.collection.objects.link(path_obj)

    # --- Camera + Track-To target ---------------------------------------
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 18.0  # ~28mm equivalent on full frame
    if args.motion_blur:
        cam_data.dof.use_dof = False  # motion blur is handled at scene level
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    target = bpy.data.objects.new("CamTarget", None)
    target.empty_display_type = "PLAIN_AXES"
    target.empty_display_size = 0.2
    bpy.context.collection.objects.link(target)
    target.location = (cx, cy, hold_z - 0.05)

    # Track to target for natural look-at
    tcon = cam_obj.constraints.new(type="TRACK_TO")
    tcon.target = target
    tcon.track_axis = "TRACK_NEGATIVE_Z"
    tcon.up_axis = "UP_Y"

    # Follow path
    fcon = cam_obj.constraints.new(type="FOLLOW_PATH")
    fcon.target = path_obj
    fcon.use_curve_follow = False
    fcon.forward_axis = "TRACK_NEGATIVE_Y"
    fcon.up_axis = "UP_Z"

    # Animate evaluation_time on the curve (controls path progress 0..frames)
    curve_data.use_path = True
    curve_data.path_duration = frames

    # Animate the Bezier "evaluation_time" by setting two keys on the curve's
    # `eval_time` (via path_duration the value is in frames).
    curve_data.animation_data_create()
    action = bpy.data.actions.new("CamPathAction")
    curve_data.animation_data.action = action
    fc = action.fcurves.new(data_path="eval_time")
    kp1 = fc.keyframe_points.insert(frame=1, value=0.0)
    kp2 = fc.keyframe_points.insert(frame=frames, value=float(frames))
    for kp in (kp1, kp2):
        kp.interpolation = "LINEAR"

    # Animate the target slightly so the camera tilts/pans naturally.
    target.animation_data_create()
    target_action = bpy.data.actions.new("CamTargetAction")
    target.animation_data.action = target_action
    rng = random.Random(args.seed)
    n_target_keys = 8
    for axis_idx, axis_name in enumerate(("location_x", "location_y", "location_z")):
        # We use plain location keyframes via insert_keyframe is not exposed
        # easily; instead build f-curves directly.
        path = "location"
        fcurve = target_action.fcurves.new(data_path=path, index=axis_idx)
        for k in range(n_target_keys + 1):
            f = 1 + int(round(k * (frames - 1) / n_target_keys))
            base = (cx, cy, hold_z - 0.05)[axis_idx]
            jitter = rng.uniform(-0.6, 0.6) if axis_idx < 2 else rng.uniform(-0.10, 0.10)
            kp = fcurve.keyframe_points.insert(frame=f, value=base + jitter)
            kp.interpolation = "BEZIER"

    # Add Noise F-Curve modifiers on the camera's rotation for natural sway
    # plus tiny location wobble (handheld feel).
    cam_obj.animation_data_create()
    cam_action = bpy.data.actions.new("CamAction")
    cam_obj.animation_data.action = cam_action
    # Insert a placeholder keyframe on rotation_euler so f-curves exist.
    cam_obj.rotation_mode = "XYZ"
    for i in range(3):
        fc = cam_action.fcurves.new(data_path="rotation_euler", index=i)
        kp = fc.keyframe_points.insert(frame=1, value=0.0)
        kp.interpolation = "LINEAR"
        kp = fc.keyframe_points.insert(frame=frames, value=0.0)
        kp.interpolation = "LINEAR"
        # slow sway
        m1 = fc.modifiers.new(type="NOISE")
        m1.scale = 80.0
        m1.strength = math.radians(0.6)
        m1.phase = (args.seed % 50) + i * 7
        # high-frequency micro tremor
        m2 = fc.modifiers.new(type="NOISE")
        m2.scale = 6.0
        m2.strength = math.radians(0.18)
        m2.phase = (args.seed % 50) + i * 13 + 100
    # Tiny location jitter as well
    for i in range(3):
        fc = cam_action.fcurves.new(data_path="location", index=i)
        kp = fc.keyframe_points.insert(frame=1, value=0.0)
        kp.interpolation = "LINEAR"
        kp = fc.keyframe_points.insert(frame=frames, value=0.0)
        kp.interpolation = "LINEAR"
        m = fc.modifiers.new(type="NOISE")
        m.scale = 30.0
        m.strength = 0.012  # 1.2 cm RMS
        m.phase = (args.seed % 50) + i * 19 + 200

    bpy.context.scene.camera = cam_obj
    return cam_obj, path_obj, target


# ---------------------------------------------------------------------
# Render settings + Compositor (RGB + Depth + Segmentation)
# ---------------------------------------------------------------------


def configure_render(args: RenderArgs) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.fps = args.fps
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    # Cycles
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    scene.cycles.use_persistent_data = True
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 6
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 6

    # Motion blur
    scene.render.use_motion_blur = bool(args.motion_blur)
    scene.cycles.motion_blur_position = "CENTER"
    scene.render.motion_blur_shutter = 0.5

    # Frames
    scene.frame_start = 1
    scene.frame_end = args.frames

    # Color management: Filmic gives natural smartphone-like look
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium Contrast"
    scene.view_settings.exposure = 0.0


def configure_compositor(args: RenderArgs) -> None:
    """Wire up RGB, Depth (EXR), and Segmentation File Output nodes."""
    scene = bpy.context.scene
    scene.use_nodes = True

    # Enable required render passes
    vl = scene.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_object_index = True
    vl.use_pass_combined = True

    nt = scene.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    rl = nt.nodes.new("CompositorNodeRLayers")

    out_dir = Path(args.output_dir)
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    seg_dir = out_dir / "seg"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    # --- RGB (PNG, 8 bit) ----------------------------------------------------
    fo_rgb = nt.nodes.new("CompositorNodeOutputFile")
    fo_rgb.label = "RGB Output"
    fo_rgb.base_path = str(rgb_dir)
    fo_rgb.file_slots[0].path = "frame_"
    fo_rgb.format.file_format = "PNG"
    fo_rgb.format.color_mode = "RGB"
    fo_rgb.format.color_depth = "8"
    nt.links.new(rl.outputs["Image"], fo_rgb.inputs[0])

    # --- Depth (OpenEXR, 32 bit float, in metres) ---------------------------
    fo_depth = nt.nodes.new("CompositorNodeOutputFile")
    fo_depth.label = "Depth Output"
    fo_depth.base_path = str(depth_dir)
    fo_depth.file_slots[0].path = "frame_"
    fo_depth.format.file_format = "OPEN_EXR"
    fo_depth.format.color_mode = "RGB"  # Blender 4.2+ does not support "BW" on EXR File Output
    fo_depth.format.color_depth = "32"
    nt.links.new(rl.outputs["Depth"], fo_depth.inputs[0])

    # --- Segmentation (PNG, 16 bit) ----------------------------------------
    # Normalise IndexOB by 65535 so it fits a 16-bit grayscale PNG. The
    # downstream consumer recovers the integer index by multiplying back.
    div = nt.nodes.new("CompositorNodeMath")
    div.operation = "DIVIDE"
    div.inputs[1].default_value = 65535.0
    nt.links.new(rl.outputs["IndexOB"], div.inputs[0])

    fo_seg = nt.nodes.new("CompositorNodeOutputFile")
    fo_seg.label = "Seg Output"
    fo_seg.base_path = str(seg_dir)
    fo_seg.file_slots[0].path = "frame_"
    fo_seg.format.file_format = "PNG"
    fo_seg.format.color_mode = "RGB"  # Blender 4.2+ does not support "BW" on PNG File Output
    fo_seg.format.color_depth = "16"
    nt.links.new(div.outputs[0], fo_seg.inputs[0])


# ---------------------------------------------------------------------
# Camera intrinsics + path JSON dump
# ---------------------------------------------------------------------


def _camera_intrinsics(cam_obj, scene) -> dict:
    cam = cam_obj.data
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    sensor_w = cam.sensor_width
    fx = (cam.lens / sensor_w) * w
    fy = fx  # square pixels
    cx = w / 2.0
    cy = h / 2.0
    fov_h = 2 * math.atan(0.5 * sensor_w / cam.lens)
    return {
        "width_px": w,
        "height_px": h,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "horizontal_fov_deg": math.degrees(fov_h),
        "lens_mm": cam.lens,
        "sensor_width_mm": sensor_w,
    }


def dump_camera_path(cam_obj, args: RenderArgs, out_path: Path) -> None:
    """Evaluate the camera at every frame and dump a JSON in the same
    schema as the CPU pipeline's ``synthetic_floor_camera_path.v1``."""
    scene = bpy.context.scene
    intr = _camera_intrinsics(cam_obj, scene)
    deps = bpy.context.evaluated_depsgraph_get()
    frames = []
    for f in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(f)
        deps.update()
        cam_eval = cam_obj.evaluated_get(deps)
        m = cam_eval.matrix_world
        eye = (m.translation.x, m.translation.y, m.translation.z)
        # Camera looks down -Z in its local frame (Blender convention).
        local_fwd = mathutils.Vector((0.0, 0.0, -1.0))
        fwd_world = (m.to_3x3() @ local_fwd).normalized()
        target = (eye[0] + fwd_world.x, eye[1] + fwd_world.y, eye[2] + fwd_world.z)
        local_up = mathutils.Vector((0.0, 1.0, 0.0))
        up_world = (m.to_3x3() @ local_up).normalized()
        cam_to_world = [list(row) for row in m]
        frames.append({
            "frame_index": f - 1,
            "timestamp_sec": (f - 1) / scene.render.fps,
            "eye_world_m": list(eye),
            "target_world_m": list(target),
            "up_world": [up_world.x, up_world.y, up_world.z],
            "cam_to_world_4x4": cam_to_world,
        })

    payload = {
        "schema_version": "synthetic_floor_camera_path.v1",
        "renderer": "blender_cycles_gpu",
        "stage_id": args.stage_id,
        "fps": args.fps,
        "intrinsics": intr,
        "frames": frames,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# Segmentation palette preview (optional sanity output)
# ---------------------------------------------------------------------


def write_seg_palette(out_path: Path) -> None:
    try:
        import numpy as np
        from PIL import Image
    except Exception as e:
        _log(f"skipping seg palette ({e})")
        return
    palette = np.zeros((20 * len(CATEGORY_PASS_INDEX), 200, 3), dtype=np.uint8)
    rng = np.random.default_rng(123)
    for i, (name, idx) in enumerate(CATEGORY_PASS_INDEX.items()):
        c = (rng.uniform(0.2, 1.0, size=3) * 255).astype(np.uint8)
        palette[i * 20:(i + 1) * 20, :, :] = c
    Image.fromarray(palette).save(out_path)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def main() -> int:
    global bpy, mathutils
    try:
        import bpy as _bpy   # noqa: F401
        import mathutils as _mu  # noqa: F401
    except ImportError as e:
        print(f"ERROR: this script must run inside Blender: {e}", file=sys.stderr)
        return 2
    bpy = _bpy
    mathutils = _mu

    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.output_dir / "blender_render.log"
    sys.stdout = open(log_file, "w", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout

    t0 = time.time()
    _log(f"=== blender_gpu_renderer stage {args.stage_id} ===")
    _log(f"input_mesh   : {args.input_mesh}")
    _log(f"elements_json: {args.elements_json}")
    _log(f"output_dir   : {args.output_dir}")
    _log(f"resolution   : {args.width}x{args.height}")
    _log(f"frames       : {args.frames} @ {args.fps} fps  ({args.frames / args.fps:.1f}s)")
    _log(f"samples      : {args.samples}")
    _log(f"sun          : el={args.sun_elevation_deg}deg az={args.sun_azimuth_deg}deg")

    random.seed(args.seed)

    reset_scene()
    backend = configure_gpu(args.device)
    _log(f"using backend: {backend}")

    mesh_objs = import_mesh(args.input_mesh)
    if not mesh_objs:
        _log("ERROR: no mesh objects imported")
        return 3

    mapping = map_objects_to_elements(mesh_objs, args.elements_json)
    assign_materials(mesh_objs, mapping, seed=args.seed)
    setup_world_sky(args.sun_elevation_deg, args.sun_azimuth_deg)
    n_hidden = hide_ceiling_for_early_stages(mesh_objs, mapping, args.stage_id, args.sky_only_below_stage)
    if n_hidden:
        _log(f"hid {n_hidden} ceiling object(s) so the open sky is visible at stage {args.stage_id}.")

    cam_obj, _path_obj, _target = build_camera(mesh_objs, args.frames, args.fps, args)
    _log("camera built.")

    configure_render(args)
    _log("render settings configured.")

    try:
        configure_compositor(args)
        _log("compositor wired (rgb + depth + seg).")
    except Exception as e:
        import traceback
        _log(f"ERROR in configure_compositor: {e}")
        _log(traceback.format_exc())
        return 5

    # Render the animation
    _log("rendering animation ...")
    try:
        bpy.ops.render.render(animation=True)
    except Exception as e:
        import traceback
        _log(f"ERROR during render: {e}")
        _log(traceback.format_exc())
        return 6
    _log(f"render complete in {time.time() - t0:.1f}s")

    # Dump camera path + palette
    dump_camera_path(cam_obj, args, args.output_dir / "camera_path.json")
    write_seg_palette(args.output_dir / "seg_palette.png")
    _log("camera_path.json written.")

    # Sanity counts
    rgb_count = len(list((args.output_dir / "rgb").glob("frame_*.png")))
    depth_count = len(list((args.output_dir / "depth").glob("frame_*.exr")))
    seg_count = len(list((args.output_dir / "seg").glob("frame_*.png")))
    _log(f"output frame counts: rgb={rgb_count} depth={depth_count} seg={seg_count}")

    if rgb_count == 0:
        _log("ERROR: no RGB frames produced")
        return 4

    _log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
