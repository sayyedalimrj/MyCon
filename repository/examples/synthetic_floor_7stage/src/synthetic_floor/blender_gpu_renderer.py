"""GPU-accelerated Blender renderer for the 7-stage synthetic floor.

Run INSIDE Blender::

    blender -b --python blender_gpu_renderer.py -- \
        --input-mesh stage_07.glb --elements-json stage_07_elements.json \
        --output-dir renders/stage_07 --stage-id 7 \
        --frames 900 --fps 30 --samples 128 --resolution 1280 720

Key design for REALISTIC INTERIOR rendering:
- Window openings are REAL GAPS in the wall geometry (light passes through)
- Light Portals placed at every window opening (guides Cycles sampling)
- 6 recessed Area Lights for stage 7 (ceiling fixtures)
- High bounce counts (12/8/4/12) for proper indirect illumination
- Nishita Sky + Sun light for natural daylight
- Exposure compensation for interior brightness
- Camera path covers entire 22x14m room with 30+ waypoints
- Handheld jitter via F-Curve Noise modifiers
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

bpy = None  # type: ignore[assignment]
mathutils = None  # type: ignore[assignment]


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
    device: str
    motion_blur: bool


def _parse_args() -> RenderArgs:
    import argparse
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--input-mesh", required=True, type=Path)
    p.add_argument("--elements-json", type=Path, default=None)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--stage-id", required=True, type=int, choices=range(1, 8))
    p.add_argument("--frames", type=int, default=900)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--resolution", nargs=2, type=int, default=[1280, 720])
    p.add_argument("--sun-elevation", type=float, default=38.0)
    p.add_argument("--sun-azimuth", type=float, default=135.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="OPTIX", choices=("OPTIX", "CUDA", "CPU"))
    p.add_argument("--no-motion-blur", action="store_true")
    a = p.parse_args(argv)
    return RenderArgs(
        input_mesh=a.input_mesh, elements_json=a.elements_json,
        output_dir=a.output_dir, stage_id=a.stage_id,
        frames=a.frames, fps=a.fps, samples=a.samples,
        width=int(a.resolution[0]), height=int(a.resolution[1]),
        sun_elevation_deg=float(a.sun_elevation),
        sun_azimuth_deg=float(a.sun_azimuth),
        seed=int(a.seed), device=a.device, motion_blur=not a.no_motion_blur,
    )



# ---------------------------------------------------------------------
# Scene reset + GPU bootstrap
# ---------------------------------------------------------------------


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)


def configure_gpu(device_pref: str) -> str:
    try:
        from synthetic_floor.blender_compat import activate_gpu
        return activate_gpu(device_pref)
    except ImportError:
        pass
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in [device_pref, "CUDA", "CPU"]:
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        usable = [d for d in prefs.devices if d.type == backend]
        if backend == "CPU" or usable:
            for d in prefs.devices:
                d.use = (d.type == backend) or (backend != "CPU" and d.type == "CPU")
            bpy.context.scene.cycles.device = "GPU" if backend != "CPU" else "CPU"
            _log(f"GPU backend: {backend}")
            return backend
    return "CPU"



# ---------------------------------------------------------------------
# Mesh import + element mapping
# ---------------------------------------------------------------------


def import_mesh(path: Path) -> list:
    before = set(bpy.data.objects)
    suffix = path.suffix.lower()
    if suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path), merge_vertices=False)
    elif suffix == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=str(path))
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=str(path))
    else:
        raise ValueError(f"Unsupported: {path}")
    new_objs = [o for o in bpy.data.objects if o not in before]
    mesh_objs = [o for o in new_objs if o.type == "MESH"]
    _log(f"Imported {len(mesh_objs)} mesh objects from {path.name}")
    return mesh_objs


def map_objects_to_elements(mesh_objs: list, elements_json: Path | None) -> dict:
    if elements_json is None or not Path(elements_json).exists():
        return {o.name: {"category": "unknown", "finishing": "raw_concrete",
                         "element_id": o.name} for o in mesh_objs}
    payload = json.loads(Path(elements_json).read_text())
    by_id = {e["element_id"]: e for e in payload["elements"]}
    by_guid = {e["ifc_global_id"]: e for e in payload["elements"]}
    out: dict = {}
    for o in mesh_objs:
        name = o.name
        record = by_id.get(name) or by_guid.get(name)
        if record is None:
            base = name.split(".")[0]
            record = by_id.get(base) or by_guid.get(base)
        if record is None:
            for k, v in by_id.items():
                if k in name or name in k:
                    record = v
                    break
        if record is None:
            record = {"category": "unknown", "finishing": "raw_concrete",
                      "element_id": name}
        out[o.name] = record
    matched = sum(1 for v in out.values() if v.get("category") != "unknown")
    _log(f"Element mapping: {matched}/{len(mesh_objs)} matched.")
    return out



# ---------------------------------------------------------------------
# Materials — realistic PBR presets for construction stages
# ---------------------------------------------------------------------

CATEGORY_PASS_INDEX = {
    "slab": 1, "columns": 2, "ceiling_slab": 3, "ceiling_finish": 4,
    "overhead_pipes": 5, "sill_north": 6, "sill_east": 7,
    "north_wall": 8, "east_wall": 9, "west_wall": 10, "south_wall": 11,
    "windows": 12, "door": 13, "plaster_left_lower": 14,
    "plaster_left_upper": 15, "plaster_other": 16, "ceiling_lights": 17,
    "unknown": 99,
}

# finishing -> material properties
FINISHING_PRESETS = {
    "raw_concrete": {
        "color": (0.45, 0.44, 0.42, 1.0), "roughness": 0.92,
        "metallic": 0.0, "noise_scale": 8.0, "bump": 0.15,
    },
    "cinderblock": {
        "color": (0.55, 0.52, 0.48, 1.0), "roughness": 0.95,
        "metallic": 0.0, "noise_scale": 25.0, "bump": 0.25,
    },
    "plaster_base": {
        "color": (0.88, 0.87, 0.85, 1.0), "roughness": 0.70,
        "metallic": 0.0, "noise_scale": 40.0, "bump": 0.04,
    },
    "painted_white": {
        "color": (0.95, 0.94, 0.93, 1.0), "roughness": 0.45,
        "metallic": 0.0, "noise_scale": 50.0, "bump": 0.02,
    },
    "metal_grey": {
        "color": (0.42, 0.44, 0.46, 1.0), "roughness": 0.35,
        "metallic": 0.85, "noise_scale": 5.0, "bump": 0.01,
    },
    "window_glazed": {
        "color": (0.85, 0.92, 0.98, 0.3), "roughness": 0.02,
        "metallic": 0.0, "noise_scale": 0.0, "bump": 0.0,
        "transmission": 0.92, "ior": 1.52,
    },
    "light_fixture": {
        "color": (1.0, 0.98, 0.95, 1.0), "roughness": 0.20,
        "metallic": 0.10, "noise_scale": 0.0, "bump": 0.0,
        "emission": 5.0,
    },
    "none": {
        "color": (0.5, 0.5, 0.5, 1.0), "roughness": 0.8,
        "metallic": 0.0, "noise_scale": 10.0, "bump": 0.05,
    },
}



def _make_material(name: str, finishing: str, seed: int):
    """Create a Principled BSDF material with procedural noise."""
    preset = FINISHING_PRESETS.get(finishing, FINISHING_PRESETS["none"])
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    base_rgba = list(preset["color"])
    bsdf.inputs["Base Color"].default_value = base_rgba
    bsdf.inputs["Roughness"].default_value = float(preset["roughness"])
    bsdf.inputs["Metallic"].default_value = float(preset.get("metallic", 0.0))

    # Transmission (glass)
    if "transmission" in preset:
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = float(preset["transmission"])
        elif "Transmission" in bsdf.inputs:
            bsdf.inputs["Transmission"].default_value = float(preset["transmission"])
        if "IOR" in bsdf.inputs:
            bsdf.inputs["IOR"].default_value = float(preset.get("ior", 1.45))

    # Emission (light fixtures)
    if "emission" in preset:
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = float(preset["emission"])
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = base_rgba
        elif "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = base_rgba

    # Procedural noise for surface variation
    noise_scale = float(preset.get("noise_scale", 0.0))
    if noise_scale > 0.1:
        tex_coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Location"].default_value = (
            ((seed * 13) % 100) * 0.01,
            ((seed * 7) % 100) * 0.01,
            ((seed * 11) % 100) * 0.01,
        )
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = noise_scale
        noise.inputs["Detail"].default_value = 5.0
        noise.inputs["Roughness"].default_value = 0.65
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = 0.35
        mix.inputs["Color1"].default_value = tuple(base_rgba)
        nt.links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        nt.links.new(noise.outputs["Fac"], mix.inputs["Color2"])
        nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])

    # Bump
    bump_str = float(preset.get("bump", 0.0))
    if bump_str > 0.005:
        voronoi = nt.nodes.new("ShaderNodeTexVoronoi")
        voronoi.inputs["Scale"].default_value = max(2.0, noise_scale * 0.4)
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = bump_str
        bump.inputs["Distance"].default_value = 0.04
        if noise_scale > 0.1:
            nt.links.new(mapping.outputs["Vector"], voronoi.inputs["Vector"])
        nt.links.new(voronoi.outputs["Distance"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def assign_materials(mesh_objs: list, mapping: dict, seed: int) -> dict:
    cache: dict = {}
    for obj in mesh_objs:
        rec = mapping.get(obj.name, {"category": "unknown", "finishing": "raw_concrete"})
        cat = rec.get("category", "unknown")
        fin = rec.get("finishing", "raw_concrete")
        key = f"{cat}__{fin}"
        if key not in cache:
            cache[key] = _make_material(f"mat_{key}", fin, seed=seed + abs(hash(key)) % 1000)
        obj.data.materials.clear()
        obj.data.materials.append(cache[key])
        obj.pass_index = CATEGORY_PASS_INDEX.get(cat, 99)
    _log(f"Assigned {len(cache)} unique materials to {len(mesh_objs)} objects.")
    return cache



# ---------------------------------------------------------------------
# Lighting — the CRITICAL section for realistic interior renders
# ---------------------------------------------------------------------


def setup_world_sky(sun_elevation_deg: float, sun_azimuth_deg: float) -> None:
    """Nishita Sky Texture + Sun Light for natural daylight."""
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.2  # slightly boosted sky
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

    # Sun light for sharp directional shadows
    sun_data = bpy.data.lights.new("SunLight", type="SUN")
    sun_data.energy = 5.0
    sun_data.angle = math.radians(0.55)
    sun_obj = bpy.data.objects.new("SunLight", sun_data)
    bpy.context.collection.objects.link(sun_obj)
    az = math.radians(sun_azimuth_deg)
    el = math.radians(sun_elevation_deg)
    sun_obj.rotation_euler = (math.pi / 2 - el, 0.0, az + math.pi / 2)


def setup_light_portals(elements_json: Path | None, stage_id: int) -> int:
    """Place Area Light portals at every window opening.

    Light portals tell Cycles WHERE light enters the room, drastically
    reducing noise for interior scenes. They emit no light themselves —
    they just guide the importance sampling toward the sky/sun.

    For stages 1-3 (open perimeters), we place extra portals at the
    open wall gaps too.
    """
    # Read window positions from the elements sidecar
    portals_placed = 0

    if elements_json and Path(elements_json).exists():
        payload = json.loads(Path(elements_json).read_text())
        # Room dimensions for facade detection
        room_L, room_W = 22.0, 14.0
        for elem in payload["elements"]:
            cat = elem.get("category", "")
            if cat == "windows":
                bmin = elem["box_min"]
                bmax = elem["box_max"]
                cx = (bmin[0] + bmax[0]) / 2
                cy = (bmin[1] + bmax[1]) / 2
                cz = (bmin[2] + bmax[2]) / 2
                sx = bmax[0] - bmin[0]
                sy = bmax[1] - bmin[1]
                sz = bmax[2] - bmin[2]

                # Detect facade from position (which wall is it flush with?)
                if bmax[1] > room_W - 0.5:
                    facade = "north"
                elif bmin[1] < 0.5:
                    facade = "south"
                elif bmax[0] > room_L - 0.5:
                    facade = "east"
                else:
                    facade = "west"

                portal_data = bpy.data.lights.new(f"Portal_{elem['element_id']}", type="AREA")
                portal_data.energy = 0
                portal_data.cycles.is_portal = True
                portal_obj = bpy.data.objects.new(f"Portal_{elem['element_id']}", portal_data)
                bpy.context.collection.objects.link(portal_obj)

                if facade == "north":
                    portal_obj.location = (cx, cy, cz)
                    portal_obj.rotation_euler = (math.pi / 2, 0, 0)
                    portal_data.size = sx
                    portal_data.size_y = sz
                elif facade == "east":
                    portal_obj.location = (cx, cy, cz)
                    portal_obj.rotation_euler = (math.pi / 2, 0, -math.pi / 2)
                    portal_data.size = sy
                    portal_data.size_y = sz
                elif facade == "west":
                    portal_obj.location = (cx, cy, cz)
                    portal_obj.rotation_euler = (math.pi / 2, 0, math.pi / 2)
                    portal_data.size = sy
                    portal_data.size_y = sz
                elif facade == "south":
                    portal_obj.location = (cx, cy, cz)
                    portal_obj.rotation_euler = (-math.pi / 2, 0, 0)
                    portal_data.size = sx
                    portal_data.size_y = sz

                portals_placed += 1

    # For early stages with open perimeters, add large portals
    # at the open sides to help Cycles find the sky
    if stage_id <= 3:
        # The room is 22x14m. Early stages have open left/south walls.
        open_portals = []
        if stage_id <= 3:  # west (left) open or partial
            open_portals.append(("OpenWest", (0, 7, 1.6), (math.pi/2, 0, math.pi/2), 14, 3.2))
        if stage_id <= 3:  # south open
            open_portals.append(("OpenSouth", (11, 0, 1.6), (-math.pi/2, 0, 0), 22, 3.2))
        for name, loc, rot, size, size_y in open_portals:
            pd = bpy.data.lights.new(name, type="AREA")
            pd.energy = 0
            pd.cycles.is_portal = True
            pd.size = size
            pd.size_y = size_y
            po = bpy.data.objects.new(name, pd)
            po.location = loc
            po.rotation_euler = rot
            bpy.context.collection.objects.link(po)
            portals_placed += 1

    _log(f"Placed {portals_placed} light portal(s).")
    return portals_placed


def setup_ceiling_lights(stage_id: int) -> int:
    """Add 6 recessed Area Lights for stage 7 (finished interior).

    These simulate real recessed LED panel fixtures in a finished
    ceiling. Each is a 0.6m×0.6m warm-white area light pointing down.
    """
    if stage_id < 7:
        return 0

    # 3 columns × 2 rows, matching the ceiling_lights elements in layout.py
    room_length, room_width, room_height = 22.0, 14.0, 3.20
    x_positions = [room_length * frac for frac in [0.2, 0.5, 0.8]]
    y_positions = [room_width * frac for frac in [0.35, 0.65]]
    z = room_height - 0.12  # just below the finished ceiling

    count = 0
    for ix, x in enumerate(x_positions):
        for iy, y in enumerate(y_positions):
            light_data = bpy.data.lights.new(f"CeilLight_{ix}_{iy}", type="AREA")
            light_data.energy = 80.0  # watts
            light_data.color = (1.0, 0.97, 0.92)  # warm white
            light_data.shape = "SQUARE"
            light_data.size = 0.55
            light_obj = bpy.data.objects.new(f"CeilLight_{ix}_{iy}", light_data)
            light_obj.location = (x, y, z)
            light_obj.rotation_euler = (math.pi, 0, 0)  # pointing down
            bpy.context.collection.objects.link(light_obj)
            count += 1

    _log(f"Placed {count} ceiling area light(s) for stage 7.")
    return count



# ---------------------------------------------------------------------
# Camera path — comprehensive room coverage (30-60 seconds)
# ---------------------------------------------------------------------


def _scene_bbox(mesh_objs: list) -> tuple:
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
    """Build a comprehensive camera path that covers the entire 22x14m room.

    The path enters from the door (west wall), walks along the south
    wall, sweeps across to the east wall, looks out the east window,
    continues to the north-east corner, walks along the back (north)
    wall looking at the windows, sweeps to the north-west corner,
    returns toward the center, and finishes with a slow pan of the
    room from the middle.

    Total: 32 waypoints for full spatial coverage.
    """
    L, W, H = 22.0, 14.0, 3.20
    hold_z = 1.55  # smartphone camera height

    # 32 waypoints for comprehensive coverage
    waypoints = [
        # Enter from west door
        (0.5, 2.5, hold_z),
        (1.5, 2.5, hold_z),
        # Walk along south wall toward east
        (3.0, 1.5, hold_z),
        (6.0, 1.8, hold_z),
        (9.0, 2.0, hold_z),
        (12.0, 1.5, hold_z),
        (15.0, 1.8, hold_z),
        (18.0, 2.0, hold_z),
        # SE corner, look at east wall
        (20.0, 2.5, hold_z),
        (20.5, 4.0, hold_z),
        # Walk up along east wall (toward north)
        (20.5, 6.0, hold_z),
        (20.0, 7.0, hold_z),   # center east, facing east window
        (20.5, 9.0, hold_z),
        (20.0, 11.0, hold_z),
        # NE corner
        (20.0, 12.5, hold_z),
        (18.0, 12.5, hold_z),
        # Walk along north wall (back), facing the 2 windows
        (15.0, 12.5, hold_z),  # near WIN_N2
        (13.0, 12.0, hold_z),
        (11.0, 12.5, hold_z),
        (9.0, 12.0, hold_z),
        (7.0, 12.5, hold_z),   # near WIN_N1
        (5.0, 12.0, hold_z),
        # NW corner
        (2.5, 12.5, hold_z),
        (2.0, 11.0, hold_z),
        # Walk down west wall (look at door + plaster)
        (1.5, 9.0, hold_z),
        (1.5, 7.0, hold_z),
        (1.5, 5.0, hold_z),
        (1.5, 3.5, hold_z),
        # Move to center for panoramic sweep
        (6.0, 7.0, hold_z),
        (11.0, 7.0, hold_z),
        (16.0, 7.0, hold_z),
        # Final position: center of room looking around
        (11.0, 7.0, hold_z),
    ]

    # Build Bezier curve
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

    # Camera
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 4.25        # ~26mm full-frame equivalent
    cam_data.sensor_width = 6.17  # smartphone sensor
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    # Look-at target (moves slowly through the room center)
    target = bpy.data.objects.new("CamTarget", None)
    target.empty_display_type = "PLAIN_AXES"
    target.empty_display_size = 0.2
    bpy.context.collection.objects.link(target)
    target.location = (L / 2, W / 2, hold_z - 0.1)

    # Track to target
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

    # Animate path evaluation time
    curve_data.use_path = True
    curve_data.path_duration = frames
    curve_data.animation_data_create()
    action = bpy.data.actions.new("CamPathAction")
    curve_data.animation_data.action = action
    fc = action.fcurves.new(data_path="eval_time")
    fc.keyframe_points.insert(frame=1, value=0.0).interpolation = "LINEAR"
    fc.keyframe_points.insert(frame=frames, value=float(frames)).interpolation = "LINEAR"

    # Animate target (subtle drift to different areas)
    rng = random.Random(args.seed)
    target.animation_data_create()
    target_action = bpy.data.actions.new("TargetAction")
    target.animation_data.action = target_action
    n_keys = 10
    for axis_idx in range(3):
        path = "location"
        fcurve = target_action.fcurves.new(data_path=path, index=axis_idx)
        for k in range(n_keys + 1):
            f = 1 + int(round(k * (frames - 1) / n_keys))
            base = (L / 2, W / 2, hold_z - 0.1)[axis_idx]
            jitter = rng.uniform(-1.5, 1.5) if axis_idx < 2 else rng.uniform(-0.1, 0.1)
            kp = fcurve.keyframe_points.insert(frame=f, value=base + jitter)
            kp.interpolation = "BEZIER"

    # Handheld noise on camera rotation (micro-jitter + sway)
    cam_obj.animation_data_create()
    cam_action = bpy.data.actions.new("CamAction")
    cam_obj.animation_data.action = cam_action
    cam_obj.rotation_mode = "XYZ"
    for i in range(3):
        fc = cam_action.fcurves.new(data_path="rotation_euler", index=i)
        fc.keyframe_points.insert(frame=1, value=0.0).interpolation = "LINEAR"
        fc.keyframe_points.insert(frame=frames, value=0.0).interpolation = "LINEAR"
        # Slow sway
        m1 = fc.modifiers.new(type="NOISE")
        m1.scale = 80.0
        m1.strength = math.radians(0.7)
        m1.phase = (args.seed % 50) + i * 7
        # Fast micro-tremor
        m2 = fc.modifiers.new(type="NOISE")
        m2.scale = 5.0
        m2.strength = math.radians(0.2)
        m2.phase = (args.seed % 50) + i * 13 + 100

    # Tiny location jitter (handheld bobbing)
    for i in range(3):
        fc = cam_action.fcurves.new(data_path="location", index=i)
        fc.keyframe_points.insert(frame=1, value=0.0).interpolation = "LINEAR"
        fc.keyframe_points.insert(frame=frames, value=0.0).interpolation = "LINEAR"
        m = fc.modifiers.new(type="NOISE")
        m.scale = 25.0
        m.strength = 0.015  # 1.5cm RMS
        m.phase = (args.seed % 50) + i * 19 + 200

    bpy.context.scene.camera = cam_obj
    _log(f"Camera built: {len(waypoints)} waypoints, {frames} frames.")
    return cam_obj, path_obj, target



# ---------------------------------------------------------------------
# Render settings + Compositor
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

    # Cycles — HIGH BOUNCES for realistic interior lighting
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    scene.cycles.use_persistent_data = True

    # CRITICAL: High bounce counts for interior scenes
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 8
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 12
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 12

    # Disable filter glossy and clamping for accurate glass transmission
    scene.cycles.blur_glossy = 0.0
    scene.cycles.sample_clamp_indirect = 0.0

    # Motion blur
    scene.render.use_motion_blur = bool(args.motion_blur)
    scene.cycles.motion_blur_position = "CENTER"
    scene.render.motion_blur_shutter = 0.5

    # Frames
    scene.frame_start = 1
    scene.frame_end = args.frames

    # Color management: Filmic + exposure boost for interior brightness
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium Contrast"
    scene.view_settings.exposure = 1.2  # brighten interior


def configure_compositor(args: RenderArgs) -> None:
    """Wire RGB, Depth (EXR), and Segmentation output nodes."""
    scene = bpy.context.scene
    scene.use_nodes = True
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

    # RGB
    fo_rgb = nt.nodes.new("CompositorNodeOutputFile")
    fo_rgb.base_path = str(rgb_dir)
    fo_rgb.file_slots[0].path = "frame_"
    fo_rgb.format.file_format = "PNG"
    fo_rgb.format.color_mode = "RGB"
    fo_rgb.format.color_depth = "8"
    nt.links.new(rl.outputs["Image"], fo_rgb.inputs[0])

    # Depth (EXR 32-bit)
    fo_depth = nt.nodes.new("CompositorNodeOutputFile")
    fo_depth.base_path = str(depth_dir)
    fo_depth.file_slots[0].path = "frame_"
    fo_depth.format.file_format = "OPEN_EXR"
    fo_depth.format.color_mode = "RGB"
    fo_depth.format.color_depth = "32"
    nt.links.new(rl.outputs["Depth"], fo_depth.inputs[0])

    # Segmentation (Object Index)
    div = nt.nodes.new("CompositorNodeMath")
    div.operation = "DIVIDE"
    div.inputs[1].default_value = 65535.0
    nt.links.new(rl.outputs["IndexOB"], div.inputs[0])
    fo_seg = nt.nodes.new("CompositorNodeOutputFile")
    fo_seg.base_path = str(seg_dir)
    fo_seg.file_slots[0].path = "frame_"
    fo_seg.format.file_format = "PNG"
    fo_seg.format.color_mode = "RGB"
    fo_seg.format.color_depth = "16"
    nt.links.new(div.outputs[0], fo_seg.inputs[0])



# ---------------------------------------------------------------------
# Camera path JSON dump
# ---------------------------------------------------------------------


def dump_camera_path(cam_obj, args: RenderArgs, out_path: Path) -> None:
    scene = bpy.context.scene
    cam = cam_obj.data
    w, h = scene.render.resolution_x, scene.render.resolution_y
    fx = (cam.lens / cam.sensor_width) * w
    fy = fx
    intr = {"width_px": w, "height_px": h, "fx": fx, "fy": fy,
            "cx": w / 2.0, "cy": h / 2.0, "lens_mm": cam.lens,
            "sensor_width_mm": cam.sensor_width,
            "horizontal_fov_deg": math.degrees(2 * math.atan(0.5 * cam.sensor_width / cam.lens))}
    deps = bpy.context.evaluated_depsgraph_get()
    frames_data = []
    for f in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(f)
        deps.update()
        cam_eval = cam_obj.evaluated_get(deps)
        m = cam_eval.matrix_world
        eye = (m.translation.x, m.translation.y, m.translation.z)
        fwd = (m.to_3x3() @ mathutils.Vector((0, 0, -1))).normalized()
        target = (eye[0] + fwd.x, eye[1] + fwd.y, eye[2] + fwd.z)
        up = (m.to_3x3() @ mathutils.Vector((0, 1, 0))).normalized()
        frames_data.append({
            "frame_index": f - 1,
            "timestamp_sec": (f - 1) / scene.render.fps,
            "eye_world_m": list(eye),
            "target_world_m": list(target),
            "up_world": [up.x, up.y, up.z],
            "cam_to_world_4x4": [list(row) for row in m],
        })
    payload = {"schema_version": "synthetic_floor_camera_path.v1",
               "renderer": "blender_cycles_gpu", "stage_id": args.stage_id,
               "fps": args.fps, "intrinsics": intr, "frames": frames_data}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def main() -> int:
    global bpy, mathutils
    try:
        import bpy as _bpy
        import mathutils as _mu
    except ImportError as e:
        print(f"ERROR: must run inside Blender: {e}", file=sys.stderr)
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
    _log(f"output_dir   : {args.output_dir}")
    _log(f"resolution   : {args.width}x{args.height}")
    _log(f"frames       : {args.frames} @ {args.fps} fps ({args.frames/args.fps:.1f}s)")
    _log(f"samples      : {args.samples}")

    random.seed(args.seed)
    reset_scene()
    backend = configure_gpu(args.device)
    _log(f"GPU backend  : {backend}")

    mesh_objs = import_mesh(args.input_mesh)
    if not mesh_objs:
        _log("ERROR: no mesh objects imported")
        return 3

    mapping = map_objects_to_elements(mesh_objs, args.elements_json)
    assign_materials(mesh_objs, mapping, seed=args.seed)

    # LIGHTING (the key to not having a black render)
    setup_world_sky(args.sun_elevation_deg, args.sun_azimuth_deg)
    setup_light_portals(args.elements_json, args.stage_id)
    setup_ceiling_lights(args.stage_id)

    # Camera
    cam_obj, _, _ = build_camera(mesh_objs, args.frames, args.fps, args)

    # Render configuration
    configure_render(args)
    try:
        configure_compositor(args)
        _log("Compositor configured.")
    except Exception as e:
        import traceback
        _log(f"ERROR in compositor: {e}")
        _log(traceback.format_exc())
        return 5

    # Render
    _log("Rendering animation...")
    try:
        bpy.ops.render.render(animation=True)
    except Exception as e:
        import traceback
        _log(f"ERROR during render: {e}")
        _log(traceback.format_exc())
        return 6

    _log(f"Render complete in {time.time() - t0:.1f}s")

    # Export camera path
    dump_camera_path(cam_obj, args, args.output_dir / "camera_path.json")

    # Sanity check
    rgb_count = len(list((args.output_dir / "rgb").glob("frame_*.png")))
    _log(f"Output frames: rgb={rgb_count}")
    if rgb_count == 0:
        _log("ERROR: no RGB frames produced")
        return 4

    _log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
