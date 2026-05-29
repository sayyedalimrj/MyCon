"""Layout builder — converts the scene spec into concrete 3-D elements.

This module generates every geometric element (axis-aligned box) for a
single 22×14m room with progressive construction stages.  The output is
a flat list of ``Element`` instances, each described by:

- A unique stable ``id``
- A category (used by the stage controller to decide visibility)
- An axis-aligned bounding box (``box_min``, ``box_max``)
- Rich metadata (IFC GUID, name, facade, role, …)

The geometry is **constant** across all 7 stages.  Only the *visibility*
and *finishing* of each element changes per stage — the stage controller
handles that.

Element categories (matching scene.yaml):
  slab, columns, ceiling_slab, ceiling_finish, overhead_pipes,
  sill_north, sill_east, north_wall, east_wall, west_wall, south_wall,
  windows, door, plaster_left_lower, plaster_left_upper, plaster_other,
  ceiling_lights
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .scene_spec import FloorSpec, SceneSpec


# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------

# All categories that the stage controller knows about.
ALL_CATEGORIES: tuple[str, ...] = (
    "slab",
    "columns",
    "ceiling_slab",
    "ceiling_finish",
    "overhead_pipes",
    "sill_north",
    "sill_east",
    "north_wall",
    "east_wall",
    "west_wall",
    "south_wall",
    "windows",
    "door",
    "plaster_left_lower",
    "plaster_left_upper",
    "plaster_other",
    "ceiling_lights",
)


@dataclass(frozen=True)
class Element:
    """One physical building element defined as an axis-aligned box.

    Geometrically it is described as an axis-aligned bounding box in
    world coordinates.
    """

    id: str
    ifc_global_id: str
    name: str
    category: str
    box_min: tuple[float, float, float]
    box_max: tuple[float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(b - a for a, b in zip(self.box_min, self.box_max))

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple(0.5 * (a + b) for a, b in zip(self.box_min, self.box_max))

    @property
    def volume_m3(self) -> float:
        s = self.size
        return s[0] * s[1] * s[2]


# ---------------------------------------------------------------------
# Stable ID helpers
# ---------------------------------------------------------------------


def _make_id(*parts) -> str:
    return "_".join(str(p) for p in parts)


def deterministic_ifc_guid(seed_str: str) -> str:
    """Generate a stable 22-character IFC GlobalId from a seed string.

    The IFC GUID is Base64url-encoded (A-Z, a-z, 0-9, _, $) truncated
    to 22 characters.  We derive it deterministically from SHA-256.
    """
    h = hashlib.sha256(seed_str.encode()).digest()[:16]
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
    out: list[str] = []
    for byte in h:
        out.append(chars[byte % 64])
        if len(out) >= 22:
            break
    return "".join(out[:22])


# ---------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------


def build_layout(spec: SceneSpec) -> list[Element]:
    """Build the complete element inventory for the 22×14m room.

    Returns all elements regardless of stage. The stage controller
    will filter them by category and completion later.
    """
    f = spec.floor
    elements: list[Element] = []
    elements.extend(_build_slab(f))
    elements.extend(_build_columns(f))
    elements.extend(_build_ceiling_slab(f))
    elements.extend(_build_overhead_pipes(f))
    elements.extend(_build_sills(f))
    elements.extend(_build_walls(f))
    elements.extend(_build_door(f))
    elements.extend(_build_windows(f))
    elements.extend(_build_ceiling_finish(f))
    elements.extend(_build_ceiling_lights(f))
    elements.extend(_build_plaster_layers(f))
    return elements


def elements_by_category(elements: list[Element]) -> dict[str, list[Element]]:
    """Group elements by category."""
    out: dict[str, list[Element]] = {}
    for e in elements:
        out.setdefault(e.category, []).append(e)
    return out


# ---------------------------------------------------------------------
# Individual builders
# ---------------------------------------------------------------------


def _build_slab(f: FloorSpec) -> list[Element]:
    """Floor slab: full-size concrete pad."""
    return [Element(
        id="SLAB_01",
        ifc_global_id=deterministic_ifc_guid("slab|01"),
        name="Floor Slab",
        category="slab",
        box_min=(0.0, 0.0, -f.slab_thickness_m),
        box_max=(f.length_m, f.width_m, 0.0),
        metadata={"role": "structural"},
    )]


def _build_columns(f: FloorSpec) -> list[Element]:
    """Structural pillars at grid intersections (floor-to-ceiling)."""
    bx, by = f.grid.bays_x, f.grid.bays_y
    cs = f.grid.column_size_m
    half = cs / 2.0
    elements: list[Element] = []

    for ix in range(bx + 1):
        x = ix * f.length_m / bx
        for iy in range(by + 1):
            y = iy * f.width_m / by
            elements.append(Element(
                id=_make_id("COL", ix, iy),
                ifc_global_id=deterministic_ifc_guid(f"col|{ix}|{iy}"),
                name=f"Column ({ix},{iy})",
                category="columns",
                box_min=(x - half, y - half, 0.0),
                box_max=(x + half, y + half, f.height_m),
                metadata={"grid_ix": ix, "grid_iy": iy, "role": "structural"},
            ))
    return elements


def _build_ceiling_slab(f: FloorSpec) -> list[Element]:
    """Raw concrete ceiling slab (hidden in stage 7 by finished ceiling)."""
    return [Element(
        id="CEIL_SLAB_01",
        ifc_global_id=deterministic_ifc_guid("ceil_slab|01"),
        name="Ceiling Slab (raw concrete)",
        category="ceiling_slab",
        box_min=(0.0, 0.0, f.height_m),
        box_max=(f.length_m, f.width_m, f.height_m + f.slab_thickness_m),
        metadata={"role": "structural"},
    )]


def _build_overhead_pipes(f: FloorSpec) -> list[Element]:
    """Visible overhead pipes/conduit under the ceiling (stages 1-6).

    Modeled as a few horizontal rectangular runs.
    """
    pipe_z_top = f.height_m - 0.05
    pipe_z_bot = f.height_m - 0.18
    pipes: list[Element] = []
    # 3 east-west pipe runs
    for i, y in enumerate([3.5, 7.0, 10.5]):
        pipes.append(Element(
            id=_make_id("PIPE", i),
            ifc_global_id=deterministic_ifc_guid(f"pipe|{i}"),
            name=f"Overhead Pipe Run {i+1}",
            category="overhead_pipes",
            box_min=(0.5, y - 0.06, pipe_z_bot),
            box_max=(f.length_m - 0.5, y + 0.06, pipe_z_top),
            metadata={"direction": "east-west", "role": "services"},
        ))
    return pipes


def _build_sills(f: FloorSpec) -> list[Element]:
    """Low masonry sills on north and east walls (stage 1 only).

    These are ~1m-high cinderblock parapets before the full wall is built.
    """
    t = f.exterior_wall_thickness_m
    sill_h = 1.0  # 1m high sill

    sills: list[Element] = []
    # North sill (full length of the back wall, between columns)
    cs = f.grid.column_size_m / 2
    sills.append(Element(
        id="SILL_NORTH_01",
        ifc_global_id=deterministic_ifc_guid("sill_north|01"),
        name="North Wall Sill (masonry parapet)",
        category="sill_north",
        box_min=(cs, f.width_m - t, 0.0),
        box_max=(f.length_m - cs, f.width_m, sill_h),
        metadata={"facade": "north", "role": "masonry_sill"},
    ))
    # East sill
    sills.append(Element(
        id="SILL_EAST_01",
        ifc_global_id=deterministic_ifc_guid("sill_east|01"),
        name="East Wall Sill (masonry parapet)",
        category="sill_east",
        box_min=(f.length_m - t, cs, 0.0),
        box_max=(f.length_m, f.width_m - cs, sill_h),
        metadata={"facade": "east", "role": "masonry_sill"},
    ))
    return sills


def _build_walls(f: FloorSpec) -> list[Element]:
    """Perimeter walls (cinderblock infill between structural pillars).

    Each wall is built as segments between column lines, with window
    and door openings cut out (i.e. those segments are shorter or
    missing). The walls span floor to ceiling height.

    Windows/doors create REAL openings in the wall geometry — the wall
    simply has a gap where the opening is. This allows light to pass
    through naturally.
    """
    t = f.exterior_wall_thickness_m
    cs = f.grid.column_size_m / 2
    H = f.height_m
    bx, by = f.grid.bays_x, f.grid.bays_y
    win = f.window_opening
    door = f.door_opening

    elements: list[Element] = []

    # --- NORTH wall (back, Y = width) — has 2 window openings -------
    # Window positions along X
    win_positions_north = [w.offset_m for w in f.windows if w.facade == "north"]
    win_half_w = win.width_m / 2
    win_sill = win.sill_height_m
    win_top = win_sill + win.height_m

    north_segs = _wall_segments_with_openings(
        wall_start=cs, wall_end=f.length_m - cs,
        openings=[(wp - win_half_w, wp + win_half_w, win_sill, win_top)
                  for wp in win_positions_north],
        full_height=H,
    )
    for i, (x0, x1, z0, z1) in enumerate(north_segs):
        elements.append(Element(
            id=_make_id("NWALL", i),
            ifc_global_id=deterministic_ifc_guid(f"nwall|{i}"),
            name=f"North Wall segment {i}",
            category="north_wall",
            box_min=(x0, f.width_m - t, z0),
            box_max=(x1, f.width_m, z1),
            metadata={"facade": "north", "segment": i},
        ))

    # --- EAST wall (right, X = length) — has 1 window opening --------
    win_positions_east = [w.offset_m for w in f.windows if w.facade == "east"]
    east_segs = _wall_segments_with_openings(
        wall_start=cs, wall_end=f.width_m - cs,
        openings=[(wp - win_half_w, wp + win_half_w, win_sill, win_top)
                  for wp in win_positions_east],
        full_height=H,
    )
    for i, (x0, x1, z0, z1) in enumerate(east_segs):
        elements.append(Element(
            id=_make_id("EWALL", i),
            ifc_global_id=deterministic_ifc_guid(f"ewall|{i}"),
            name=f"East Wall segment {i}",
            category="east_wall",
            box_min=(f.length_m - t, x0, z0),
            box_max=(f.length_m, x1, z1),
            metadata={"facade": "east", "segment": i},
        ))

    # --- WEST wall (left, X = 0) — has 1 door opening ----------------
    door_positions_west = [d.offset_m for d in f.doors if d.side == "west"]
    door_half_w = door.width_m / 2
    door_top = door.height_m

    west_segs = _wall_segments_with_openings(
        wall_start=cs, wall_end=f.width_m - cs,
        openings=[(dp - door_half_w, dp + door_half_w, 0.0, door_top)
                  for dp in door_positions_west],
        full_height=H,
    )
    for i, (x0, x1, z0, z1) in enumerate(west_segs):
        elements.append(Element(
            id=_make_id("WWALL", i),
            ifc_global_id=deterministic_ifc_guid(f"wwall|{i}"),
            name=f"West Wall segment {i}",
            category="west_wall",
            box_min=(0.0, x0, z0),
            box_max=(t, x1, z1),
            metadata={"facade": "west", "segment": i},
        ))

    # --- SOUTH wall (front, Y = 0) — solid, no openings ---------------
    elements.append(Element(
        id="SWALL_01",
        ifc_global_id=deterministic_ifc_guid("swall|01"),
        name="South Wall (solid)",
        category="south_wall",
        box_min=(cs, -t, 0.0),
        box_max=(f.length_m - cs, 0.0, H),
        metadata={"facade": "south"},
    ))

    return elements


def _wall_segments_with_openings(
    wall_start: float,
    wall_end: float,
    openings: list[tuple[float, float, float, float]],
    full_height: float,
) -> list[tuple[float, float, float, float]]:
    """Split a wall run into solid segments around rectangular openings.

    Each opening is ``(start_along_wall, end_along_wall, z_bottom, z_top)``.

    Returns ``[(along_start, along_end, z_bottom, z_top), ...]`` for
    solid wall pieces. An opening creates:
    - A piece BELOW the opening (if z_bottom > 0)
    - A piece ABOVE the opening (if z_top < full_height)
    - Solid pieces to the LEFT and RIGHT of the opening (full height)

    This is a 2-D boolean-subtract approach so light actually passes
    through the openings.
    """
    if not openings:
        return [(wall_start, wall_end, 0.0, full_height)]

    # Sort openings by position
    openings_sorted = sorted(openings, key=lambda o: o[0])
    segments: list[tuple[float, float, float, float]] = []

    prev_end = wall_start
    for op_start, op_end, z_bot, z_top in openings_sorted:
        # Solid piece before the opening (full height)
        if op_start > prev_end + 0.01:
            segments.append((prev_end, op_start, 0.0, full_height))
        # Piece below the opening (sill)
        if z_bot > 0.01:
            segments.append((op_start, op_end, 0.0, z_bot))
        # Piece above the opening (lintel)
        if z_top < full_height - 0.01:
            segments.append((op_start, op_end, z_top, full_height))
        prev_end = op_end

    # Solid piece after the last opening
    if prev_end < wall_end - 0.01:
        segments.append((prev_end, wall_end, 0.0, full_height))

    return segments


def _build_door(f: FloorSpec) -> list[Element]:
    """Door panel (metal door installed in stage 6+).

    Positioned in the rough opening on the west wall.
    """
    d = f.doors[0]  # single door
    door = f.door_opening
    t = f.exterior_wall_thickness_m
    half_w = door.width_m / 2
    cy = d.offset_m

    return [Element(
        id="DOOR_01",
        ifc_global_id=deterministic_ifc_guid("door|01"),
        name="Metal Door (west wall)",
        category="door",
        box_min=(0.0, cy - half_w, 0.0),
        box_max=(t * 0.5, cy + half_w, door.height_m),
        metadata={"facade": "west", "material": "metal_grey"},
    )]


def _build_windows(f: FloorSpec) -> list[Element]:
    """Window frames + glazing (installed only in stage 7).

    Each window is a thin panel positioned flush with the outer face
    of its wall, matching the opening dimensions exactly. The Blender
    renderer will assign a glass/transparent material to these.
    """
    win = f.window_opening
    t = f.exterior_wall_thickness_m
    half_w = win.width_m / 2
    sill = win.sill_height_m
    top = sill + win.height_m

    elements: list[Element] = []
    for w in f.windows:
        if w.facade == "north":
            box_min = (w.offset_m - half_w, f.width_m - t * 0.3, sill)
            box_max = (w.offset_m + half_w, f.width_m, top)
        elif w.facade == "east":
            box_min = (f.length_m - t * 0.3, w.offset_m - half_w, sill)
            box_max = (f.length_m, w.offset_m + half_w, top)
        elif w.facade == "west":
            box_min = (0.0, w.offset_m - half_w, sill)
            box_max = (t * 0.3, w.offset_m + half_w, top)
        elif w.facade == "south":
            box_min = (w.offset_m - half_w, -t * 0.3, sill)
            box_max = (w.offset_m + half_w, 0.0, top)
        else:
            raise ValueError(f"Unknown facade: {w.facade}")

        elements.append(Element(
            id=_make_id("WIN", w.id),
            ifc_global_id=deterministic_ifc_guid(f"win|{w.id}"),
            name=f"Window {w.id}",
            category="windows",
            box_min=box_min,
            box_max=box_max,
            metadata={"facade": w.facade, "offset_m": w.offset_m,
                      "width_m": win.width_m, "height_m": win.height_m},
        ))
    return elements


def _build_ceiling_finish(f: FloorSpec) -> list[Element]:
    """Finished dropped ceiling panel (stage 7 only).

    Sits ~10cm below the raw ceiling slab. Covers the full room.
    """
    return [Element(
        id="CEIL_FINISH_01",
        ifc_global_id=deterministic_ifc_guid("ceil_finish|01"),
        name="Finished Ceiling (dropped panel)",
        category="ceiling_finish",
        box_min=(0.0, 0.0, f.height_m - 0.10),
        box_max=(f.length_m, f.width_m, f.height_m - 0.02),
        metadata={"role": "architectural"},
    )]


def _build_ceiling_lights(f: FloorSpec) -> list[Element]:
    """6 recessed square light fixtures in the finished ceiling.

    Arranged in a 3×2 grid centered in the room.
    """
    fixture_size = 0.60  # 60cm × 60cm recessed panels
    fixture_depth = 0.05
    z_top = f.height_m - 0.02
    z_bot = z_top - fixture_depth
    half = fixture_size / 2

    # 3 columns × 2 rows
    x_positions = [f.length_m * frac for frac in [0.2, 0.5, 0.8]]
    y_positions = [f.width_m * frac for frac in [0.35, 0.65]]

    elements: list[Element] = []
    idx = 0
    for x in x_positions:
        for y in y_positions:
            elements.append(Element(
                id=_make_id("LIGHT", idx),
                ifc_global_id=deterministic_ifc_guid(f"light|{idx}"),
                name=f"Ceiling Light {idx+1}",
                category="ceiling_lights",
                box_min=(x - half, y - half, z_bot),
                box_max=(x + half, y + half, z_top),
                metadata={"fixture_index": idx, "role": "lighting"},
            ))
            idx += 1
    return elements


def _build_plaster_layers(f: FloorSpec) -> list[Element]:
    """Plaster overlay layers on the west (left) wall.

    - plaster_left_lower: lower half of west wall (slab to 1.6m)
    - plaster_left_upper: upper half (1.6m to ceiling)
    - plaster_other: all OTHER walls (north, east, south) — thin overlay

    These are thin (2cm) shells on the interior face of the cinderblock.
    They only become visible in stages 5/6/7.
    """
    t = f.exterior_wall_thickness_m
    plaster_t = 0.02
    H = f.height_m
    mid_h = 1.60  # lower/upper split height
    cs = f.grid.column_size_m / 2

    elements: list[Element] = []

    # West wall plaster — lower half
    elements.append(Element(
        id="PLASTER_W_LOW",
        ifc_global_id=deterministic_ifc_guid("plaster_w_low"),
        name="Plaster West Wall (lower half)",
        category="plaster_left_lower",
        box_min=(t, cs, 0.0),
        box_max=(t + plaster_t, f.width_m - cs, mid_h),
        metadata={"facade": "west", "zone": "lower"},
    ))

    # West wall plaster — upper half
    elements.append(Element(
        id="PLASTER_W_UP",
        ifc_global_id=deterministic_ifc_guid("plaster_w_up"),
        name="Plaster West Wall (upper half)",
        category="plaster_left_upper",
        box_min=(t, cs, mid_h),
        box_max=(t + plaster_t, f.width_m - cs, H),
        metadata={"facade": "west", "zone": "upper"},
    ))

    # Plaster on other walls (north, east, south) — one element each
    # North (interior face)
    elements.append(Element(
        id="PLASTER_N",
        ifc_global_id=deterministic_ifc_guid("plaster_n"),
        name="Plaster North Wall",
        category="plaster_other",
        box_min=(cs, f.width_m - t - plaster_t, 0.0),
        box_max=(f.length_m - cs, f.width_m - t, H),
        metadata={"facade": "north"},
    ))

    # East (interior face)
    elements.append(Element(
        id="PLASTER_E",
        ifc_global_id=deterministic_ifc_guid("plaster_e"),
        name="Plaster East Wall",
        category="plaster_other",
        box_min=(f.length_m - t - plaster_t, cs, 0.0),
        box_max=(f.length_m - t, f.width_m - cs, H),
        metadata={"facade": "east"},
    ))

    # South (interior face)
    elements.append(Element(
        id="PLASTER_S",
        ifc_global_id=deterministic_ifc_guid("plaster_s"),
        name="Plaster South Wall",
        category="plaster_other",
        box_min=(cs, 0.0, 0.0),
        box_max=(f.length_m - cs, plaster_t, H),
        metadata={"facade": "south"},
    ))

    return elements
