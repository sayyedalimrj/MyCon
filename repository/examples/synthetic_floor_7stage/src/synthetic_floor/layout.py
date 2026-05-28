"""Deterministic geometry of the synthetic floor.

Given a :class:`SceneSpec` this module produces the *full* set of
architectural elements once. The output is a list of axis-aligned box
elements (slab, columns, exterior wall segments, interior wall
segments, ceiling, floor finish, baseboards, doors, window glass,
fixtures). The same set is reused at every stage; the per-stage
controller only filters and re-skins them.

Every element receives:

* an ``id`` (stable, deterministic, consistent across stages);
* an ``ifc_global_id`` (22-character base64 string, IFC4 format);
* a ``category`` matching the keys used in ``stages.*.elements``;
* a ``box`` describing it as ``(min, max)`` in world coordinates.

Coordinate convention
---------------------
* ``x`` runs east (along the floor length).
* ``y`` runs north (along the floor width).
* ``z`` runs up.  ``z = 0`` is the bottom of the slab, ``z = slab_thickness``
  is the top of the slab (everything else stands on top of that).
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .scene_spec import DoorSpec, FloorSpec, RoomSpec, SceneSpec, WindowSpec


CATEGORIES: tuple[str, ...] = (
    "slab",
    "columns",
    "exterior_walls",
    "interior_walls",
    "windows",
    "doors",
    "ceiling",
    "floor_finish",
    "baseboards",
    "fixtures",
)


@dataclass(frozen=True)
class Element:
    """A single architectural element.

    Geometrically it is described as an axis-aligned bounding box in
    world coordinates. For doors and windows we also keep an
    ``opening`` flag (used by the renderer to draw glass / wood instead
    of plaster).
    """

    id: str
    ifc_global_id: str
    name: str
    category: str
    box_min: tuple[float, float, float]
    box_max: tuple[float, float, float]
    metadata: dict

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            0.5 * (self.box_min[0] + self.box_max[0]),
            0.5 * (self.box_min[1] + self.box_max[1]),
            0.5 * (self.box_min[2] + self.box_max[2]),
        )

    @property
    def size(self) -> tuple[float, float, float]:
        return (
            self.box_max[0] - self.box_min[0],
            self.box_max[1] - self.box_min[1],
            self.box_max[2] - self.box_min[2],
        )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


_IFC_GUID_ALPHABET = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "_$"
)


def deterministic_ifc_guid(seed: str) -> str:
    """Generate a stable, deterministic IFC GlobalId.

    IFC GlobalIds are 22-character base64-like strings encoding 128 bits.
    We hash the input seed with SHA-256, take the first 128 bits, and
    re-encode them with the IFC alphabet (RFC-style 64-char). The
    output is exactly 22 characters long, contains only IFC-legal
    characters, and is identical for identical seeds.
    """
    h = hashlib.sha256(seed.encode("utf-8")).digest()
    n = int.from_bytes(h[:16], byteorder="big")  # 128-bit number
    chars: list[str] = []
    # First character holds the top 2 bits, then 21 chars of 6 bits each.
    chars.append(_IFC_GUID_ALPHABET[(n >> 126) & 0x03])
    for i in range(21):
        shift = 120 - i * 6
        chars.append(_IFC_GUID_ALPHABET[(n >> shift) & 0x3F])
    return "".join(chars)


def _make_id(category: str, *parts) -> str:
    return f"{category}.{'_'.join(str(p) for p in parts)}"


# ---------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------


def _slab(floor: FloorSpec) -> Element:
    return Element(
        id=_make_id("SLAB", "GROUND"),
        ifc_global_id=deterministic_ifc_guid("slab|ground"),
        name="Floor 01 Slab",
        category="slab",
        box_min=(0.0, 0.0, 0.0),
        box_max=(floor.length_m, floor.width_m, floor.slab_thickness_m),
        metadata={"role": "structural"},
    )


def _ceiling(floor: FloorSpec) -> Element:
    z_top = floor.height_m
    return Element(
        id=_make_id("CEILING", "01"),
        ifc_global_id=deterministic_ifc_guid("ceiling|01"),
        name="Floor 01 Ceiling",
        category="ceiling",
        box_min=(0.0, 0.0, z_top - floor.slab_thickness_m),
        box_max=(floor.length_m, floor.width_m, z_top),
        metadata={"role": "finishing"},
    )


def _floor_finish(floor: FloorSpec) -> Element:
    z = floor.slab_thickness_m
    return Element(
        id=_make_id("FLOOR_FIN", "01"),
        ifc_global_id=deterministic_ifc_guid("floor_finish|01"),
        name="Floor 01 Finish",
        category="floor_finish",
        box_min=(0.0, 0.0, z),
        box_max=(floor.length_m, floor.width_m, z + 0.02),
        metadata={"role": "finishing"},
    )


def _columns(floor: FloorSpec) -> list[Element]:
    cols: list[Element] = []
    bx, by = floor.grid.bays_x, floor.grid.bays_y
    cs = floor.grid.column_size_m
    z0 = floor.slab_thickness_m
    z1 = floor.height_m
    for ix in range(bx + 1):
        for iy in range(by + 1):
            cx = ix * floor.length_m / bx
            cy = iy * floor.width_m / by
            cols.append(Element(
                id=_make_id("COL", ix, iy),
                ifc_global_id=deterministic_ifc_guid(f"col|{ix}|{iy}"),
                name=f"Column {ix}-{iy}",
                category="columns",
                box_min=(cx - cs / 2, cy - cs / 2, z0),
                box_max=(cx + cs / 2, cy + cs / 2, z1),
                metadata={"grid_x": ix, "grid_y": iy, "role": "structural"},
            ))
    return cols


# Each exterior wall facade is split into segments that fit between
# columns, with window cut-outs subtracted as separate "lintel" pieces.
# To keep things simple but engineering-faithful we represent each wall
# as a single full-height piece per facade *segment* and treat the
# window/door openings as separate elements that REPLACE the masonry
# region at run time.


def _exterior_walls(floor: FloorSpec) -> list[Element]:
    """Build exterior wall segments split by column lines."""
    bx, by = floor.grid.bays_x, floor.grid.bays_y
    t = floor.exterior_wall_thickness_m
    z0 = floor.slab_thickness_m
    z1 = floor.height_m

    segs: list[Element] = []

    # South facade (y = 0) and North facade (y = W). One segment per bay.
    for facade, y_inner, y_outer in (
        ("south", 0.0, -t),
        ("north", floor.width_m, floor.width_m + t),
    ):
        for ix in range(bx):
            x0 = ix * floor.length_m / bx
            x1 = (ix + 1) * floor.length_m / bx
            ymin, ymax = sorted([y_inner, y_outer])
            segs.append(Element(
                id=_make_id("EXT", facade, ix),
                ifc_global_id=deterministic_ifc_guid(f"ext|{facade}|{ix}"),
                name=f"Exterior Wall {facade.title()} bay {ix}",
                category="exterior_walls",
                box_min=(x0, ymin, z0),
                box_max=(x1, ymax, z1),
                metadata={"facade": facade, "bay_index": ix, "role": "exterior"},
            ))

    # West facade (x = 0) and East facade (x = L). One segment per bay.
    for facade, x_inner, x_outer in (
        ("west", 0.0, -t),
        ("east", floor.length_m, floor.length_m + t),
    ):
        for iy in range(by):
            y0 = iy * floor.width_m / by
            y1 = (iy + 1) * floor.width_m / by
            xmin, xmax = sorted([x_inner, x_outer])
            segs.append(Element(
                id=_make_id("EXT", facade, iy),
                ifc_global_id=deterministic_ifc_guid(f"ext|{facade}|{iy}"),
                name=f"Exterior Wall {facade.title()} bay {iy}",
                category="exterior_walls",
                box_min=(xmin, y0, z0),
                box_max=(xmax, y1, z1),
                metadata={"facade": facade, "bay_index": iy, "role": "exterior"},
            ))
    return segs


def _interior_walls(floor: FloorSpec) -> list[Element]:
    """Generate interior partition walls from the room layout.

    For every pair of rooms that share a vertical or horizontal edge we
    add a single wall segment along that shared edge. This keeps the
    geometry deterministic and easy to reason about.
    """
    t = floor.interior_wall_thickness_m
    z0 = floor.slab_thickness_m
    z1 = floor.height_m

    walls: list[Element] = []
    rooms = list(floor.rooms)
    seen: set[tuple[float, float, float, float]] = set()

    def add(x0: float, x1: float, y0: float, y1: float, label: str) -> None:
        key = (round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))
        if key in seen:
            return
        seen.add(key)
        idx = len(walls)
        walls.append(Element(
            id=_make_id("INT", idx),
            ifc_global_id=deterministic_ifc_guid(f"int|{idx}|{label}"),
            name=f"Interior Wall {idx} ({label})",
            category="interior_walls",
            box_min=(x0, y0, z0),
            box_max=(x1, y1, z1),
            metadata={"role": "partition", "label": label},
        ))

    # For every room, walk its 4 edges and add a wall along it if the
    # edge does not lie on the building exterior. We slightly inset the
    # wall on the interior side so two adjacent rooms share one wall.
    for r in rooms:
        x0, y0 = r.sw
        x1, y1 = r.ne
        # South edge (y = y0) - skip if at building south facade
        if y0 > 0.0:
            add(x0, x1, y0 - t / 2, y0 + t / 2, f"{r.id}.south")
        # North edge (y = y1) - skip if at building north facade
        if y1 < floor.width_m:
            add(x0, x1, y1 - t / 2, y1 + t / 2, f"{r.id}.north")
        # West edge (x = x0)
        if x0 > 0.0:
            add(x0 - t / 2, x0 + t / 2, y0, y1, f"{r.id}.west")
        # East edge (x = x1)
        if x1 < floor.length_m:
            add(x1 - t / 2, x1 + t / 2, y0, y1, f"{r.id}.east")
    return walls


def _windows(floor: FloorSpec) -> list[Element]:
    """Window glass panels (placed in their respective facade openings)."""
    t = floor.exterior_wall_thickness_m
    win = floor.window_opening
    z0 = floor.slab_thickness_m + win.sill_height_m
    z1 = z0 + win.height_m
    out: list[Element] = []
    for w in floor.windows:
        if w.facade == "south":
            cx = w.offset_m
            box_min = (cx - win.width_m / 2, -t / 2, z0)
            box_max = (cx + win.width_m / 2, +t / 2, z1)
        elif w.facade == "north":
            cx = w.offset_m
            box_min = (cx - win.width_m / 2, floor.width_m - t / 2, z0)
            box_max = (cx + win.width_m / 2, floor.width_m + t / 2, z1)
        elif w.facade == "west":
            cy = w.offset_m
            box_min = (-t / 2, cy - win.width_m / 2, z0)
            box_max = (+t / 2, cy + win.width_m / 2, z1)
        elif w.facade == "east":
            cy = w.offset_m
            box_min = (floor.length_m - t / 2, cy - win.width_m / 2, z0)
            box_max = (floor.length_m + t / 2, cy + win.width_m / 2, z1)
        else:
            raise ValueError(f"Unknown facade {w.facade!r}")
        out.append(Element(
            id=_make_id("WIN", w.id),
            ifc_global_id=deterministic_ifc_guid(f"win|{w.id}"),
            name=f"Window {w.id}",
            category="windows",
            box_min=box_min,
            box_max=box_max,
            metadata={
                "facade": w.facade,
                "offset_m": w.offset_m,
                "opening_w": win.width_m,
                "opening_h": win.height_m,
                "sill_h": win.sill_height_m,
            },
        ))
    return out


def _doors(floor: FloorSpec) -> list[Element]:
    """Door panels positioned at the room interfaces."""
    door = floor.door_opening
    t_int = floor.interior_wall_thickness_m
    z0 = floor.slab_thickness_m
    z1 = z0 + door.height_m

    by_id = {r.id: r for r in floor.rooms}
    out: list[Element] = []
    for d in floor.doors:
        if d.from_ == "EXTERIOR":
            # Main door on the south facade of the corridor
            cx = d.offset_m
            t = floor.exterior_wall_thickness_m
            box_min = (cx - door.width_m / 2, -t / 2, z0)
            box_max = (cx + door.width_m / 2, +t / 2, z1)
        else:
            # Find the shared edge between the two rooms
            r_a = by_id[d.from_]
            r_b = by_id[d.to]
            shared = _shared_edge(r_a, r_b)
            if shared is None:
                # If rooms do not share an edge, skip (defensive)
                continue
            (sx0, sy0), (sx1, sy1) = shared
            if abs(sx1 - sx0) > abs(sy1 - sy0):
                # Horizontal shared edge -> door cuts through it
                center_x = sx0 + d.offset_m
                y_center = sy0
                box_min = (center_x - door.width_m / 2, y_center - t_int / 2, z0)
                box_max = (center_x + door.width_m / 2, y_center + t_int / 2, z1)
            else:
                center_y = sy0 + d.offset_m
                x_center = sx0
                box_min = (x_center - t_int / 2, center_y - door.width_m / 2, z0)
                box_max = (x_center + t_int / 2, center_y + door.width_m / 2, z1)

        out.append(Element(
            id=_make_id("DOOR", d.id),
            ifc_global_id=deterministic_ifc_guid(f"door|{d.id}"),
            name=f"Door {d.id}",
            category="doors",
            box_min=box_min,
            box_max=box_max,
            metadata={"from": d.from_, "to": d.to, "side": d.side},
        ))
    return out


def _shared_edge(r_a: RoomSpec, r_b: RoomSpec):
    """Return the shared edge between two rooms, if any.

    The shared edge is returned as ``((x0, y0), (x1, y1))`` with
    ``y0 <= y1`` and ``x0 <= x1``. ``None`` if the rooms don't share
    an edge.
    """
    ax0, ay0 = r_a.sw
    ax1, ay1 = r_a.ne
    bx0, by0 = r_b.sw
    bx1, by1 = r_b.ne
    # Vertical shared edge (room A east == room B west or vice versa)
    for ax, bx in ((ax1, bx0), (ax0, bx1)):
        if abs(ax - bx) < 1e-6:
            y0 = max(ay0, by0)
            y1 = min(ay1, by1)
            if y1 > y0 + 1e-6:
                return (ax, y0), (ax, y1)
    # Horizontal shared edge (room A north == room B south or vice versa)
    for ay, by in ((ay1, by0), (ay0, by1)):
        if abs(ay - by) < 1e-6:
            x0 = max(ax0, bx0)
            x1 = min(ax1, bx1)
            if x1 > x0 + 1e-6:
                return (x0, ay), (x1, ay)
    return None


def _baseboards(floor: FloorSpec) -> list[Element]:
    """A thin trim along the bottom of every interior wall."""
    h = 0.10
    t = 0.012
    z0 = floor.slab_thickness_m
    z1 = z0 + h
    out: list[Element] = []
    for r in floor.rooms:
        x0, y0 = r.sw
        x1, y1 = r.ne
        for i, (ax0, ay0, ax1, ay1) in enumerate([
            (x0, y0, x1, y0),  # south
            (x0, y1, x1, y1),  # north
            (x0, y0, x0, y1),  # west
            (x1, y0, x1, y1),  # east
        ]):
            box_min = (
                min(ax0, ax1) - t / 2,
                min(ay0, ay1) - t / 2,
                z0,
            )
            box_max = (
                max(ax0, ax1) + t / 2,
                max(ay0, ay1) + t / 2,
                z1,
            )
            out.append(Element(
                id=_make_id("BASE", r.id, i),
                ifc_global_id=deterministic_ifc_guid(f"base|{r.id}|{i}"),
                name=f"Baseboard {r.id} edge{i}",
                category="baseboards",
                box_min=box_min,
                box_max=box_max,
                metadata={"room_id": r.id, "edge": i},
            ))
    return out


def _fixtures(floor: FloorSpec) -> list[Element]:
    """A few simple light fixtures so stage 7 looks finished."""
    z = floor.height_m - 0.05
    fixtures: list[Element] = []
    for r in floor.rooms:
        cx = 0.5 * (r.sw[0] + r.ne[0])
        cy = 0.5 * (r.sw[1] + r.ne[1])
        fixtures.append(Element(
            id=_make_id("FIX", r.id),
            ifc_global_id=deterministic_ifc_guid(f"fix|{r.id}"),
            name=f"Light Fixture {r.id}",
            category="fixtures",
            box_min=(cx - 0.30, cy - 0.30, z - 0.02),
            box_max=(cx + 0.30, cy + 0.30, z + 0.02),
            metadata={"room_id": r.id, "kind": "ceiling_light"},
        ))
    return fixtures


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def build_layout(spec: SceneSpec) -> list[Element]:
    """Return the complete deterministic list of elements."""
    floor = spec.floor
    elements: list[Element] = []
    elements.append(_slab(floor))
    elements.extend(_columns(floor))
    elements.extend(_exterior_walls(floor))
    elements.extend(_interior_walls(floor))
    elements.extend(_windows(floor))
    elements.extend(_doors(floor))
    elements.append(_ceiling(floor))
    elements.append(_floor_finish(floor))
    elements.extend(_baseboards(floor))
    elements.extend(_fixtures(floor))

    # Sanity: unique IDs and unique IFC GlobalIds across the whole list.
    seen_ids: set[str] = set()
    seen_guids: set[str] = set()
    for e in elements:
        if e.id in seen_ids:
            raise ValueError(f"Duplicate element id: {e.id}")
        if e.ifc_global_id in seen_guids:
            raise ValueError(f"Duplicate IFC GUID for element {e.id}")
        seen_ids.add(e.id)
        seen_guids.add(e.ifc_global_id)
    return elements


def elements_by_category(elements: Iterable[Element]) -> dict[str, list[Element]]:
    """Group elements by category, preserving deterministic order."""
    out: dict[str, list[Element]] = {c: [] for c in CATEGORIES}
    for e in elements:
        out.setdefault(e.category, []).append(e)
    return out
