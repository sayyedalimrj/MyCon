"""Mesh export per stage.

Builds a single ``trimesh.Scene`` containing one box per kept element,
with a flat (single-colour) PBR material assigned to each part. The
output formats are OBJ + MTL (for human inspection) and GLB (for
viewers).

The mesh is *not* used by the in-repo software renderer (which uses the
element list directly), but it is the standard way to feed the data
into Blender, Open3D, Meshlab, three.js, the rest of the MyCon
geometry stack, etc.

We also emit a ``stage_NN_elements.json`` sidecar with the element
metadata (id, category, finishing, ifc_global_id, bounding box). This
is the contract the Blender GPU renderer reads to assign procedural
PBR materials and segmentation pass indices, since GLB ``extras`` do
not reliably round-trip through Blender 4.x's glTF importer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import trimesh

from .layout import Element
from .materials import Material, material_for
from .scene_spec import SceneSpec
from .stage_controller import StagedElement, kept_only


def _box_mesh(box_min: tuple[float, float, float], box_max: tuple[float, float, float]) -> trimesh.Trimesh:
    extents = (
        box_max[0] - box_min[0],
        box_max[1] - box_min[1],
        box_max[2] - box_min[2],
    )
    center = (
        0.5 * (box_min[0] + box_max[0]),
        0.5 * (box_min[1] + box_max[1]),
        0.5 * (box_min[2] + box_max[2]),
    )
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    return box


def write_stage_meshes(
    spec: SceneSpec,
    stage_id: int,
    staged: Sequence[StagedElement],
    library: Mapping[str, Material],
    out_dir: Path,
) -> dict[str, Path]:
    """Write OBJ + GLB meshes for a single stage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    kept = kept_only(staged)

    for s in kept:
        e = s.element
        mesh = _box_mesh(e.box_min, e.box_max)
        mat = material_for(e.category, s.finishing, library)
        # Apply a flat per-part colour from the material tint.
        rgb = np.array([mat.tint[0], mat.tint[1], mat.tint[2], 1.0]) * 255
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            face_colors=np.tile(rgb.astype(np.uint8), (len(mesh.faces), 1)),
        )
        scene.add_geometry(
            mesh,
            geom_name=e.id,
            node_name=e.ifc_global_id,
            metadata={
                "ifc_global_id": e.ifc_global_id,
                "category": e.category,
                "name": e.name,
                "finishing": s.finishing,
                "completion": s.completion,
            },
        )

    obj_path = out_dir / f"stage_{stage_id:02d}.obj"
    glb_path = out_dir / f"stage_{stage_id:02d}.glb"
    ply_path = out_dir / f"stage_{stage_id:02d}.ply"
    sidecar_path = out_dir / f"stage_{stage_id:02d}_elements.json"

    # OBJ + MTL
    with obj_path.open("wb") as f:
        f.write(trimesh.exchange.obj.export_obj(scene).encode("utf-8"))

    # GLB (single binary)
    with glb_path.open("wb") as f:
        f.write(trimesh.exchange.gltf.export_glb(scene))

    # PLY (point cloud / mesh inspection)
    if len(scene.geometry) > 0:
        merged = trimesh.util.concatenate(list(scene.geometry.values()))
        with ply_path.open("wb") as f:
            f.write(trimesh.exchange.ply.export_ply(merged))

    # Sidecar JSON: stable element_id -> {category, finishing, ifc_global_id, bbox}
    # The Blender renderer reads this to assign per-category procedural
    # PBR materials and per-element segmentation pass indices.
    sidecar = {
        "schema_version": "synthetic_floor_elements.v1",
        "stage_id": stage_id,
        "elements": [
            {
                "element_id": s.element.id,
                "ifc_global_id": s.element.ifc_global_id,
                "name": s.element.name,
                "category": s.element.category,
                "finishing": s.finishing,
                "completion": s.completion,
                "box_min": list(s.element.box_min),
                "box_max": list(s.element.box_max),
            }
            for s in kept
        ],
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    return {"obj": obj_path, "glb": glb_path, "ply": ply_path, "elements_json": sidecar_path}
