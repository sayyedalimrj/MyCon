"""Read-only, explicitly scoped IFC mesh inventory. No inferred correspondences."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .core import digest, point


def local_file(root: Path, rel: str, expected_hash: str | None = None) -> Path:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Evidence paths must be project-relative")
    root = root.resolve()
    p = (root / p).resolve()
    if not p.is_relative_to(root) or not p.is_file() or p.stat().st_size == 0:
        raise ValueError("Missing or out-of-root input: " + rel)
    if expected_hash is not None and digest(p) != expected_hash:
        raise ValueError("SHA256 mismatch: " + rel)
    return p


def write_new(path: Path, data: dict) -> None:
    encoded = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as f:
        f.write(encoded)


def mesh_record(guid, kind, vertices, faces):
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces)
    if (v.ndim != 2 or v.shape[1] != 3 or len(v) < 3 or
            not np.isfinite(v).all() or f.ndim != 2 or f.shape[1] != 3 or
            len(f) == 0 or f.dtype.kind not in "iu" or
            np.any(f < 0) or np.any(f >= len(v))):
        raise ValueError("Invalid IFC mesh: " + guid)
    tri = v[f]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if not np.any(area2 > 1e-12):
        raise ValueError("No nondegenerate IFC faces: " + guid)
    return {"ifc_global_id": guid, "ifc_class": kind,
            "vertices_xyz_m": v.tolist(), "triangle_vertex_indices": f.tolist()}


def extract_inventory(root: Path, ifc_rel: str, revision: str,
                      guids: list[str], output_rel: str) -> dict:
    if not revision.strip() or not guids or len(set(guids)) != len(guids):
        raise ValueError("Explicit revision and unique IFC GlobalIds required")
    root = root.resolve()
    src = local_file(root, ifc_rel)
    dst = (root / output_rel).resolve()
    if Path(output_rel).is_absolute() or not dst.is_relative_to(root) or dst.exists():
        raise ValueError("Inventory destination must be new and project-relative")
    try:
        import ifcopenshell
        import ifcopenshell.geom
        import ifcopenshell.util.unit
    except ImportError as exc:
        raise ValueError("Optional IFC dependency missing: install ifcopenshell==0.8.4") from exc
    sha = digest(src)
    model = ifcopenshell.open(str(src))
    units = float(ifcopenshell.util.unit.calculate_unit_scale(model))
    if not np.isfinite(units) or units <= 0:
        raise ValueError("Invalid IFC declared length unit")
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    settings.set("convert-back-units", False)  # kernel vertices already SI metres
    elements = []
    for guid in guids:
        obj = model.by_guid(guid)
        if obj is None or not obj.is_a("IfcElement") or obj.is_a("IfcOpeningElement"):
            raise ValueError("Select an explicit physical IfcElement: " + guid)
        shape = ifcopenshell.geom.create_shape(settings, obj)
        v = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
        f = np.asarray(shape.geometry.faces, dtype=int).reshape(-1, 3)
        elements.append(mesh_record(guid, obj.is_a(), v, f))
    if digest(src) != sha:
        raise ValueError("IFC changed during extraction")
    doc = {"schema_version": "mycon.ifc_inventory.v1", "coordinate_frame": "bim_world",
           "units": "m", "ifc_project_unit_to_m": units,
           "frame_note": "IFC object placements applied; engineering world, no CRS/map conversion",
           "model": {"path": src.relative_to(root).as_posix(), "sha256": sha,
                     "revision": revision, "ifc_schema": model.schema},
           "ifcopenshell_version": str(ifcopenshell.version),
           "elements": elements, "metric_validation": "pending",
           "review_state": "mesh_candidates_only"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_new(dst, doc)
    return doc


def selected_vertex(element: dict, index: int):
    if type(index) is not int or not 0 <= index < len(element["vertices_xyz_m"]):
        raise ValueError("Invalid IFC vertex index")
    return point(element["vertices_xyz_m"][index], "IFC vertex").tolist()


def selected_triangle(element: dict, index: int, normal_sign: int):
    faces = element["triangle_vertex_indices"]
    if type(index) is not int or not 0 <= index < len(faces) or type(normal_sign) is not int or normal_sign not in (-1, 1):
        raise ValueError("Invalid triangle index or normal orientation")
    poly = np.array([selected_vertex(element, i) for i in faces[index]])
    n = np.cross(poly[1] - poly[0], poly[2] - poly[0])
    length = float(np.linalg.norm(n))
    if length <= 1e-12:
        raise ValueError("Degenerate IFC triangle")
    return poly.tolist(), (normal_sign * n / length).tolist()
