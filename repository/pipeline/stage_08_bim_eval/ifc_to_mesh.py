"""Convert IFC geometry into Open3D geometry and element metadata."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config_access import cfg_bool, cfg_float, cfg_get, cfg_int, cfg_list
from .geometry_utils import bounds_summary, make_box_point_cloud, require_open3d, write_mesh, write_point_cloud
from .io_utils import write_jsonl_atomic
from .schedule_filter import build_schedule_filter


@dataclass
class BimExtractionResult:
    mesh: Any
    point_cloud: Any
    elements: list[dict[str, Any]]
    units: str
    source: str
    warnings: list[str]
    schedule_filter: dict[str, Any] = field(default_factory=dict)
    visibility_filter: dict[str, Any] = field(default_factory=dict)


DEFAULT_CLASSES = ["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcDoor", "IfcWindow", "IfcStair", "IfcRailing"]


def _import_ifcopenshell() -> Any:
    try:
        import ifcopenshell
        import ifcopenshell.geom
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"IfcOpenShell is required for Stage 8 IFC extraction but could not be imported: {exc}") from exc
    return ifcopenshell


def _detect_ifc_units(model: Any) -> str:
    try:
        for unit_assignment in model.by_type("IfcUnitAssignment"):
            for unit in getattr(unit_assignment, "Units", []) or []:
                unit_type = getattr(unit, "UnitType", "")
                if str(unit_type).upper() == "LENGTHUNIT":
                    prefix = str(getattr(unit, "Prefix", "") or "")
                    name = str(getattr(unit, "Name", "") or "")
                    return (prefix + " " + name).strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"
    return "unknown"


def _settings_apply(settings: Any) -> None:
    for key, value in [
        ("USE_WORLD_COORDS", True),
        ("DISABLE_OPENING_SUBTRACTIONS", False),
    ]:
        try:
            settings.set(getattr(settings, key), value)
        except Exception:  # noqa: BLE001
            try:
                settings.set(key, value)
            except Exception:  # noqa: BLE001
                pass


def _element_to_mesh(ifcopenshell: Any, settings: Any, element: Any) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        geom = shape.geometry
        verts = np.asarray(geom.verts, dtype=np.float64).reshape((-1, 3))
        faces = np.asarray(geom.faces, dtype=np.int32).reshape((-1, 3))
    except Exception:
        return None
    if verts.size == 0:
        return None
    if faces.size == 0:
        faces = np.empty((0, 3), dtype=np.int32)
    return verts, faces


def _metadata_for_element(element: Any, verts: np.ndarray, faces: np.ndarray, vertex_offset: int) -> dict[str, Any]:
    mn = verts.min(axis=0).tolist() if len(verts) else [None, None, None]
    mx = verts.max(axis=0).tolist() if len(verts) else [None, None, None]
    return {
        "global_id": str(getattr(element, "GlobalId", "") or ""),
        "ifc_class": str(element.is_a() if hasattr(element, "is_a") else type(element).__name__),
        "name": str(getattr(element, "Name", "") or ""),
        "object_type": str(getattr(element, "ObjectType", "") or ""),
        "tag": str(getattr(element, "Tag", "") or ""),
        "vertex_start": int(vertex_offset),
        "vertex_count": int(len(verts)),
        "face_count": int(len(faces)),
        "bounds_min": [float(x) for x in mn],
        "bounds_max": [float(x) for x in mx],
    }


def _build_mesh_from_arrays(vertices: list[np.ndarray], faces: list[np.ndarray]) -> Any:
    o3d = require_open3d()
    mesh = o3d.geometry.TriangleMesh()
    if not vertices:
        return mesh
    all_vertices = np.vstack(vertices).astype(np.float64)
    all_faces = np.vstack(faces).astype(np.int32) if faces else np.empty((0, 3), dtype=np.int32)
    mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(all_faces)
    if len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
    return mesh


def _sample_mesh(mesh: Any, target_points: int) -> Any:
    o3d = require_open3d()
    if len(mesh.vertices) == 0:
        return o3d.geometry.PointCloud()
    if len(mesh.triangles) > 0:
        try:
            return mesh.sample_points_uniformly(number_of_points=max(100, int(target_points)), use_triangle_normal=True)
        except Exception:  # noqa: BLE001
            pass
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    return pcd


def _virtual_camera_locations(pcd: Any, cfg: Any) -> tuple[list[np.ndarray], float]:
    b = bounds_summary(pcd)
    center = np.asarray(b.center, dtype=float)
    extent = np.asarray(b.extent, dtype=float)
    diag = max(float(b.diagonal), 1e-6)
    radius = diag * cfg_float(cfg, "bim.visible_shell_hpr_radius_multiplier", 20.0)
    camera_radius = diag * cfg_float(cfg, "bim.visible_shell_camera_radius_multiplier", 2.0)
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 1.0, 0.5]),
        np.array([-1.0, 1.0, 0.5]),
        np.array([1.0, -1.0, 0.5]),
        np.array([-1.0, -1.0, 0.5]),
    ]
    cameras = []
    for d in directions:
        norm = np.linalg.norm(d)
        if norm <= 0:
            continue
        cameras.append(center + (d / norm) * camera_radius)
    return cameras, radius


def _visible_shell_filter(pcd: Any, cfg: Any, logger: logging.Logger) -> tuple[Any, dict[str, Any], list[str]]:
    warnings: list[str] = []
    enabled_raw = str(cfg_get(cfg, "bim.visible_shell_filter_enabled", "false")).strip().lower()
    enabled = enabled_raw in {"1", "true", "yes", "y", "on", "auto"}
    summary: dict[str, Any] = {
        "enabled": enabled,
        "mode": enabled_raw,
        "input_points": int(len(pcd.points)),
        "output_points": int(len(pcd.points)),
        "kept_ratio": 1.0,
        "applied": False,
    }
    if not enabled or len(pcd.points) < 50:
        return pcd, summary, warnings
    cameras, radius = _virtual_camera_locations(pcd, cfg)
    visible: set[int] = set()
    for camera in cameras:
        try:
            _mesh, indices = pcd.hidden_point_removal(camera.tolist(), radius)
            visible.update(int(i) for i in indices)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Hidden point removal failed for one virtual camera: {exc}")
    if not visible:
        warnings.append("Visible-shell filtering found no visible BIM points; keeping unfiltered BIM sample.")
        return pcd, summary, warnings
    keep = sorted(visible)
    kept_ratio = len(keep) / max(1, len(pcd.points))
    min_keep = cfg_float(cfg, "bim.visible_shell_min_keep_ratio", 0.20)
    max_remove = cfg_float(cfg, "bim.visible_shell_max_removed_ratio", 0.80)
    if kept_ratio < min_keep or (1.0 - kept_ratio) > max_remove:
        warnings.append(
            f"Visible-shell filtering would keep {kept_ratio:.3f} of BIM points, outside safe bounds; keeping unfiltered BIM sample."
        )
        return pcd, summary, warnings
    filtered = pcd.select_by_index(keep)
    summary.update(
        {
            "output_points": int(len(filtered.points)),
            "kept_ratio": float(kept_ratio),
            "applied": True,
            "virtual_camera_count": len(cameras),
            "hpr_radius": float(radius),
        }
    )
    logger.info("Applied BIM visible-shell filter: kept %d/%d points", len(filtered.points), len(pcd.points))
    return filtered, summary, warnings


def _synthetic_bim_from_scan(scan_pcd: Any, margin_ratio: float = 0.05) -> BimExtractionResult:
    b = bounds_summary(scan_pcd)
    mn = np.asarray(b.min_bound, dtype=float)
    mx = np.asarray(b.max_bound, dtype=float)
    extent = np.maximum(mx - mn, 1e-6)
    margin = extent * margin_ratio
    pcd = make_box_point_cloud(mn - margin, mx + margin, points_per_edge=18)
    o3d = require_open3d()
    mesh = o3d.geometry.TriangleMesh.create_box(width=float(extent[0] + 2 * margin[0]), height=float(extent[1] + 2 * margin[1]), depth=float(extent[2] + 2 * margin[2]))
    mesh.translate((mn - margin).tolist())
    mesh.compute_vertex_normals()
    return BimExtractionResult(
        mesh=mesh,
        point_cloud=pcd,
        elements=[
            {
                "global_id": "SYNTHETIC_STAGE8_TEST_BOX",
                "ifc_class": "SyntheticBox",
                "name": "Synthetic BIM fallback for smoke tests only",
                "vertex_count": int(len(mesh.vertices)),
                "face_count": int(len(mesh.triangles)),
                "bounds_min": (mn - margin).tolist(),
                "bounds_max": (mx + margin).tolist(),
            }
        ],
        units="meters",
        source="synthetic_test_fallback",
        warnings=["Synthetic BIM fallback used. Do not use this output for real scan-vs-BIM evaluation."],
    )


def _write_result(result: BimExtractionResult, output_mesh_path: Path, output_ply_path: Path, element_metadata_path: Path) -> None:
    write_mesh(output_mesh_path, result.mesh)
    write_point_cloud(output_ply_path, result.point_cloud)
    write_jsonl_atomic(element_metadata_path, result.elements)


def extract_ifc_geometry(
    cfg: Any,
    ifc_path: Path,
    scan_pcd: Any,
    output_mesh_path: Path,
    output_ply_path: Path,
    element_metadata_path: Path,
    logger: logging.Logger,
) -> BimExtractionResult:
    """Extract IFC geometry to Open3D mesh/point cloud and element JSONL."""
    classes = [str(x) for x in cfg_list(cfg, "bim.element_classes", DEFAULT_CLASSES)]
    target_points = cfg_int(cfg, "bim.bim_sample_points", 200_000)
    allow_synthetic = cfg_bool(cfg, "bim.allow_synthetic_ifc_fallback_for_tests", False)

    if allow_synthetic and (not ifc_path.exists() or ifc_path.stat().st_size == 0):
        result = _synthetic_bim_from_scan(scan_pcd)
        _write_result(result, output_mesh_path, output_ply_path, element_metadata_path)
        return result

    ifcopenshell = _import_ifcopenshell()
    try:
        model = ifcopenshell.open(str(ifc_path))
    except Exception as exc:  # noqa: BLE001
        if allow_synthetic:
            logger.warning("IfcOpenShell could not open IFC; using synthetic fallback because test flag is enabled: %s", exc)
            result = _synthetic_bim_from_scan(scan_pcd)
            _write_result(result, output_mesh_path, output_ply_path, element_metadata_path)
            return result
        raise RuntimeError(f"IfcOpenShell failed to open IFC file {ifc_path}: {exc}") from exc

    settings = ifcopenshell.geom.settings()
    _settings_apply(settings)
    schedule_filter = build_schedule_filter(cfg)
    units = _detect_ifc_units(model)
    vertices_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    elements: list[dict[str, Any]] = []
    warnings: list[str] = []
    vertex_offset = 0
    attempted = 0
    extracted = 0
    skipped_by_schedule = 0

    for class_name in classes:
        try:
            candidates = list(model.by_type(class_name))
        except Exception:
            candidates = []
        for element in candidates:
            attempted += 1
            if not schedule_filter.allow(element):
                skipped_by_schedule += 1
                continue
            arrays = _element_to_mesh(ifcopenshell, settings, element)
            if arrays is None:
                continue
            verts, faces = arrays
            if len(verts) == 0:
                continue
            if len(faces) > 0:
                faces_shifted = faces + vertex_offset
                faces_list.append(faces_shifted)
            vertices_list.append(verts)
            elements.append(_metadata_for_element(element, verts, faces, vertex_offset))
            vertex_offset += len(verts)
            extracted += 1

    if not vertices_list:
        message = f"No extractable IFC geometry found for classes {classes}; attempted={attempted}; skipped_by_schedule={skipped_by_schedule}."
        if allow_synthetic:
            logger.warning("%s Using synthetic fallback because test flag is enabled.", message)
            result = _synthetic_bim_from_scan(scan_pcd)
            _write_result(result, output_mesh_path, output_ply_path, element_metadata_path)
            return result
        raise RuntimeError(message)

    mesh = _build_mesh_from_arrays(vertices_list, faces_list)
    point_cloud = _sample_mesh(mesh, target_points)
    if len(point_cloud.points) == 0:
        raise RuntimeError("Extracted IFC mesh did not yield a BIM point cloud")
    point_cloud, visibility_summary, visibility_warnings = _visible_shell_filter(point_cloud, cfg, logger)
    warnings.extend(visibility_warnings)
    _write_result(
        BimExtractionResult(mesh=mesh, point_cloud=point_cloud, elements=elements, units=units, source="ifcopenshell", warnings=[]),
        output_mesh_path,
        output_ply_path,
        element_metadata_path,
    )
    warnings.append(f"IFC extraction attempted {attempted} elements, extracted {extracted}, skipped_by_schedule={skipped_by_schedule}.")
    if schedule_filter.warnings:
        warnings.extend(schedule_filter.warnings)
    return BimExtractionResult(
        mesh=mesh,
        point_cloud=point_cloud,
        elements=elements,
        units=units,
        source="ifcopenshell",
        warnings=warnings,
        schedule_filter=schedule_filter.summary(),
        visibility_filter=visibility_summary,
    )
