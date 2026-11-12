"""Optional, non-destructive and reference-agnostic Stage-7 geometry research output.

Scientific scope: observed finite plane patches and locally supported feature
hypotheses in the ORIGINAL reconstruction coordinate frame. Neither metric
scale nor BIM registration nor camera visibility can be inferred from a PLY.

Only NumPy/SciPy is required for geometry; trimesh is used for PLY I/O and
matplotlib for optional QA image. Does not change the Stage-7 legacy outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

SCHEMA = "mycon.comparison_ready_geometry.v1"
RUN_ID = re.compile(r"^geom-ready-[A-Za-z0-9][A-Za-z0-9._-]*$")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _xyz(arr: Any) -> np.ndarray:
    pts = np.asarray(arr, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError("At least 3 Nx3 points are required")
    return pts


def _spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    nn = cKDTree(points).query(points, k=2, workers=1)[0][:, 1]
    positive = nn[nn > 1e-12]
    return float(np.median(positive)) if len(positive) else 0.0


def clean_points(points: Any, rgb: Any = None, *, voxel: float = 0.0,
                 outlier_mad: float = 5.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Conservative SOR followed by medoid voxelization preserving source RGB.

    Stage-7 cleaned outputs should normally be supplied directly, with
    outlier_mad=0 and voxel=0 unless a further decimation is justified.
    """
    pts = _xyz(points)
    colors = None if rgb is None else np.asarray(rgb)
    if colors is not None and (colors.ndim != 2 or len(colors) != len(pts) or colors.shape[1] < 3):
        raise ValueError("RGB shape must be N x 3 or N x 4")
    if voxel < 0 or not np.isfinite(voxel) or outlier_mad < 0:
        raise ValueError("Invalid cleaning thresholds")
    fin = np.isfinite(pts).all(axis=1)
    if colors is not None:
        fin &= np.isfinite(colors[:, :3]).all(axis=1)
    raw_count = len(pts)
    pts = pts[fin]
    colors = None if colors is None else colors[fin, :3]
    finite_count = len(pts)
    if len(pts) < 3:
        raise ValueError("No usable finite input")
    spacing = _spacing(pts)
    outliers = 0
    if outlier_mad > 0 and len(pts) >= 20:
        neighbors = cKDTree(pts).query(pts, k=min(13, len(pts)), workers=1)[0]
        local = np.mean(neighbors[:, 1:], axis=1)
        med = float(np.median(local))
        robust_std = 1.4826 * float(np.median(np.abs(local - med)))
        cutoff = med + outlier_mad * max(robust_std, spacing * 0.1, 1e-12)
        # Only erase isolated spikes when nearest-neighbour gap is also high.
        isolated = neighbors[:, 1] > max(2.5 * spacing, cutoff / 2)
        keep = ~((local > cutoff) & isolated)
        outliers = int((~keep).sum())
        pts = pts[keep]
        if colors is not None:
            colors = colors[keep]
    after_outlier = len(pts)
    if voxel > 0 and len(pts):
        vox = np.floor(pts / voxel).astype(np.int64)
        _, inverse = np.unique(vox, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        centroids = np.column_stack([np.bincount(inverse, weights=pts[:, c]) / counts for c in range(3)])
        distance2 = np.sum((pts - centroids[inverse]) ** 2, axis=1)
        order = np.lexsort((np.arange(len(pts)), distance2, inverse))
        chosen = np.sort(order[np.unique(inverse[order], return_index=True)[1]])
        pts = pts[chosen]
        if colors is not None:
            colors = colors[chosen]
    report = {"original": raw_count, "finite": finite_count,
              "after_outlier": after_outlier, "after_voxel": int(len(pts)),
              "removed_nonfinite": raw_count - finite_count, "removed_outlier": outliers,
              "removed_voxel": after_outlier - len(pts),
              "removed_fraction": float(1 - len(pts) / raw_count),
              "input_spacing_units": spacing, "output_spacing_units": _spacing(pts),
              "color_preserved": colors is not None, "voxel_representative": "nearest_input_point_medoid",
              "seed": seed}
    return pts, colors, report


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(normal, dtype=float)
    z = z / np.linalg.norm(z)
    axis = np.eye(3)[int(np.argmin(np.abs(z)))]
    u = np.cross(z, axis)
    u /= np.linalg.norm(u)
    return u, np.cross(z, u)


def _normal(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v)
    if length <= 1e-12:
        raise ValueError("Degenerate plane")
    v = v / length
    main = int(np.argmax(np.abs(v)))
    return -v if v[main] < 0 else v


def _plane_from_sample(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float] | None:
    n = np.cross(b - a, c - a)
    if np.linalg.norm(n) < 1e-10:
        return None
    n = _normal(n)
    return n, float(-n @ a)


def _make_patch(points: np.ndarray, indices: np.ndarray, label: str, threshold: float) -> dict | None:
    p = points[indices]
    if len(p) < 3:
        return None
    center = p.mean(axis=0)
    _, singular, v = np.linalg.svd(p - center, full_matrices=False)
    if len(singular) < 2 or singular[1] < 1e-12:
        return None
    n = _normal(v[-1])
    d = -float(n @ center)
    residual = np.abs(p @ n + d)
    u, w = plane_basis(n)
    local = np.column_stack(((p - center) @ u, (p - center) @ w))
    try:
        hull = ConvexHull(local)
        boundary = local[hull.vertices]
        area = float(hull.volume)
        halfspaces = hull.equations.tolist()
    except QhullError:
        return None
    if area <= 1e-12:
        return None
    boundary_xyz = center + boundary[:, :1] * u + boundary[:, 1:] * w
    signature = hashlib.sha256(np.round(np.r_[n, d, center], 5).tobytes()).hexdigest()[:16]
    return {"plane_id": f"pl-{label}", "stable_geometry_signature": signature,
            "coefficients": [*n.tolist(), d], "normal": n.tolist(),
            "centroid": center.tolist(), "finite_boundary_xyz": boundary_xyz.tolist(),
            "finite_boundary_uv": boundary.tolist(), "basis_u": u.tolist(), "basis_v": w.tolist(),
            "halfspaces_uv": halfspaces, "area_reconstruction_units_squared": area,
            "point_count": int(len(indices)), "support_indices": indices.tolist(),
            "residual_median_units": float(np.median(residual)),
            "residual_p95_units": float(np.quantile(residual, .95)),
            "fit_state": "candidate_observed_patch", "threshold_units": threshold,
            "infinite_plane_confirmed": False}


def normal_diagnostics(points: Any, *, neighborhood_k: int = 18, max_samples: int = 10000) -> dict:
    """Sample local PCA normals/curvature; no invented global orientation."""
    pts = _xyz(points)
    ids = np.linspace(0, len(pts) - 1, min(len(pts), max_samples), dtype=int)
    if len(pts) < 4:
        return {"status": "insufficient_points", "samples": 0}
    k = min(neighborhood_k, len(pts))
    _, near = cKDTree(pts).query(pts[ids], k=k, workers=1)
    local = pts[near]
    center = local.mean(axis=1, keepdims=True)
    diff = local - center
    cov = np.einsum("nki,nkj->nij", diff, diff) / k
    eigenvalue, eigenvectors = np.linalg.eigh(cov)
    curvature = eigenvalue[:, 0] / np.maximum(eigenvalue.sum(axis=1), 1e-15)
    normals = eigenvectors[:, :, 0]
    return {"status": "sampled_pca_unoriented", "samples": len(ids),
            "sampled_fraction": float(len(ids) / len(pts)),
            "curvature_median": float(np.median(curvature)),
            "curvature_p95": float(np.quantile(curvature, .95)),
            "normal_orientation_verified": False,
            "normals_sample_indices": ids, "normals_sample_xyz": normals}


def reuse_stage7_planes(points: Any, stage7_planes_json: Path, *,
                       threshold: float, min_points: int) -> list[dict]:
    """Reuse Stage-7's existing guided RANSAC segmentation and each plane PLY.

    Plane clouds may be voxel centroids, not identical source rows; attach to
    nearest existing clean source point only within a documented tolerance.
    No point may be assigned to multiple plane records.
    """
    pts = _xyz(points)
    payload = json.loads(stage7_planes_json.read_text(encoding="utf-8"))
    records = payload.get("planes")
    if not isinstance(records, list) or not records:
        raise ValueError("Provided Stage7 plane JSON has no reusable records")
    tree = cKDTree(pts)
    tolerance = max(threshold * 2, _spacing(pts) * 2)
    already = set()
    patches = []
    for rec in records:
        loc = Path(str(rec.get("cloud_path", "")))
        if not loc.is_absolute():
            loc = (stage7_planes_json.parent / loc).resolve()
        if not loc.is_file() or loc.stat().st_size == 0:
            continue
        cloud, _ = _ply_read(loc)
        dist, matched = tree.query(cloud, k=1, workers=1)
        indices = np.unique(matched[dist <= tolerance])
        indices = np.asarray([i for i in indices if int(i) not in already], dtype=int)
        if len(indices) < min_points:
            continue
        patch = _make_patch(pts, indices, f"{len(patches)+1:03d}", threshold)
        if patch is None:
            continue
        patch["source_stage7_plane_id"] = rec.get("plane_id")
        patch["source_stage7_plane_label"] = rec.get("label")
        patch["extraction_source"] = "reused_stage7_plane_cloud"
        patches.append(patch)
        already.update(int(i) for i in indices)
    if not patches:
        raise ValueError("No sufficiently supported observed Stage-7 plane clouds")
    patches.sort(key=lambda a: tuple(np.round(a["centroid"], 5)))
    for i, p in enumerate(patches, 1):
        p["plane_id"] = f"pl-{i:03d}"
    return patches


def extract_planes(points: Any, *, threshold: float, min_points: int = 40,
                   max_planes: int = 12, trials: int = 250, seed: int = 42) -> list[dict]:
    """Deterministic RANSAC + SVD patch refinement and observed convex hull."""
    pts = _xyz(points)
    if threshold <= 0 or min_points < 3:
        raise ValueError("Invalid plane support parameters")
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(pts))
    patches = []
    for rank in range(max_planes):
        if len(remaining) < min_points:
            break
        sample_points = pts[remaining]
        best = np.empty((0,), dtype=int)
        for _ in range(trials):
            sample = rng.choice(len(remaining), size=3, replace=False)
            fitted = _plane_from_sample(*sample_points[sample])
            if fitted is None:
                continue
            n, d = fitted
            support = np.flatnonzero(np.abs(sample_points @ n + d) <= threshold)
            if len(support) > len(best):
                best = support
        if len(best) < min_points:
            break
        # Refit once, then accept only points really near refined plane.
        refit = _make_patch(pts, remaining[best], str(rank + 1).zfill(3), threshold)
        if refit is None:
            break
        n = np.asarray(refit["normal"])
        d = refit["coefficients"][3]
        within = np.abs(sample_points @ n + d) <= threshold
        selected = remaining[within]
        patch = _make_patch(pts, selected, str(rank + 1).zfill(3), threshold)
        if patch is None or len(selected) < min_points:
            break
        # Do not assign a point to more than one plane in this decomposition.
        patches.append(patch)
        remaining = remaining[~within]
    # Deterministic ordering independent of RANSAC discovery order (within a run).
    patches.sort(key=lambda a: tuple(np.round(a["centroid"], 5)))
    for idx, patch in enumerate(patches, 1):
        patch["plane_id"] = f"pl-{idx:03d}"
    return patches


def _line_intersection(p1: dict, p2: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Finite, normalized two-plane intersection (also accepts integer normals)."""
    n1, n2 = (np.asarray(p["normal"], dtype=np.float64) for p in (p1, p2))
    if n1.shape != (3,) or n2.shape != (3,) or not np.isfinite(n1).all() or not np.isfinite(n2).all():
        return None
    lengths = (float(np.linalg.norm(n1)), float(np.linalg.norm(n2)))
    if min(lengths) <= 1e-12:
        return None
    n1, n2 = n1 / lengths[0], n2 / lengths[1]
    direction = np.cross(n1, n2)
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length < math.sin(math.radians(12)):
        return None
    direction /= length
    # Plane offsets are normalized with the normals; no inferred physical scale.
    d1 = float(p1["coefficients"][3]) / lengths[0]
    d2 = float(p2["coefficients"][3]) / lengths[1]
    if not np.isfinite([d1, d2]).all():
        return None
    a = np.vstack((n1, n2, direction))
    b = np.array([-d1, -d2, 0.], dtype=np.float64)
    return np.linalg.solve(a, b), direction


def _supported_edge_intervals(coords: np.ndarray, radius: float) -> list[tuple[float, float, np.ndarray]]:
    """Split near-line samples at *unobserved* gaps; never bridge isolated patches.

    The gap threshold uses the local median positive sampling step and the
    geometry-support radius, not the overall min/max extent of all points.
    """
    if not len(coords):
        return []
    order = np.argsort(coords, kind="stable")
    sorted_coords = coords[order]
    steps = np.diff(sorted_coords)
    positive = steps[steps > 1e-9]
    median_step = float(np.median(positive)) if len(positive) else 0.
    # Cap the inferred spacing; two isolated endpoints are not a continuous observed edge.
    gap_threshold = max(2.5 * radius, min(3.0 * median_step, 4.0 * radius))
    breaks = np.flatnonzero(steps > gap_threshold) + 1
    chunks = np.split(order, breaks)
    return [(float(coords[c].min()), float(coords[c].max()), c) for c in chunks if len(c)]


def extract_edges(points: Any, planes: list[dict], *, radius: float,
                  min_support: int = 3, min_length: float | None = None) -> list[dict]:
    """Finite *candidate* segments supported by samples from BOTH local planes.

    Unlike single-extremum intersections, this can emit two separated edge
    segments for disjoint patches and will not fill an unsampled middle span.
    This remains geometric evidence, not a verified construction edge.
    """
    pts = _xyz(points)
    if not np.isfinite(pts).all():
        raise ValueError("Edge detection requires finite points")
    if not np.isfinite(radius) or radius <= 0 or not isinstance(min_support, int) or min_support < 2:
        raise ValueError("Invalid edge support radius/count")
    threshold_length = 2 * radius if min_length is None else float(min_length)
    if not np.isfinite(threshold_length) or threshold_length <= 0:
        raise ValueError("Invalid minimum segment length")
    edges: list[dict] = []
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            first, second = planes[i], planes[j]
            intersection = _line_intersection(first, second)
            if intersection is None:
                continue
            origin, axis = intersection
            per_plane: list[list[tuple[float, float, np.ndarray]]] = []
            for plane in (first, second):
                indices = np.asarray(plane["support_indices"], dtype=np.int64)
                if indices.ndim != 1 or (len(indices) and (indices.min() < 0 or indices.max() >= len(pts))):
                    raise ValueError("Invalid plane support indices")
                near = np.linalg.norm(np.cross(pts[indices] - origin, axis), axis=1) <= radius
                selected = indices[near]
                if len(selected) < min_support:
                    break
                coord = (pts[selected] - origin) @ axis
                runs = _supported_edge_intervals(coord, radius)
                per_plane.append([(lo, hi, selected[local]) for lo, hi, local in runs
                                  if len(local) >= min_support])
            if len(per_plane) != 2:
                continue
            for a_lo, a_hi, a_ids in per_plane[0]:
                for b_lo, b_hi, b_ids in per_plane[1]:
                    lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
                    if hi - lo < threshold_length:
                        continue
                    # Retain only actual support points in this finite interval.
                    support = []
                    for ids in (a_ids, b_ids):
                        t = (pts[ids] - origin) @ axis
                        ids = ids[(t >= lo - radius) & (t <= hi + radius)]
                        if len(ids) < min_support:
                            break
                        support.append(ids)
                    if len(support) != 2:
                        continue
                    start, end = origin + axis * lo, origin + axis * hi
                    edges.append({"edge_id": f"ed-{len(edges) + 1:03d}",
                                  "plane_ids": [first["plane_id"], second["plane_id"]],
                                  "endpoints_xyz": [start.tolist(), end.tolist()],
                                  "length_reconstruction_units": float(hi - lo),
                                  "support_count_per_plane": [len(x) for x in support],
                                  "support_indices_per_plane": [x.tolist() for x in support],
                                  "support_radius_units": radius,
                                  "fit_state": "locally_supported_candidate",
                                  "confirmed_physical_edge": False})
    return edges


def _point_within_patch(point: np.ndarray, plane: dict, margin: float) -> bool:
    ctr = np.asarray(plane["centroid"])
    rel = point - ctr
    uv = np.array([rel @ np.asarray(plane["basis_u"]), rel @ np.asarray(plane["basis_v"])])
    equations = np.asarray(plane["halfspaces_uv"])
    return bool(np.all(equations[:, :2] @ uv + equations[:, 2] <= margin))


def extract_corners(points: Any, planes: list[dict], edges: list[dict], *,
                    radius: float, min_support: int = 3,
                    max_condition: float = 100.0) -> list[dict]:
    pts = _xyz(points)
    edge_by_plane: dict[frozenset[str], list[dict]] = {}
    for edge in edges:
        edge_by_plane.setdefault(frozenset(edge["plane_ids"]), []).append(edge)
    corners = []
    from itertools import combinations
    for trio in combinations(planes, 3):
        ids = [p["plane_id"] for p in trio]
        a = np.array([p["normal"] for p in trio])
        cond = float(np.linalg.cond(a))
        if not np.isfinite(cond) or cond > max_condition:
            continue  # near-parallel / degenerate trihedral solutions
        candidate = np.linalg.solve(a, -np.array([p["coefficients"][3] for p in trio]))
        within = [_point_within_patch(candidate, p, radius) for p in trio]
        support = []
        for p in trio:
            indices = np.asarray(p["support_indices"], dtype=int)
            count = int(np.sum(np.linalg.norm(pts[indices] - candidate, axis=1) <= radius))
            support.append(count)
        # A pair of planes can have several disconnected *finite* edge segments.
        # A corner is observed only if each parent pair has a segment at THIS corner.
        paired = []
        for pair in combinations(ids, 2):
            local = []
            for edge in edge_by_plane.get(frozenset(pair), []):
                e0, e1 = (np.asarray(v, dtype=float) for v in edge["endpoints_xyz"])
                segment = e1 - e0
                length = float(np.linalg.norm(segment))
                if length <= 1e-12:
                    continue
                offset = candidate - e0
                t = float(np.dot(offset, segment) / (length * length))
                projected = e0 + np.clip(t, 0., 1.) * segment
                distance = float(np.linalg.norm(candidate - projected))
                if distance <= radius:
                    local.append((distance, edge))
            paired.append(min(local, key=lambda x: x[0])[1] if local else None)
        edge_contains = all(e is not None for e in paired)
        observed = all(within) and all(s >= min_support for s in support) and bool(edge_contains)
        # Finite-support failure is not promoted to a measured corner.
        state = "observed_supported" if observed else "candidate_extrapolated"
        corners.append({"corner_id": f"co-{len(corners) + 1:03d}",
                        "xyz": candidate.tolist(), "plane_ids": ids,
                        "edge_ids": [e["edge_id"] for e in paired if e is not None],
                        "support_near_corner_per_plane": support,
                        "inside_observed_patch_per_plane": within,
                        "trihedral_condition": cond, "state": state,
                        "multiscale_confirmed": False,
                        "camera_projection_verified": False})
    return corners


def project_corner(camera: dict, xyz: np.ndarray) -> dict:
    """Only FOV projection, never visibility/occlusion certification."""
    if camera.get("frame_convention") != "colmap_opencv_world_to_camera" or not camera.get("calibration_verified", False):
        return {"view_id": camera.get("view_id"), "status": "unverified_camera"}
    pose = np.asarray(camera.get("world_to_camera_4x4"), dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        return {"view_id": camera.get("view_id"), "status": "invalid_camera"}
    fx, fy, cx, cy = [float(camera[k]) for k in ("fx", "fy", "cx", "cy")]
    width, height = int(camera["width"]), int(camera["height"])
    if not all(np.isfinite([fx, fy, cx, cy])) or min(fx, fy) <= 0 or min(width, height) < 1:
        return {"view_id": camera.get("view_id"), "status": "invalid_camera"}
    cam = pose @ np.r_[xyz, 1.0]
    if cam[2] <= 0:
        return {"view_id": camera.get("view_id"), "status": "outside_fov"}
    x = fx * cam[0] / cam[2] + cx
    y = fy * cam[1] / cam[2] + cy
    return {"view_id": camera.get("view_id"), "pixel_xy": [float(x), float(y)],
            "status": "in_fov_occlusion_unverified" if 0 <= x < width and 0 <= y < height else "outside_fov"}


def _ply_read(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        import trimesh
    except ImportError as e:
        raise RuntimeError("trimesh is required for PLY input; pip install trimesh") from e
    obj = trimesh.load(str(path), process=False)
    if isinstance(obj, trimesh.Scene):
        raise ValueError("PLY scenes containing separate meshes are not a point cloud")
    points = np.asarray(obj.vertices, dtype=np.float64)
    colors = getattr(obj, "colors", None)
    if colors is None and hasattr(obj, "visual"):
        colors = getattr(obj.visual, "vertex_colors", None)
    colors = np.asarray(colors)[:, :3] if colors is not None and len(colors) == len(points) else None
    return points, colors


def _ply_write(path: Path, points: np.ndarray, colors: np.ndarray | None):
    n = len(points)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header"]
    dtype = [(k, "<f4") for k in ("x", "y", "z")]
    if colors is not None:
        dtype += [(k, "u1") for k in ("red", "green", "blue")]
    a = np.empty(n, dtype=dtype)
    for i, k in enumerate(("x", "y", "z")):
        a[k] = points[:, i]
    if colors is not None:
        col = np.asarray(colors, dtype=float)
        if np.max(col) <= 1.01:
            col *= 255
        for i, k in enumerate(("red", "green", "blue")):
            a[k] = np.uint8(np.clip(col[:, i], 0, 255))
    with path.open("xb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(a.tobytes())


def _preview(points: np.ndarray, planes: list[dict], edges: list[dict], corners: list[dict], output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    selected = np.arange(len(points))[::max(1, len(points) // 35000)]
    ax.scatter(points[selected, 0], points[selected, 1], s=.5, alpha=.45)
    for p in planes:
        b = np.asarray(p["finite_boundary_xyz"])
        ax.plot(np.r_[b[:, 0], b[0, 0]], np.r_[b[:, 1], b[0, 1]], lw=1)
    for e in edges:
        a = np.asarray(e["endpoints_xyz"])
        ax.plot(a[:, 0], a[:, 1], lw=1.8)
    for c in corners:
        x = c["xyz"]
        ax.scatter([x[0]], [x[1]], marker="x", s=45)
    ax.set_xlabel("X (source reconstruction units)")
    ax.set_ylabel("Y (source reconstruction units)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("XY preview: local observed plane boundaries / candidate edges / corners")
    fig.tight_layout()
    fig.savefig(output, dpi=145)
    plt.close(fig)
    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(projection="3d")
    ax3.scatter(points[selected, 0], points[selected, 1], points[selected, 2], s=.3, alpha=.3)
    for p in planes:
        b = np.asarray(p["finite_boundary_xyz"])
        closed = np.vstack([b, b[0]])
        ax3.plot(closed[:, 0], closed[:, 1], closed[:, 2], lw=1.2)
    for e in edges:
        a = np.asarray(e["endpoints_xyz"])
        ax3.plot(a[:, 0], a[:, 1], a[:, 2], lw=2)
    for c in corners:
        x = c["xyz"]
        ax3.scatter(*x, marker="x", s=50)
    ax3.set_title("3D observed finite patches / supported feature candidates")
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(output.parent / "xyz_geometry_preview.png", dpi=145)
    plt.close(fig)


def run_ready(source: Path, root: Path, run_id: str, *, threshold: float = 0.03,
              voxel: float = 0., outlier_mad: float = 0.,
              min_plane_points: int = 40, max_planes: int = 12,
              trials: int = 250, seed: int = 42, camera_json: Path | None = None,
              source_frame: str = "unknown_reconstruction_frame",
              source_units: str = "unknown_reconstruction_units", preview: bool = True,
              max_feature_points: int = 125000, gravity_up: tuple[float, float, float] | None = None,
              stage7_planes_json: Path | None = None) -> dict:
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("Run ID must match geom-ready-*")
    source, root = source.resolve(), root.resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    target = root / "runs" / run_id / "geometry"
    if target.exists():
        raise FileExistsError("Refusing to replace existing run: " + str(target))
    raw, rgb = _ply_read(source)
    pts, rgb, cleaning = clean_points(raw, rgb, voxel=voxel, outlier_mad=outlier_mad, seed=seed)
    if max_feature_points < min_plane_points:
        raise ValueError("max_feature_points must be >= min_plane_points")
    selection = np.linspace(0, len(pts) - 1, min(len(pts), max_feature_points), dtype=int)
    reduced = pts[selection]
    if stage7_planes_json is not None:
        patches = reuse_stage7_planes(pts, stage7_planes_json, threshold=threshold,
                                     min_points=min_plane_points)
    else:
        patches = extract_planes(reduced, threshold=threshold, min_points=min_plane_points,
                                 max_planes=max_planes, trials=trials, seed=seed)
        for patch in patches:
            patch["support_indices"] = selection[np.asarray(patch["support_indices"], dtype=int)].tolist()
    if gravity_up is None:
        for patch in patches:
            patch["orientation_class"] = "unknown_unverified_gravity"
    else:
        up = np.asarray(gravity_up, dtype=float)
        if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-12:
            raise ValueError("gravity_up must be a finite nonzero xyz vector")
        up /= np.linalg.norm(up)
        for patch in patches:
            vertical_component = abs(float(np.dot(up, patch["normal"])))
            patch["orientation_class"] = ("horizontal_floor_or_ceiling" if vertical_component > .92
                                           else "vertical_wall_candidate" if vertical_component < .2
                                           else "sloped_unclassified")
    diagnostics = normal_diagnostics(pts)
    normal_samples = diagnostics.pop("normals_sample_xyz")
    normal_indices = diagnostics.pop("normals_sample_indices")
    spacing = cleaning["output_spacing_units"]
    radius = max(threshold * 4, spacing * 2.5)
    edges = extract_edges(pts, patches, radius=radius, min_support=2)
    corners = extract_corners(pts, patches, edges, radius=radius * 1.25, min_support=2)
    # Multiscale stability of trihedral coordinates; each level independent.
    coarse = reduced[::2]
    if len(coarse) >= min_plane_points * 3 and corners:
        second = extract_planes(coarse, threshold=threshold * 1.5,
                                min_points=max(12, min_plane_points // 2),
                                max_planes=max_planes, trials=trials, seed=seed)
        se = extract_edges(coarse, second, radius=radius * 1.5, min_support=2)
        sc = extract_corners(coarse, second, se, radius=radius * 1.6, min_support=2)
        for c in corners:
            diffs = [float(np.linalg.norm(np.asarray(c["xyz"]) - np.asarray(x["xyz"]))) for x in sc]
            drift = min(diffs) if diffs else None
            c["multiscale_drift_units"] = drift
            c["multiscale_confirmed"] = drift is not None and drift < 2.5 * radius
            if c["state"] == "observed_supported" and not c["multiscale_confirmed"]:
                c["state"] = "unknown"
    if camera_json is not None:
        cameras = json.loads(camera_json.read_text(encoding="utf-8"))
        if not isinstance(cameras.get("cameras", []), list):
            raise ValueError("camera_json requires cameras[]")
        for c in corners:
            c["image_projections"] = [project_corner(cam, np.asarray(c["xyz"])) for cam in cameras["cameras"]]
            c["camera_projection_verified"] = any(x["status"] == "in_fov_occlusion_unverified" for x in c["image_projections"])
    target.mkdir(parents=True, exist_ok=False)
    np.save(target / "clean_cloud.npy", pts.astype("<f8"), allow_pickle=False)
    np.savez_compressed(target / "normal_samples.npz", indices=normal_indices,
                        xyz=normal_samples)
    _ply_write(target / "clean_cloud.ply", pts, rgb)
    data = {"planes.json": {"planes": patches}, "edges.json": {"edges": edges},
            "corners.json": {"corners": corners}}
    for name, payload in data.items():
        (target / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    outputs = {name: {"path": (Path("runs") / run_id / "geometry" / name).as_posix(),
                      "sha256": digest(target / name), "bytes": (target / name).stat().st_size}
               for name in ["clean_cloud.ply", "clean_cloud.npy", "planes.json", "edges.json", "corners.json", "normal_samples.npz"]}
    if preview:
        try:
            _preview(pts, patches, edges, corners, target / "xy_geometry_preview.png")
            for preview_name in ("xy_geometry_preview.png", "xyz_geometry_preview.png"):
                outputs[preview_name] = {"path": (Path("runs") / run_id / "geometry" / preview_name).as_posix(),
                                         "sha256": digest(target / preview_name)}
        except ImportError:
            pass
    manifest = {"schema_version": SCHEMA, "run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": digest(source), "source_path": str(source),
                "source_camera_sha256": digest(camera_json) if camera_json else None,
                "source_stage7_planes_sha256": digest(stage7_planes_json) if stage7_planes_json else None,
                "source_frame": source_frame, "coordinate_frame": source_frame, "units": source_units,
                "metric_scale_verified": False, "registration_verified": False,
                "method": "finite_local_ransac_plane_patches_intersections_observation_gated",
                "sampling_policy": "observed_cloud_only_no_unobserved_fill",
                "plane_source": "reused_stage7_plane_clouds" if stage7_planes_json else "research_ransac",
                "original_points": cleaning["original"], "clean_points": len(pts),
                "cleaning": cleaning, "normal_quality": diagnostics,
                "feature_sample_count": int(len(reduced)), "plane_count": len(patches), "edge_count": len(edges),
                "corner_count": len(corners),
                "supported_corners": sum(c["state"] == "observed_supported" for c in corners),
                "unverified_corners": sum(c["state"] != "observed_supported" for c in corners),
                "parameters": {"threshold_reconstruction_units": threshold, "voxel_reconstruction_units": voxel,
                               "outlier_mad": outlier_mad, "min_plane_points": min_plane_points,
                               "max_planes": max_planes, "trials": trials, "seed": seed,
                               "max_feature_points": max_feature_points,
                               "gravity_up_operator_supplied": list(gravity_up) if gravity_up else None},
                "warnings": ["Unverified physical scale and reference registration",
                             "RGB appearance and Gaussians are not measurement-grade geometric ground truth",
                             "Convex hull can bridge missing scan surface areas; no complete surface is implied"],
                "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                             "scipy": __import__("scipy").__version__, "platform": platform.platform()},
                "outputs": outputs}
    (target / "comparison_ready.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Geometry after Stage 7: conservative comparison-ready observed planes/edges/corners")
    p.add_argument("--input-ply", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--threshold", type=float, default=.03, help="Original reconstruction UNITS, NOT metres unless independently scaled")
    p.add_argument("--voxel", type=float, default=0.)
    p.add_argument("--outlier-mad", type=float, default=0.)
    p.add_argument("--min-plane-points", type=int, default=40)
    p.add_argument("--max-planes", type=int, default=12)
    p.add_argument("--trials", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--camera-json", type=Path)
    p.add_argument("--source-frame", default="unknown_reconstruction_frame")
    p.add_argument("--source-units", default="unknown_reconstruction_units")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--max-feature-points", type=int, default=125000)
    p.add_argument("--gravity-up", type=float, nargs=3)
    p.add_argument("--stage7-planes-json", type=Path, help="Reuse existing Stage-7 guided RANSAC plane clouds")
    a = p.parse_args(argv)
    try:
        result = run_ready(a.input_ply, a.project_root, a.run_id, threshold=a.threshold,
                           voxel=a.voxel, outlier_mad=a.outlier_mad, min_plane_points=a.min_plane_points,
                           max_planes=a.max_planes, trials=a.trials, seed=a.seed,
                           camera_json=a.camera_json, source_frame=a.source_frame,
                           source_units=a.source_units, preview=not a.no_preview,
                           max_feature_points=a.max_feature_points,
                           gravity_up=tuple(a.gravity_up) if a.gravity_up else None,
                           stage7_planes_json=a.stage7_planes_json)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(json.dumps({"status": "failed_or_blocked", "error": str(exc)}))
        return 2
    print(json.dumps({"status": "created", "run_id": result["run_id"],
                      "planes": result["plane_count"], "edges": result["edge_count"],
                      "corners": result["corner_count"], "metric_scale_verified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
