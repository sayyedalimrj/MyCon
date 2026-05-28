from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.common.determinism import derived_seed


def _o3d():
    import open3d as o3d
    return o3d


def _sample_points(points: np.ndarray, colors: np.ndarray | None, max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    if len(points) <= max_points:
        return points, colors
    # B5: was np.random.default_rng(75) — literal seed bypassed project.random_seed.
    rng = np.random.default_rng(derived_seed("stage_07_5_vlm_qa.sample_points"))
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], colors[idx] if colors is not None and len(colors) == len(points) else None


def _set_axes(ax: Any, points: np.ndarray, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=28, azim=-55)

    if len(points):
        center = points.mean(axis=0)
        extent = np.ptp(points, axis=0)
        radius = max(float(extent.max()) / 2.0, 1e-6)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)


def _save_text_card(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 6))
    fig.text(0.05, 0.93, title, fontsize=16, weight="bold")
    y = 0.86
    for line in lines:
        fig.text(0.05, y, line, fontsize=10)
        y -= 0.05
        if y < 0.08:
            break
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_clean_cloud(cleaned_cloud: Path, out_path: Path, *, max_points: int = 50000) -> None:
    o3d = _o3d()
    pcd = o3d.io.read_point_cloud(str(cleaned_cloud))
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    pts, colors = _sample_points(pts, colors, max_points)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if colors is not None and len(colors) == len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=colors)
    elif len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=pts[:, 2], cmap="viridis")
    _set_axes(ax, pts, "Stage 7.5 Cleaned Cloud View")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_mesh(mesh_path: Path | None, out_path: Path, *, max_points: int = 50000) -> None:
    if mesh_path is None or not mesh_path.exists() or mesh_path.stat().st_size <= 0:
        _save_text_card(out_path, "Stage 7.5 Mesh View", ["Mesh file is missing."])
        return

    o3d = _o3d()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        _save_text_card(out_path, "Stage 7.5 Mesh View", ["Mesh has zero vertices."])
        return

    verts, _ = _sample_points(verts, None, max_points)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=1, c=verts[:, 2], cmap="plasma")
    _set_axes(ax, verts, "Stage 7.5 Mesh Vertex View")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_planes(cleaned_cloud: Path, planes_json: Path | None, out_path: Path, *, max_points: int = 25000) -> None:
    import json

    o3d = _o3d()
    pcd = o3d.io.read_point_cloud(str(cleaned_cloud))
    pts = np.asarray(pcd.points)
    pts, _ = _sample_points(pts, None, max_points)

    records: list[dict[str, Any]] = []
    if planes_json is not None and planes_json.exists() and planes_json.stat().st_size > 0:
        payload = json.loads(planes_json.read_text(encoding="utf-8"))
        raw = payload.get("planes") or payload.get("records") or []
        if isinstance(raw, list):
            records = [r for r in raw if isinstance(r, dict)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.15, c=pts[:, 2], cmap="Greys")

    centroids = []
    labels = []
    for r in records:
        c = r.get("centroid")
        if isinstance(c, list) and len(c) == 3:
            centroids.append([float(c[0]), float(c[1]), float(c[2])])
            labels.append(str(r.get("label", r.get("plane_id", "plane"))))

    if centroids:
        cpts = np.asarray(centroids)
        ax.scatter(cpts[:, 0], cpts[:, 1], cpts[:, 2], s=55, marker="o")
        for i, label in enumerate(labels[:20]):
            ax.text(cpts[i, 0], cpts[i, 1], cpts[i, 2], label, fontsize=7)

    _set_axes(ax, pts if len(pts) else np.asarray(centroids), "Stage 7.5 Plane Overlay View")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_overview(metrics: dict[str, Any], quality_gate: dict[str, Any], out_path: Path) -> None:
    cloud = metrics.get("cleaned_cloud", {})
    mesh = metrics.get("mesh", {})
    planes = metrics.get("planes", {})
    lines = [
        f"QA status: {quality_gate.get('status')}",
        f"Confidence: {quality_gate.get('confidence')}",
        f"Cleaned points: {cloud.get('point_count')}",
        f"Finite ratio: {cloud.get('finite_ratio')}",
        f"Has colors: {cloud.get('has_colors')}",
        f"Has normals: {cloud.get('has_normals')}",
        f"Mesh status: {mesh.get('status')}",
        f"Mesh vertices: {mesh.get('vertex_count')}",
        f"Mesh triangles: {mesh.get('triangle_count')}",
        f"Plane count: {planes.get('plane_count')}",
        f"Plane labels: {', '.join(planes.get('labels', []))}",
        f"Failures: {quality_gate.get('failures')}",
        f"Warnings: {quality_gate.get('warnings')}",
    ]
    _save_text_card(out_path, "Stage 7.5 VLM QA Overview", lines)


def render_stage75_views(paths: dict[str, Path], metrics: dict[str, Any], quality_gate: dict[str, Any], *, max_points: int) -> dict[str, str]:
    render_dir = paths["render_dir"]
    render_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "clean_cloud_view": render_dir / "clean_cloud_view.png",
        "mesh_view": render_dir / "mesh_view.png",
        "plane_overlay_view": render_dir / "plane_overlay_view.png",
        "qa_overview": render_dir / "qa_overview.png",
    }

    render_clean_cloud(paths["cleaned_cloud"], outputs["clean_cloud_view"], max_points=max_points)
    render_mesh(paths.get("mesh"), outputs["mesh_view"], max_points=max_points)
    render_planes(paths["cleaned_cloud"], paths.get("planes_json"), outputs["plane_overlay_view"], max_points=max_points)
    render_overview(metrics, quality_gate, outputs["qa_overview"])

    return {k: v.as_posix() for k, v in outputs.items()}
