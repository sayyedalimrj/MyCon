
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]

    def K(self) -> np.ndarray:
        model = self.model.upper()
        p = self.params
        if model == "SIMPLE_PINHOLE":
            f, cx, cy = p[:3]
            fx = fy = f
        elif model == "PINHOLE":
            fx, fy, cx, cy = p[:4]
        elif model in {"SIMPLE_RADIAL", "RADIAL"}:
            f, cx, cy = p[:3]
            fx = fy = f
        elif model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
            fx, fy, cx, cy = p[:4]
        else:
            raise ValueError(f"Unsupported COLMAP camera model for visual anchors: {self.model}")
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    image_name: str

    @property
    def R(self) -> np.ndarray:
        return qvec_to_rotmat(self.qvec)

    def projection_matrix(self, camera: CameraModel) -> np.ndarray:
        Rt = np.hstack([self.R, self.tvec.reshape(3, 1)])
        return camera.K() @ Rt


@dataclass(frozen=True)
class VisualAnchorObservation:
    anchor_id: str
    image_name: str
    u_px: float
    v_px: float
    confidence: float
    method: str = "manual_or_edge"
    notes: str = ""


@dataclass(frozen=True)
class TriangulatedAnchor:
    anchor_id: str
    scan_xyz: np.ndarray
    view_count: int
    mean_reprojection_error_px: float
    max_reprojection_error_px: float
    status: str


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = [float(x) for x in qvec]
    return np.array(
        [
            [
                1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
                2.0 * qx * qy - 2.0 * qz * qw,
                2.0 * qz * qx + 2.0 * qy * qw,
            ],
            [
                2.0 * qx * qy + 2.0 * qz * qw,
                1.0 - 2.0 * qz * qz - 2.0 * qx * qx,
                2.0 * qy * qz - 2.0 * qx * qw,
            ],
            [
                2.0 * qz * qx - 2.0 * qy * qw,
                2.0 * qy * qz + 2.0 * qx * qw,
                1.0 - 2.0 * qy * qy - 2.0 * qx * qx,
            ],
        ],
        dtype=float,
    )


def read_colmap_cameras_txt(path: str | Path) -> dict[int, CameraModel]:
    path = Path(path)
    cameras: dict[int, CameraModel] = {}
    if not path.exists():
        raise FileNotFoundError(f"Missing COLMAP cameras.txt: {path}")

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(x) for x in parts[4:]]
        cameras[camera_id] = CameraModel(camera_id, model, width, height, params)
    return cameras


def read_colmap_images_txt(path: str | Path) -> dict[str, ImagePose]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing COLMAP images.txt: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    images: dict[str, ImagePose] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            try:
                image_id = int(parts[0])
                qvec = np.array([float(x) for x in parts[1:5]], dtype=float)
                tvec = np.array([float(x) for x in parts[5:8]], dtype=float)
                camera_id = int(parts[8])
                image_name = " ".join(parts[9:])
                images[image_name] = ImagePose(image_id, qvec, tvec, camera_id, image_name)
                i += 2
                continue
            except ValueError:
                pass
        i += 1
    return images


def read_visual_anchor_observations_csv(path: str | Path) -> list[VisualAnchorObservation]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing visual anchor observations CSV: {path}")

    observations: list[VisualAnchorObservation] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            anchor_id = str(row.get("anchor_id", "")).strip()
            image_name = str(row.get("image_name", "")).strip()
            if not anchor_id or not image_name:
                continue
            observations.append(
                VisualAnchorObservation(
                    anchor_id=anchor_id,
                    image_name=image_name,
                    u_px=float(row.get("u_px") or row.get("x_px") or 0.0),
                    v_px=float(row.get("v_px") or row.get("y_px") or 0.0),
                    confidence=float(row.get("confidence") or 1.0),
                    method=str(row.get("method") or "manual_or_edge"),
                    notes=str(row.get("notes") or ""),
                )
            )
    return observations


def triangulate_dlt(observations: list[tuple[np.ndarray, float, float]]) -> np.ndarray:
    A: list[np.ndarray] = []
    for P, u, v in observations:
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])
    A_mat = np.asarray(A, dtype=float)
    _, _, vt = np.linalg.svd(A_mat)
    X = vt[-1, :]
    if abs(float(X[3])) < 1e-12:
        raise ValueError("Triangulation failed: homogeneous coordinate is near zero.")
    return X[:3] / X[3]


def project_point(P: np.ndarray, xyz: np.ndarray) -> tuple[float, float]:
    X = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=float)
    h = P @ X
    if abs(float(h[2])) < 1e-12:
        return float("inf"), float("inf")
    return float(h[0] / h[2]), float(h[1] / h[2])


def triangulate_visual_anchors(
    cameras_txt: str | Path,
    images_txt: str | Path,
    observations_csv: str | Path,
    output_picked_csv: str | Path,
    *,
    min_views: int = 2,
    min_confidence: float = 0.0,
    max_mean_reprojection_error_px: float = 12.0,
) -> list[TriangulatedAnchor]:
    cameras = read_colmap_cameras_txt(cameras_txt)
    images = read_colmap_images_txt(images_txt)
    observations = [
        obs for obs in read_visual_anchor_observations_csv(observations_csv)
        if obs.confidence >= min_confidence
    ]

    grouped: dict[str, list[VisualAnchorObservation]] = {}
    for obs in observations:
        grouped.setdefault(obs.anchor_id, []).append(obs)

    results: list[TriangulatedAnchor] = []
    for anchor_id, group in sorted(grouped.items()):
        dlt_obs: list[tuple[np.ndarray, float, float]] = []
        used: list[VisualAnchorObservation] = []

        for obs in group:
            pose = images.get(obs.image_name)
            if pose is None:
                continue
            cam = cameras.get(pose.camera_id)
            if cam is None:
                continue
            dlt_obs.append((pose.projection_matrix(cam), obs.u_px, obs.v_px))
            used.append(obs)

        if len(dlt_obs) < min_views:
            continue

        xyz = triangulate_dlt(dlt_obs)

        errors: list[float] = []
        for P, u, v in dlt_obs:
            uu, vv = project_point(P, xyz)
            errors.append(math.hypot(uu - u, vv - v))

        mean_err = float(np.mean(errors)) if errors else float("inf")
        max_err = float(np.max(errors)) if errors else float("inf")
        status = "ok" if mean_err <= max_mean_reprojection_error_px else "high_reprojection_error"

        results.append(
            TriangulatedAnchor(
                anchor_id=anchor_id,
                scan_xyz=xyz,
                view_count=len(used),
                mean_reprojection_error_px=mean_err,
                max_reprojection_error_px=max_err,
                status=status,
            )
        )

    output_picked_csv = Path(output_picked_csv)
    output_picked_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_picked_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "anchor_id",
            "scan_x_m",
            "scan_y_m",
            "scan_z_m",
            "view_count",
            "mean_reprojection_error_px",
            "max_reprojection_error_px",
            "source",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "anchor_id": item.anchor_id,
                    "scan_x_m": f"{item.scan_xyz[0]:.9f}",
                    "scan_y_m": f"{item.scan_xyz[1]:.9f}",
                    "scan_z_m": f"{item.scan_xyz[2]:.9f}",
                    "view_count": item.view_count,
                    "mean_reprojection_error_px": f"{item.mean_reprojection_error_px:.6f}",
                    "max_reprojection_error_px": f"{item.max_reprojection_error_px:.6f}",
                    "source": "visual_anchor_triangulation",
                    "status": item.status,
                }
            )
    return results


def merge_picked_scan_anchors(
    metric_anchors_template_csv: str | Path,
    picked_scan_anchors_csv: str | Path,
    output_metric_anchors_csv: str | Path,
) -> int:
    template_path = Path(metric_anchors_template_csv)
    picked_path = Path(picked_scan_anchors_csv)

    if not template_path.exists():
        raise FileNotFoundError(f"Missing metric anchor template CSV: {template_path}")
    if not picked_path.exists():
        raise FileNotFoundError(f"Missing picked scan anchor CSV: {picked_path}")

    with picked_path.open("r", encoding="utf-8", newline="") as f:
        picked_rows = {
            str(row.get("anchor_id", "")).strip(): row
            for row in csv.DictReader(f)
            if str(row.get("anchor_id", "")).strip()
        }

    with template_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        template_fields = list(reader.fieldnames or [])
        rows = list(reader)

    required_fields = [
        "scan_x_m",
        "scan_y_m",
        "scan_z_m",
        "scan_anchor_source",
        "scan_anchor_status",
        "scan_anchor_view_count",
        "scan_anchor_mean_reprojection_error_px",
    ]
    fieldnames = template_fields[:]
    for field in required_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    merged = 0
    for row in rows:
        anchor_id = str(row.get("anchor_id", "")).strip()
        picked = picked_rows.get(anchor_id)
        if not picked:
            continue
        row["scan_x_m"] = picked.get("scan_x_m", "")
        row["scan_y_m"] = picked.get("scan_y_m", "")
        row["scan_z_m"] = picked.get("scan_z_m", "")
        row["scan_anchor_source"] = picked.get("source", "visual_anchor_triangulation")
        row["scan_anchor_status"] = picked.get("status", "")
        row["scan_anchor_view_count"] = picked.get("view_count", "")
        row["scan_anchor_mean_reprojection_error_px"] = picked.get("mean_reprojection_error_px", "")
        merged += 1

    output_path = Path(output_metric_anchors_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return merged
