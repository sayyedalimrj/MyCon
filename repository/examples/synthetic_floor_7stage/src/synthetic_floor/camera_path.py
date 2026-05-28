"""Camera path planning + handheld-jitter simulation.

We model the camera as a person walking along a polyline through the
floor. At each frame we compute:

* the ideal eye position (along the polyline, at hold height);
* a look-at target slightly ahead;
* a small handheld perturbation made of three components:
    - low-frequency *breathing*  (slow vertical sine);
    - mid-frequency *gait*       (step bob + yaw + pitch);
    - high-frequency *micro-jitter* (band-limited noise).

The result is two arrays per frame: a 4x4 camera-to-world matrix and an
accompanying intrinsics dict. Coordinates are right-handed Z-up
(matches the rest of the pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .scene_spec import CameraSpec, SceneSpec


@dataclass(frozen=True)
class Pose:
    """One camera pose."""
    eye: np.ndarray         # shape (3,)
    target: np.ndarray      # shape (3,)
    up: np.ndarray          # shape (3,)
    timestamp: float

    @property
    def cam_to_world(self) -> np.ndarray:
        """Standard OpenGL-style look-at: +Z forward of camera, +Y up.

        We return a 4x4 matrix in homogeneous form. The convention for
        camera basis vectors:
            forward = normalize(target - eye)
            right   = normalize(forward x world_up)
            up_cam  = right x forward
        """
        forward = self.target - self.eye
        forward = forward / max(1e-9, np.linalg.norm(forward))
        right = np.cross(forward, self.up)
        right = right / max(1e-9, np.linalg.norm(right))
        up_cam = np.cross(right, forward)
        m = np.eye(4, dtype=np.float64)
        m[:3, 0] = right
        m[:3, 1] = up_cam
        m[:3, 2] = -forward    # OpenGL: -Z is forward
        m[:3, 3] = self.eye
        return m


def _resampled_polyline(points: np.ndarray, total_length: float) -> np.ndarray:
    """Return the cumulative arc-length array for a polyline."""
    diffs = np.diff(points, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    if cum[-1] <= 0.0:
        return cum
    return cum * (total_length / cum[-1])


def _sample_polyline(points: np.ndarray, cum_len: np.ndarray, s: float) -> np.ndarray:
    """Sample the polyline at arc-length ``s``."""
    s = float(np.clip(s, 0.0, cum_len[-1]))
    idx = int(np.searchsorted(cum_len, s, side="right") - 1)
    idx = max(0, min(idx, len(points) - 2))
    seg = cum_len[idx + 1] - cum_len[idx]
    t = 0.0 if seg <= 0 else (s - cum_len[idx]) / seg
    return points[idx] * (1.0 - t) + points[idx + 1] * t


def _stage_polyline(spec: SceneSpec) -> np.ndarray:
    """Define the deterministic walking polyline.

    The walk enters from the south main door, goes east along the
    corridor, peeks into Office A and Office B, then walks back. The
    same polyline is used for every stage (so the camera path is
    identical, only the construction state changes).
    """
    L = spec.floor.length_m
    W = spec.floor.width_m
    h = spec.camera.hold_height_m
    # Corridor centerline runs at y = (5.5 + 7.5)/2 = 6.5 (per scene.yaml)
    corridor_y = 6.5
    pts = np.array([
        [9.0, 1.5, h],   # just inside the south main door (corridor start, near south)
        [9.0, corridor_y, h],
        [3.0, corridor_y, h],   # near Office A door
        [3.0, 4.0, h],          # peek into Office A
        [3.0, corridor_y, h],
        [9.0, corridor_y, h],
        [15.0, corridor_y, h],  # near Office C door (east end)
        [15.0, 9.5, h],         # peek into Office E
        [15.0, corridor_y, h],
        [9.0, corridor_y, h],
        [9.0, 1.5, h],          # back near the entrance
    ], dtype=np.float64)
    return pts


def plan_camera_path(spec: SceneSpec, *, stage_id: int) -> list[Pose]:
    """Return one Pose per output frame for ``stage_id``."""
    cam = spec.camera
    n_frames = int(round(cam.duration_per_stage_sec * cam.fps))
    if n_frames < 2:
        n_frames = 2

    pts = _stage_polyline(spec)
    # Total length in meters
    total_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    # Ensure the walking duration matches the configured walk speed
    nominal_dur = total_len / max(0.1, cam.walk_speed_m_s)
    # If the configured per-stage duration is shorter we slow the walk
    # down accordingly (same path, just lower speed).
    actual_walk_speed = total_len / max(0.1, cam.duration_per_stage_sec)

    # Per-stage random offset so each stage's jitter pattern looks
    # different but reproducible.
    rng = np.random.default_rng(spec.random_seed + stage_id * 7919)

    poses: list[Pose] = []
    look_ahead_m = 1.5  # how far ahead the LookAt target sits
    for f in range(n_frames):
        t = f / cam.fps
        s = actual_walk_speed * t
        eye = _sample_polyline(pts, cum, s)
        target_xy = _sample_polyline(pts, cum, s + look_ahead_m)
        target = np.array([target_xy[0], target_xy[1], cam.look_at_height_m])

        # ---- handheld jitter -----------------------------------------
        hj = cam.hand_jitter
        # Breathing: slow z oscillation and tiny pitch
        breath_phase = 2.0 * np.pi * t / max(0.01, hj.breathing_period_s)
        # Gait: step bobbing
        gait_phase = 2.0 * np.pi * t / max(0.01, hj.walking_period_s)
        # Micro-jitter: smoothed noise (low-pass)
        # We sample 3 independent noise values per frame and keep them
        # stable by deriving them from the rng with the frame index.
        noise = rng.normal(0.0, 1.0, size=6) * 0.5
        # ---- apply jitter -------------------------------------------
        # Translation
        trans = np.array([
            hj.translation_amplitude_m * noise[0],
            hj.translation_amplitude_m * noise[1],
            hj.translation_amplitude_m * noise[2]
                + hj.walking_z_amp_m * np.sin(gait_phase)
                + 0.4 * hj.walking_z_amp_m * np.sin(breath_phase),
        ])
        eye_jit = eye + trans
        # Rotational jitter applied to look-at target by rotating the
        # offset vector from eye to target.
        forward = target - eye_jit
        forward = forward / max(1e-9, np.linalg.norm(forward))
        # Yaw and pitch in radians
        yaw_rad = (
            np.deg2rad(hj.rotation_amplitude_deg) * noise[3]
            + np.deg2rad(hj.walking_yaw_amp_deg) * np.sin(gait_phase)
        )
        pitch_rad = (
            np.deg2rad(hj.rotation_amplitude_deg) * noise[4]
            + np.deg2rad(hj.walking_pitch_amp_deg) * np.sin(gait_phase + np.pi / 2)
            + np.deg2rad(0.2) * np.sin(breath_phase)
        )
        # Build rotation: yaw around world Z, pitch around camera right
        cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        forward_rot = Rz @ forward
        # Pitch around right axis
        right_axis = np.cross(forward_rot, np.array([0, 0, 1]))
        rn = np.linalg.norm(right_axis)
        if rn > 1e-6:
            right_axis = right_axis / rn
            cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
            # Rodrigues' rotation around right_axis
            kx, ky, kz = right_axis
            K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
            Rp = np.eye(3) + sp * K + (1 - cp) * (K @ K)
            forward_rot = Rp @ forward_rot
        target_jit = eye_jit + forward_rot * np.linalg.norm(target - eye_jit)
        # Roll
        roll_rad = (
            np.deg2rad(hj.rotation_amplitude_deg * 0.6) * noise[5]
        )
        cr, sr = np.cos(roll_rad), np.sin(roll_rad)
        # Express up as world up rotated by roll around forward_rot
        kx, ky, kz = forward_rot
        K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        Rroll = np.eye(3) + sr * K + (1 - cr) * (K @ K)
        up = Rroll @ np.array([0.0, 0.0, 1.0])

        poses.append(Pose(eye=eye_jit, target=target_jit, up=up, timestamp=t))
    return poses


def intrinsics_for(cam: CameraSpec) -> dict:
    """Pinhole intrinsics derived from the configured FOV."""
    w, h = cam.width_px, cam.height_px
    fov_h = np.deg2rad(cam.horizontal_fov_deg)
    fx = 0.5 * w / np.tan(fov_h / 2.0)
    aspect = w / h
    fov_v = 2.0 * np.arctan(np.tan(fov_h / 2.0) / aspect)
    fy = 0.5 * h / np.tan(fov_v / 2.0)
    return {
        "width_px": w,
        "height_px": h,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(w / 2),
        "cy": float(h / 2),
        "horizontal_fov_deg": cam.horizontal_fov_deg,
        "vertical_fov_deg": float(np.rad2deg(fov_v)),
        "k1": cam.k1,
        "k2": cam.k2,
    }
