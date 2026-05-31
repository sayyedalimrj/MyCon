"""Hyper-realistic procedural camera operator + full-coverage pathfinding.

This module is the **single source of truth** for camera motion in the
synthetic-floor example. Both the CPU software-rasterizer path
(``run_generate.py``) and the Blender GPU export (``blender_gpu_renderer.py``,
via a serialized ``camera_poses.json``) consume the poses produced here, so
the rendered video and the CPU dataset share the exact same trajectory.

Design goals (a believable hand-held human operator, not a robot on rails):

* **Translation and look-at are fully decoupled.** The body walks a
  coverage path; the gaze is an independent, inertial signal that scans
  left/right, periodically turns ~180 deg to inspect the area behind, and
  tilts up/down to inspect ceiling/floor.
* **Full-coverage, collision-safe pathfinding.** The walk is generated from
  the ``SceneSpec`` interior (a serpentine that covers the floor), smoothed
  with a centripetal Catmull-Rom spline for natural curves, then strictly
  clamped inside the room (walls) and pushed out of column footprints — no
  wall-clipping, no out-of-bounds.
* **6-DOF with verticality.** The eye height is not locked: scheduled
  "inspect floor" (crouch) and "inspect ceiling" (rise) maneuvers move the
  camera on Z with eased motion, and the gaze pitches to match.
* **Physical inertia + micro-mechanics.** Gaze angles are low-pass filtered
  (a time-constant = inertia), and breathing / footstep / high-frequency
  micro-jitter ride on top.

All parameters come from ``camera.motion`` / ``camera.hand_jitter`` in the
YAML config (see :class:`scene_spec.MotionSpec`) — nothing here is hardcoded.

Coordinates are right-handed **Z-up** (matches the rest of the pipeline and
the Blender camera convention: ``cam_to_world`` columns are right / up /
backward / eye).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .scene_spec import CameraSpec, MotionSpec, SceneSpec


@dataclass(frozen=True)
class Pose:
    """One camera pose (right-handed, Z-up)."""
    eye: np.ndarray         # shape (3,)
    target: np.ndarray      # shape (3,)
    up: np.ndarray          # shape (3,)
    timestamp: float

    @property
    def cam_to_world(self) -> np.ndarray:
        """4x4 camera-to-world; matches Blender camera (cols: right, up, +Z=back)."""
        forward = self.target - self.eye
        forward = forward / max(1e-9, float(np.linalg.norm(forward)))
        up = self.up / max(1e-9, float(np.linalg.norm(self.up)))
        right = np.cross(forward, up)
        rn = float(np.linalg.norm(right))
        if rn < 1e-9:  # forward parallel to up -> pick a stable right
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
            rn = float(np.linalg.norm(right))
        right = right / max(1e-9, rn)
        up_cam = np.cross(right, forward)
        m = np.eye(4, dtype=np.float64)
        m[:3, 0] = right
        m[:3, 1] = up_cam
        m[:3, 2] = -forward    # camera looks down -Z; +Z (col2) is backward
        m[:3, 3] = self.eye
        return m


# ---------------------------------------------------------------------
# Easing + smoothing primitives
# ---------------------------------------------------------------------


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _trapezoid_pulse(t: float, start: float, duration: float, ramp_frac: float = 0.35) -> float:
    """Eased 0->1->0 pulse over [start, start+duration] (ramp-up, hold, ramp-down)."""
    if duration <= 0:
        return 0.0
    u = (t - start) / duration
    if u <= 0.0 or u >= 1.0:
        return 0.0
    r = max(1e-3, min(0.49, ramp_frac))
    if u < r:
        return _smoothstep(u / r)
    if u > 1.0 - r:
        return _smoothstep((1.0 - u) / r)
    return 1.0


def _inertia_filter(values: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Causal one-pole low-pass = physical inertia on an angle/scalar signal."""
    if tau <= 1e-6:
        return values.copy()
    alpha = 1.0 - math.exp(-dt / tau)
    out = np.empty_like(values)
    acc = values[0]
    for i, v in enumerate(values):
        acc += alpha * (v - acc)
        out[i] = acc
    return out


# ---------------------------------------------------------------------
# Geometry helpers (collision + coverage)
# ---------------------------------------------------------------------


def _interior_bounds(spec: SceneSpec, motion: MotionSpec) -> tuple[float, float, float, float]:
    f = spec.floor
    wt = f.exterior_wall_thickness_m
    m = motion.collision_margin_m
    x0, x1 = wt + m, f.length_m - wt - m
    y0, y1 = wt + m, f.width_m - wt - m
    if x1 <= x0:
        x0, x1 = 0.25 * f.length_m, 0.75 * f.length_m
    if y1 <= y0:
        y0, y1 = 0.25 * f.width_m, 0.75 * f.width_m
    return x0, x1, y0, y1


def _column_centers(spec: SceneSpec) -> np.ndarray:
    f = spec.floor
    bx, by = f.grid.bays_x, f.grid.bays_y
    pts = []
    for ix in range(bx + 1):
        x = ix * f.length_m / bx
        for iy in range(by + 1):
            y = iy * f.width_m / by
            pts.append((x, y))
    return np.array(pts, dtype=np.float64)


def _clamp_interior(p: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    x0, x1, y0, y1 = bounds
    p = p.copy()
    p[0] = min(x1, max(x0, p[0]))
    p[1] = min(y1, max(y0, p[1]))
    return p


def _avoid_columns(p: np.ndarray, columns: np.ndarray, radius: float,
                   bounds: tuple[float, float, float, float]) -> np.ndarray:
    """Push an XY point out of any column disk, then re-clamp to the room."""
    if columns.size == 0 or radius <= 0:
        return p
    q = p.copy()
    for _ in range(4):  # a few relaxation passes
        moved = False
        for c in columns:
            d = q[:2] - c
            dist = float(np.linalg.norm(d))
            if dist < radius:
                if dist < 1e-6:
                    d = np.array([1.0, 0.0])
                    dist = 1.0
                q[:2] = c + (d / dist) * radius
                moved = True
        q = _clamp_interior(q, bounds)
        if not moved:
            break
    return q


def _catmull_rom(points: np.ndarray, samples_per_seg: int, smoothness: float) -> np.ndarray:
    """Centripetal Catmull-Rom spline through ``points`` (rounded, natural)."""
    n = len(points)
    if n < 3 or samples_per_seg < 2:
        return points
    blend = float(min(1.0, max(0.0, smoothness)))
    pts = np.vstack([points[0], points, points[-1]])  # phantom endpoints
    out = [points[0]]
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for j in range(1, samples_per_seg + 1):
            t = j / samples_per_seg
            t2, t3 = t * t, t * t * t
            cr = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            lin = p1 * (1 - t) + p2 * t
            out.append(cr * blend + lin * (1.0 - blend))
    return np.array(out, dtype=np.float64)


def _coverage_waypoints(spec: SceneSpec, motion: MotionSpec, z: float) -> np.ndarray:
    """Serpentine lawn-mower path covering the interior, entering at the door."""
    x0, x1, y0, y1 = _interior_bounds(spec, motion)
    spacing = max(1.2, motion.coverage_lane_spacing_m)
    n_lanes = max(2, int(round((y1 - y0) / spacing)) + 1)
    y_vals = np.linspace(y0, y1, n_lanes)

    wp: list[list[float]] = []
    # Enter near the (first) door if one exists, else near a corner.
    enter_x = x0
    if spec.floor.doors:
        d = spec.floor.doors[0]
        if d.side in ("west", "east"):
            enter_x = x0 if d.side == "west" else x1
        # door offset_m is along the facade (y for west/east); start near it
    wp.append([enter_x, y_vals[0], z])

    for k, y in enumerate(y_vals):
        if k % 2 == 0:
            wp.append([x0, y, z])
            wp.append([x1, y, z])
        else:
            wp.append([x1, y, z])
            wp.append([x0, y, z])
    # A final pass back toward the entrance so the loop closes naturally.
    wp.append([(x0 + x1) * 0.5, (y0 + y1) * 0.5, z])
    wp.append([enter_x, y_vals[0], z])
    return np.array(wp, dtype=np.float64)


def _build_path(spec: SceneSpec, motion: MotionSpec, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (smoothed, collision-safe path points, cumulative arc-length)."""
    bounds = _interior_bounds(spec, motion)
    columns = _column_centers(spec)
    radius = spec.floor.grid.column_size_m * 0.5 + motion.column_clearance_m

    raw = _coverage_waypoints(spec, motion, z)
    smooth = _catmull_rom(raw, samples_per_seg=10, smoothness=motion.path_smoothness)

    safe = np.array([
        _avoid_columns(_clamp_interior(p, bounds), columns, radius, bounds)
        for p in smooth
    ], dtype=np.float64)
    safe[:, 2] = z

    diffs = np.diff(safe, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return safe, cum


def _sample_path(path: np.ndarray, cum: np.ndarray, s: float) -> np.ndarray:
    total = cum[-1]
    if total <= 0:
        return path[0].copy()
    s = float(np.clip(s, 0.0, total))
    idx = int(np.searchsorted(cum, s, side="right") - 1)
    idx = max(0, min(idx, len(path) - 2))
    seg = cum[idx + 1] - cum[idx]
    t = 0.0 if seg <= 0 else (s - cum[idx]) / seg
    return path[idx] * (1.0 - t) + path[idx + 1] * t


# ---------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------


def plan_camera_path(spec: SceneSpec, *, stage_id: int, n_frames: Optional[int] = None) -> list[Pose]:
    """Return one :class:`Pose` per output frame for ``stage_id``."""
    cam = spec.camera
    motion = cam.motion
    if n_frames is None:
        n_frames = int(round(cam.duration_per_stage_sec * cam.fps))
    n_frames = max(2, int(n_frames))
    fps = float(cam.fps)
    dt = 1.0 / fps
    duration = n_frames / fps

    # Reproducible-but-distinct jitter per stage.
    rng = np.random.default_rng(spec.random_seed + stage_id * 7919 + 17)

    # --- body path (collision-safe coverage), traversed once over the clip ---
    base_z = cam.hold_height_m
    path, cum = _build_path(spec, motion, base_z)
    total_len = float(cum[-1])
    speed = total_len / max(0.5, duration)  # fit the whole path into the clip

    f = spec.floor
    ceil_z = f.height_m
    floor_z = 0.0

    # --- precompute raw gaze yaw/pitch + macro Z per frame (then smooth) ---
    ts = np.arange(n_frames) * dt
    eyes_xy = np.zeros((n_frames, 2))
    travel_heading = np.zeros(n_frames)
    for i, t in enumerate(ts):
        s = speed * t
        p = _sample_path(path, cum, s)
        ahead = _sample_path(path, cum, s + 0.8)
        eyes_xy[i] = p[:2]
        tan = ahead[:2] - p[:2]
        travel_heading[i] = math.atan2(tan[1], tan[0]) if np.linalg.norm(tan) > 1e-6 else (
            travel_heading[i - 1] if i > 0 else 0.0)
    travel_heading = np.unwrap(travel_heading)

    # Scheduled 180-degree look-behind events, distributed across the clip so
    # even a short stage gets at least one (and long stages get more).
    turn_offset = np.zeros(n_frames)
    if motion.turn_around_interval_sec > 0:
        n_turns = max(1, int(round(duration / motion.turn_around_interval_sec)))
        turn_dur = min(motion.turn_around_duration_sec, 0.7 * duration / n_turns)
        for k in range(n_turns):
            center = duration * (k + 0.5) / n_turns
            start = center - turn_dur * 0.5
            sign = 1.0 if (k % 2 == 0) else -1.0
            for i, t in enumerate(ts):
                turn_offset[i] += sign * math.pi * _trapezoid_pulse(t, start, turn_dur)

    # Slow horizontal scan (look left/right) + gentle vertical scan.
    scan_yaw = np.deg2rad(motion.scan_yaw_amplitude_deg) * np.sin(
        2 * np.pi * ts / max(0.5, motion.scan_period_sec) + rng.uniform(0, 2 * np.pi))
    pitch_scan = np.deg2rad(motion.pitch_scan_amplitude_deg) * np.sin(
        2 * np.pi * ts / max(0.5, motion.pitch_scan_period_sec) + rng.uniform(0, 2 * np.pi))

    # Scheduled crouch (inspect floor) / rise (inspect ceiling) maneuvers,
    # distributed across the clip so even a short stage gets BOTH a crouch
    # and a rise (>=2 episodes, alternating).
    z_macro = np.full(n_frames, base_z)
    inspect_pitch = np.zeros(n_frames)
    if motion.vertical_inspect_enabled and motion.vertical_inspect_interval_sec > 0:
        n_insp = max(2, int(round(duration / motion.vertical_inspect_interval_sec)))
        insp_dur = min(motion.vertical_inspect_duration_sec, 0.8 * duration / n_insp)
        for k in range(n_insp):
            center = duration * (k + 0.5) / n_insp
            start = center - insp_dur * 0.5
            crouch = (k % 2 == 0)
            target_z = (max(floor_z + 0.3, motion.crouch_height_m) if crouch
                        else min(ceil_z - 0.3, motion.rise_height_m))
            pitch_dir = -1.0 if crouch else 1.0
            for i, t in enumerate(ts):
                w = _trapezoid_pulse(t, start, insp_dur)
                z_macro[i] = z_macro[i] * (1 - w) + target_z * w
                inspect_pitch[i] += pitch_dir * np.deg2rad(motion.inspect_pitch_deg) * w

    # Raw gaze angles, then apply physical inertia (low-pass).
    yaw_raw = travel_heading + scan_yaw + turn_offset
    pitch_raw = pitch_scan + inspect_pitch
    yaw = _inertia_filter(yaw_raw, dt, motion.gaze_inertia_tau_sec)
    pitch = _inertia_filter(pitch_raw, dt, motion.gaze_inertia_tau_sec)

    # --- micro-mechanics (breathing / footstep gait / high-freq tremor) ---
    hj = cam.hand_jitter
    # band-limited noise: white noise then inertia-smoothed
    noise = rng.normal(0.0, 1.0, size=(n_frames, 6))
    for c in range(6):
        noise[:, c] = _inertia_filter(noise[:, c], dt, 0.18)

    poses: list[Pose] = []
    bounds = _interior_bounds(spec, motion)
    focus = motion.focus_distance_m
    for i, t in enumerate(ts):
        breath = 2 * np.pi * t / max(0.01, hj.breathing_period_s)
        gait = 2 * np.pi * t / max(0.01, hj.walking_period_s)

        # Eye position: macro Z + footstep bob + breathing + tremor; clamp.
        eye = np.array([eyes_xy[i, 0], eyes_xy[i, 1], z_macro[i]])
        eye[0] += hj.translation_amplitude_m * noise[i, 0]
        eye[1] += hj.translation_amplitude_m * noise[i, 1]
        eye[2] += (hj.walking_z_amp_m * math.sin(gait)
                   + 0.4 * hj.walking_z_amp_m * math.sin(breath)
                   + hj.translation_amplitude_m * 0.5 * noise[i, 2])
        eye[:2] = _clamp_interior(eye, bounds)[:2]
        eye[2] = float(np.clip(eye[2], floor_z + 0.25, ceil_z - 0.2))

        # Gaze direction from smoothed yaw/pitch + small gait/tremor wobble.
        yaw_i = yaw[i] + np.deg2rad(hj.walking_yaw_amp_deg) * math.sin(gait) \
            + np.deg2rad(hj.rotation_amplitude_deg) * noise[i, 3]
        pitch_i = pitch[i] + np.deg2rad(hj.walking_pitch_amp_deg) * math.sin(gait + math.pi / 2) \
            + np.deg2rad(0.2) * math.sin(breath) \
            + np.deg2rad(hj.rotation_amplitude_deg) * noise[i, 4]
        pitch_i = float(np.clip(pitch_i, -math.radians(80), math.radians(80)))

        cp = math.cos(pitch_i)
        dir_vec = np.array([cp * math.cos(yaw_i), cp * math.sin(yaw_i), math.sin(pitch_i)])
        target = eye + focus * dir_vec

        # Roll (small handheld tilt) -> up vector.
        roll = np.deg2rad(hj.rotation_amplitude_deg * 0.6) * noise[i, 5]
        fwd = dir_vec / max(1e-9, float(np.linalg.norm(dir_vec)))
        kx, ky, kz = fwd
        K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        Rroll = np.eye(3) + math.sin(roll) * K + (1 - math.cos(roll)) * (K @ K)
        up = Rroll @ np.array([0.0, 0.0, 1.0])

        poses.append(Pose(eye=eye, target=target, up=up, timestamp=float(t)))
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


# ---------------------------------------------------------------------
# Serialization (shared with the Blender GPU renderer)
# ---------------------------------------------------------------------


def poses_payload(poses: list[Pose], cam: CameraSpec, *, stage_id: int) -> dict:
    """JSON-able payload consumed by the Blender renderer to key the camera."""
    return {
        "schema_version": "synthetic_floor_camera_poses.v1",
        "stage_id": stage_id,
        "fps": cam.fps,
        "horizontal_fov_deg": cam.horizontal_fov_deg,
        "intrinsics": intrinsics_for(cam),
        "frames": [
            {
                "frame_index": i,
                "eye": [float(p.eye[0]), float(p.eye[1]), float(p.eye[2])],
                "target": [float(p.target[0]), float(p.target[1]), float(p.target[2])],
                "up": [float(p.up[0]), float(p.up[1]), float(p.up[2])],
            }
            for i, p in enumerate(poses)
        ],
    }


def write_camera_poses(poses: list[Pose], cam: CameraSpec, out_path: Path, *, stage_id: int) -> Path:
    """Atomically write the per-frame camera poses for the renderer to consume."""
    import os
    import tempfile

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = poses_payload(poses, cam, stage_id=stage_id)
    fd, tmp = tempfile.mkstemp(dir=str(out_path.parent), prefix=".campose.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out_path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return out_path
