"""Smartphone-style post-processing.

The renderer outputs a clean, noise-free RGB image. This module turns
that into something that looks like a frame from a real smartphone
camera by applying, in order:

1. **Auto-exposure** - a slow EMA on the previous frame's mean
   brightness with a per-frame EV cap.
2. **Motion blur** - mixing the current frame with the previous frame
   weighted by the configured shutter fraction.
3. **Rolling shutter** - top rows are sampled slightly earlier in
   time than bottom rows. We approximate this by row-wise mixing with
   the previous frame.
4. **Sensor noise** - heteroscedastic Gaussian + Poisson shot noise
   scaled by the configured ISO equivalent.
5. **Lens chromatic / vignette finish** - small per-channel gain shift
   plus a soft global vignette (the renderer already applies a base
   vignette; this layer adds film-like grain at the edges).

The simulator keeps a small amount of state across frames (last frame
buffer, last exposure value) so the temporal effects look natural.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene_spec import CameraSpec


@dataclass
class SimulatorState:
    last_frame: np.ndarray | None = None
    log_ev: float = 0.0


class SmartphoneSimulator:
    """Stateful image post-processor."""

    def __init__(self, cam: CameraSpec, *, seed: int):
        self.cam = cam
        self.state = SimulatorState()
        self.rng = np.random.default_rng(seed ^ 0xCAFE)

    def reset(self) -> None:
        self.state = SimulatorState()

    # ------------------------------------------------------------------
    def _auto_exposure(self, frame: np.ndarray) -> np.ndarray:
        cfg = self.cam.exposure
        # Mean brightness of the linear frame
        mean_lin = float(np.mean(frame))
        # Convert current EV state to a scalar gain
        gain = 2.0 ** self.state.log_ev
        scaled = frame * gain
        # Compute the new desired EV based on the post-gain mean vs target.
        post_mean = float(np.mean(scaled))
        if post_mean > 1e-3:
            err = np.log2(cfg.target_brightness / post_mean)
        else:
            err = 1.0
        delta = float(np.clip(cfg.adapt_speed * err, -cfg.max_ev_change_per_frame, cfg.max_ev_change_per_frame))
        self.state.log_ev = float(np.clip(self.state.log_ev + delta, -3.0, 3.0))
        gain = 2.0 ** self.state.log_ev
        return np.clip(frame * gain, 0.0, 1.0)

    def _motion_blur(self, frame: np.ndarray) -> np.ndarray:
        cfg = self.cam.motion_blur
        if self.state.last_frame is None or cfg.shutter_fraction <= 0:
            return frame
        # Approximate continuous shutter by mixing the current frame
        # with the previous one at a fraction equal to shutter_fraction.
        a = float(np.clip(cfg.shutter_fraction, 0.0, 1.0)) * 0.5
        return (1.0 - a) * frame + a * self.state.last_frame

    def _rolling_shutter(self, frame: np.ndarray) -> np.ndarray:
        delay = self.cam.rolling_shutter_row_delay_sec
        if delay <= 0 or self.state.last_frame is None:
            return frame
        H = frame.shape[0]
        # Linear ramp: 0 at top -> small mix at bottom
        max_mix = float(np.clip(delay * self.cam.fps * 8.0, 0.0, 0.18))
        mix = np.linspace(0.0, max_mix, H, dtype=np.float32)[:, None, None]
        return (1.0 - mix) * frame + mix * self.state.last_frame

    def _noise(self, frame: np.ndarray) -> np.ndarray:
        cfg = self.cam.noise
        # Photon (Poisson-like, scale-dependent) + read noise.
        iso_factor = cfg.iso_equivalent / 100.0
        read_sigma = cfg.read_noise_sigma * iso_factor
        photon = self.rng.normal(0.0, np.sqrt(np.maximum(frame, 0.0) * 0.0008 * iso_factor * cfg.photon_scale))
        read = self.rng.normal(0.0, read_sigma, size=frame.shape)
        return np.clip(frame + photon + read, 0.0, 1.0)

    def _color_cast(self, frame: np.ndarray) -> np.ndarray:
        # Slight warm-cool wobble to mimic mobile auto-WB
        gain = np.array([1.01, 1.00, 0.99], dtype=np.float32)
        return np.clip(frame * gain, 0.0, 1.0)

    # ------------------------------------------------------------------
    def __call__(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """Apply the full pipeline. Returns uint8 (H, W, 3)."""
        frame = rgb_uint8.astype(np.float32) / 255.0
        frame = self._auto_exposure(frame)
        frame = self._rolling_shutter(frame)
        frame = self._motion_blur(frame)
        frame = self._color_cast(frame)
        frame = self._noise(frame)
        out = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        # Save the *processed* frame so motion blur is recursive
        self.state.last_frame = frame.copy()
        return out
