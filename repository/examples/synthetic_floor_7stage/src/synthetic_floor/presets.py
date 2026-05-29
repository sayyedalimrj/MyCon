"""Quality presets for the CPU and GPU pipelines.

A *preset* is a small dictionary of overrides applied on top of the
``scene.yaml`` config and CLI defaults. We define three:

``debug``
    Tiny resolution + few frames + low sample count. Designed for
    smoke tests and CI -- one full 7-stage GPU run finishes in ~1 min
    on a T4. Visually noisy; do NOT use for paper figures.

``balanced``
    Sensible defaults for development and review on Colab T4.
    Roughly 5-15 minutes for a full 7-stage GPU run.

``hq``
    High-quality output for final figures / dataset releases. Larger
    resolution, more samples, longer clips. ~30+ minutes on T4,
    ~10-15 min on A100.

Both ``run_generate.py`` (CPU) and ``run_blender_gpu.py`` (GPU)
expose ``--preset {debug,balanced,hq}`` which calls
:func:`apply_cpu_preset` / :func:`apply_gpu_preset` respectively.

Explicit CLI flags **always win** over preset values, so a user can
say e.g. ``--preset hq --frames 30`` to keep everything HQ except the
frame count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PresetName = Literal["debug", "balanced", "hq"]
PRESET_NAMES: tuple[PresetName, ...] = ("debug", "balanced", "hq")
DEFAULT_PRESET: PresetName = "balanced"


# ---------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CPUPreset:
    """Overrides for the NumPy CPU rasterizer pipeline."""

    width_px: int
    height_px: int
    duration_per_stage_sec: float
    motion_blur_samples: int  # used by smartphone_sim


@dataclass(frozen=True)
class GPUPreset:
    """Overrides for the Blender + Cycles GPU pipeline."""

    width_px: int
    height_px: int
    samples: int  # Cycles samples per pixel
    frames_per_stage: int  # total animation frames; fps separate
    fps: int
    motion_blur: bool


CPU_PRESETS: dict[str, CPUPreset] = {
    "debug":    CPUPreset(width_px=320, height_px=180, duration_per_stage_sec=1.0, motion_blur_samples=2),
    "balanced": CPUPreset(width_px=640, height_px=360, duration_per_stage_sec=4.0, motion_blur_samples=4),
    "hq":       CPUPreset(width_px=1280, height_px=720, duration_per_stage_sec=6.0, motion_blur_samples=6),
}

GPU_PRESETS: dict[str, GPUPreset] = {
    "debug":    GPUPreset(width_px=480,  height_px=270, samples=32,  frames_per_stage=30,  fps=30, motion_blur=False),
    "balanced": GPUPreset(width_px=960,  height_px=540, samples=96,  frames_per_stage=120, fps=30, motion_blur=True),
    "hq":       GPUPreset(width_px=1280, height_px=720, samples=192, frames_per_stage=180, fps=30, motion_blur=True),
}


# ---------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------


def get_cpu_preset(name: str) -> CPUPreset:
    if name not in CPU_PRESETS:
        raise ValueError(f"Unknown CPU preset {name!r}. Valid options: {list(CPU_PRESETS)}")
    return CPU_PRESETS[name]


def get_gpu_preset(name: str) -> GPUPreset:
    if name not in GPU_PRESETS:
        raise ValueError(f"Unknown GPU preset {name!r}. Valid options: {list(GPU_PRESETS)}")
    return GPU_PRESETS[name]


def apply_cpu_preset(spec, preset_name: str, *,
                     width: int | None = None,
                     height: int | None = None,
                     duration: float | None = None):
    """Return a new ``SceneSpec`` with the preset (and CLI overrides) applied.

    The original ``spec`` is unchanged because ``SceneSpec`` is frozen.
    Explicit CLI overrides (``width`` / ``height`` / ``duration``) win
    over preset values when not ``None``.
    """
    from dataclasses import replace
    p = get_cpu_preset(preset_name)
    cam_kwargs = {
        "width_px": width if width is not None else p.width_px,
        "height_px": height if height is not None else p.height_px,
        "duration_per_stage_sec": duration if duration is not None else p.duration_per_stage_sec,
    }
    return replace(spec, camera=replace(spec.camera, **cam_kwargs))


def apply_gpu_preset(preset_name: str, *,
                     resolution: tuple[int, int] | None = None,
                     samples: int | None = None,
                     frames: int | None = None,
                     fps: int | None = None,
                     motion_blur: bool | None = None) -> GPUPreset:
    """Resolve preset + CLI overrides into a single :class:`GPUPreset`.

    Returns a new :class:`GPUPreset` instance; the original presets are
    not mutated. Explicit CLI overrides win over preset values.
    """
    p = get_gpu_preset(preset_name)
    return GPUPreset(
        width_px=resolution[0] if resolution else p.width_px,
        height_px=resolution[1] if resolution else p.height_px,
        samples=samples if samples is not None else p.samples,
        frames_per_stage=frames if frames is not None else p.frames_per_stage,
        fps=fps if fps is not None else p.fps,
        motion_blur=motion_blur if motion_blur is not None else p.motion_blur,
    )


def describe_preset(preset_name: str, *, gpu: bool = False) -> str:
    """Human-readable single-line description of a preset (for logs)."""
    if gpu:
        p = get_gpu_preset(preset_name)
        return (f"{preset_name}: {p.width_px}x{p.height_px} "
                f"@ {p.fps}fps, {p.frames_per_stage} frames, "
                f"{p.samples} samples, motion_blur={p.motion_blur}")
    p = get_cpu_preset(preset_name)
    return (f"{preset_name}: {p.width_px}x{p.height_px} "
            f"for {p.duration_per_stage_sec}s/stage, "
            f"motion_blur_samples={p.motion_blur_samples}")
