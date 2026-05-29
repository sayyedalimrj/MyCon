"""Blender 4.x API compatibility shims.

Blender's Python API drifts between minor versions (especially 3.x->4.x):
- ``bpy.ops.import_scene.obj`` was renamed to ``bpy.ops.wm.obj_import`` in 4.0.
- File Output nodes no longer accept ``color_mode = "BW"``; only
  ``"RGB"`` and ``"RGBA"`` are valid.
- The ``cycles.preferences.compute_device_type`` setter raises ``TypeError``
  on builds that don't ship the requested backend (e.g. OPTIX on a
  machine with only an Intel iGPU).
- Several Principled BSDF socket names changed: ``"Specular"`` ->
  ``"Specular IOR Level"``, ``"Transmission"`` ->
  ``"Transmission Weight"``, etc.

This module concentrates all of those workarounds in one place so the
renderer code stays readable and we have a single grep-friendly target
when a new Blender release breaks something.

Every function here is **pure-Python and side-effect free unless it
clearly mutates the bpy.* tree it was passed**, and every function
gracefully degrades when running on a host without ``bpy`` installed
(import-time fallbacks).

Usage from inside Blender::

    from synthetic_floor import blender_compat as compat

    backend = compat.activate_gpu(prefer="OPTIX")
    objs = compat.import_mesh(Path("stage_07.glb"))
    compat.set_file_output_color_mode(node, "DEPTH")  # auto-mapped
    compat.set_principled_socket(bsdf, "transmission", 0.85)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Iterable

# Lazy import: this module must be importable on hosts without Blender
# (e.g. for unit testing the helpers that don't actually touch bpy).
try:  # pragma: no cover - this branch only runs inside Blender
    import bpy  # type: ignore
except ImportError:
    bpy = None  # type: ignore[assignment]


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------


#: Color modes accepted by Blender 4.x ``CompositorNodeOutputFile`` and
#: ``ImageFormatSettings`` for the relevant file formats. ``"BW"`` is
#: only valid for some 2D image formats (PNG/JPEG); EXR and many others
#: refuse it. We standardise on ``"RGB"`` because every format
#: supports it and post-processing can always pick a single channel.
SUPPORTED_COLOR_MODES_PNG = {"BW", "RGB", "RGBA"}
SUPPORTED_COLOR_MODES_EXR = {"RGB", "RGBA"}  # NOT BW
SUPPORTED_COLOR_MODES_DEFAULT = {"RGB", "RGBA"}


#: GPU backend search order when the caller asks for ``OPTIX``.
GPU_BACKEND_FALLBACKS: dict[str, tuple[str, ...]] = {
    "OPTIX": ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI", "CPU"),
    "CUDA":  ("CUDA", "OPTIX", "CPU"),
    "HIP":   ("HIP", "CUDA", "CPU"),
    "METAL": ("METAL", "CPU"),
    "CPU":   ("CPU",),
}


# Mapping of "logical" socket names to the real Principled BSDF input
# names per Blender major version. Lookup falls through in order.
PRINCIPLED_SOCKET_ALIASES: dict[str, tuple[str, ...]] = {
    "base_color":      ("Base Color",),
    "roughness":       ("Roughness",),
    "metallic":        ("Metallic",),
    "ior":             ("IOR",),
    "transmission":    ("Transmission Weight", "Transmission"),
    "specular":        ("Specular IOR Level", "Specular"),
    "alpha":           ("Alpha",),
    "normal":          ("Normal",),
    "emission":        ("Emission Color", "Emission"),
    "emission_strength": ("Emission Strength",),
}


# ---------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------


def blender_version() -> tuple[int, int, int]:
    """Return ``bpy.app.version`` as a 3-tuple, or ``(0, 0, 0)`` outside Blender."""
    if bpy is None:  # pragma: no cover - host-side
        return (0, 0, 0)
    return tuple(bpy.app.version)  # type: ignore[return-value]


def is_blender_4_or_newer() -> bool:
    return blender_version()[0] >= 4


# ---------------------------------------------------------------------
# Color-mode helpers (the BW -> RGB bug)
# ---------------------------------------------------------------------


def safe_color_mode(requested: str, file_format: str) -> str:
    """Return a color mode that the given Blender file format accepts.

    Parameters
    ----------
    requested:
        The mode the caller actually wants (``"BW"``, ``"RGB"``, ``"RGBA"``).
    file_format:
        Blender's ``ImageFormatSettings.file_format`` value, e.g.
        ``"PNG"``, ``"OPEN_EXR"``, ``"JPEG"``, ``"TIFF"``.

    The function never raises -- it falls back to ``"RGB"`` which is
    accepted by every Blender 4.x file format.
    """
    requested = (requested or "RGB").upper()
    fmt = (file_format or "").upper()
    if fmt in {"OPEN_EXR", "OPEN_EXR_MULTILAYER"}:
        valid = SUPPORTED_COLOR_MODES_EXR
    elif fmt in {"PNG", "JPEG", "JPEG2000", "TIFF", "TARGA", "BMP"}:
        valid = SUPPORTED_COLOR_MODES_PNG
    else:
        valid = SUPPORTED_COLOR_MODES_DEFAULT
    if requested in valid:
        return requested
    return "RGB"  # universally safe fallback


def set_file_output_color_mode(node: Any, requested: str) -> str:
    """Safely set ``node.format.color_mode`` on a CompositorNodeOutputFile.

    Returns the color mode that was actually applied. If the requested
    mode is invalid for the node's current ``file_format``, the closest
    safe value is used and a warning is emitted.
    """
    fmt = getattr(node.format, "file_format", "")
    actual = safe_color_mode(requested, fmt)
    node.format.color_mode = actual
    if actual != requested.upper():
        print(
            f"[blender_compat] color_mode {requested!r} not valid for "
            f"{fmt!r}; using {actual!r} instead.",
            file=sys.stderr,
        )
    return actual


# ---------------------------------------------------------------------
# Principled BSDF socket aliases
# ---------------------------------------------------------------------


def get_principled_socket(bsdf: Any, logical_name: str):
    """Return the ``NodeSocket`` matching a logical name, or ``None``.

    ``logical_name`` is one of the keys of :data:`PRINCIPLED_SOCKET_ALIASES`.
    """
    aliases = PRINCIPLED_SOCKET_ALIASES.get(logical_name, (logical_name,))
    for n in aliases:
        if n in bsdf.inputs:
            return bsdf.inputs[n]
    return None


def set_principled_socket(bsdf: Any, logical_name: str, value: Any) -> bool:
    """Assign ``value`` to a Principled BSDF input by logical name.

    Returns ``True`` if the socket was found and assigned, ``False``
    otherwise. Never raises on a missing socket so the caller can apply
    a "best-effort" set of properties.
    """
    sock = get_principled_socket(bsdf, logical_name)
    if sock is None:
        return False
    sock.default_value = value
    return True


# ---------------------------------------------------------------------
# GPU activation
# ---------------------------------------------------------------------


def _try_set_compute_device_type(prefs: Any, backend: str) -> bool:
    """Attempt to switch ``compute_device_type``. Returns False on TypeError."""
    try:
        prefs.compute_device_type = backend
        return True
    except TypeError:
        return False


def activate_gpu(prefer: str = "OPTIX") -> str:
    """Enable GPU rendering in Cycles, returning the backend actually used.

    Falls back through :data:`GPU_BACKEND_FALLBACKS` until something
    sticks. Always sets ``scene.cycles.device`` consistently. Safe to
    call multiple times.
    """
    if bpy is None:  # pragma: no cover - host-side
        return "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    chain = GPU_BACKEND_FALLBACKS.get(prefer.upper(), ("CPU",))

    chosen: str | None = None
    for backend in chain:
        if not _try_set_compute_device_type(prefs, backend):
            continue
        prefs.get_devices()
        if backend == "CPU":
            chosen = "CPU"
            break
        usable = [d for d in prefs.devices if d.type == backend]
        if usable:
            chosen = backend
            break

    if chosen is None:
        prefs.compute_device_type = "CPU"
        prefs.get_devices()
        chosen = "CPU"

    # Enable everything of the chosen type, plus CPU as helper if a GPU is in use.
    enabled: list[str] = []
    for d in prefs.devices:
        if d.type == chosen or (chosen != "CPU" and d.type == "CPU"):
            d.use = True
            enabled.append(f"{d.name} ({d.type})")
        else:
            d.use = False
    bpy.context.scene.cycles.device = "GPU" if chosen != "CPU" else "CPU"
    print(f"[blender_compat] GPU backend = {chosen}; enabled {len(enabled)} device(s)")
    for name in enabled:
        print(f"[blender_compat]   - {name}")
    return chosen


# ---------------------------------------------------------------------
# Mesh import (4.x renames)
# ---------------------------------------------------------------------


def import_mesh(path: Path) -> list:
    """Import a mesh file into the current scene, returning the new objects.

    Supports ``.glb``, ``.gltf``, ``.obj``, ``.ply``. Uses the Blender
    4.x operator names with fall-back to 3.x where applicable.
    """
    if bpy is None:  # pragma: no cover - host-side
        raise RuntimeError("import_mesh() must be called inside Blender")
    path = Path(path)
    suffix = path.suffix.lower()
    before = set(bpy.data.objects)
    if suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path), merge_vertices=False)
    elif suffix == ".obj":
        op = getattr(bpy.ops.wm, "obj_import", None)
        if op is not None:
            op(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    elif suffix == ".ply":
        op = getattr(bpy.ops.wm, "ply_import", None)
        if op is not None:
            op(filepath=str(path))
        else:
            bpy.ops.import_mesh.ply(filepath=str(path))
    elif suffix in (".fbx",):
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        raise ValueError(f"Unsupported mesh format: {path}")
    new_objs = [o for o in bpy.data.objects if o not in before]
    return [o for o in new_objs if o.type == "MESH"]


# ---------------------------------------------------------------------
# Misc small shims
# ---------------------------------------------------------------------


def safe_set_denoiser(scene: Any, name: str = "OPENIMAGEDENOISE") -> str:
    """Enable Cycles denoising and try to set the requested denoiser.

    Some Blender builds (e.g. CPU-only, no OIDN bundled) refuse the
    requested value with ``TypeError``. We catch that and fall back to
    whatever default the scene already has.
    """
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = name
    except TypeError:
        # Build doesn't support this denoiser; keep the default.
        pass
    return scene.cycles.denoiser


def deg_to_rad(deg: float) -> float:
    return math.radians(deg)
