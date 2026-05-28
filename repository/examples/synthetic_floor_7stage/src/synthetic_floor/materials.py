"""Procedural PBR-style material library.

Every material exposes a base-colour image (sRGB), a roughness scalar
(0..1) and a "tint" colour used for shading. Images are produced
deterministically with NumPy + Pillow so the example has zero runtime
asset dependencies, but the texture sizes and feature scales mimic the
real-world thing (e.g. a single brick is ~25 cm wide).

The materials cover everything the seven stages need:

* ``raw_concrete``      - bare structural concrete (slab, columns, walls).
* ``rough_plaster``     - first plaster coat, irregular and dusty.
* ``fine_plaster``      - smooth plaster ready for paint.
* ``painted``           - clean painted wall (warm white).
* ``raw_wood``          - unfinished wood (door panels in mid-stages).
* ``painted_wood``      - finished door panel (light grey).
* ``tile``              - finished floor tile pattern.
* ``glass``             - window glass (semi-transparent blue tint).
* ``brick``             - exposed brick (kept for future use).
* ``ceiling``           - simple ceiling tile pattern.
* ``construction_dust`` - thin overlay used by mid-stages.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class Material:
    name: str
    tint: tuple[float, float, float]   # multiplied with sampled texel
    roughness: float                    # 0=mirror, 1=fully rough
    metallic: float                     # 0=dielectric, 1=metal
    transparency: float                 # 0=opaque, 1=fully transparent
    # Real-world tile size in meters (texture wraps across this size)
    tile_meters: tuple[float, float]
    image: Image.Image                  # base-colour map (RGB, uint8)


# ---------------------------------------------------------------------
# Procedural texture helpers
# ---------------------------------------------------------------------


def _seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.uint64(seed))


def _value_noise(shape: tuple[int, int], scale: int, rng: np.random.Generator) -> np.ndarray:
    """Cheap value noise: random low-res grid + bilinear upsample."""
    h, w = shape
    lh, lw = max(2, h // scale), max(2, w // scale)
    base = rng.random((lh, lw)).astype(np.float32)
    img = Image.fromarray((base * 255).astype(np.uint8), mode="L")
    img = img.resize((w, h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius=scale * 0.4))
    return np.asarray(img, dtype=np.float32) / 255.0


def _fbm(shape: tuple[int, int], rng: np.random.Generator,
         octaves: int = 5, persistence: float = 0.55, base_scale: int = 64) -> np.ndarray:
    """Fractal Brownian motion using value noise octaves."""
    out = np.zeros(shape, dtype=np.float32)
    amp = 1.0
    norm = 0.0
    for o in range(octaves):
        s = max(2, base_scale // (2 ** o))
        out += amp * _value_noise(shape, s, rng)
        norm += amp
        amp *= persistence
    out /= max(norm, 1e-6)
    return out


# ---------------------------------------------------------------------
# Per-material generators
# ---------------------------------------------------------------------


def _build_concrete(size: int, seed: int) -> Image.Image:
    rng = _seeded_rng(seed)
    base = 0.62 + 0.18 * _fbm((size, size), rng, octaves=6, base_scale=128)
    # Add cool tone variations
    cool = 0.04 * (_fbm((size, size), rng, octaves=3, base_scale=256) - 0.5)
    r = np.clip(base + cool * 0.6, 0.10, 0.95)
    g = np.clip(base + cool * 0.7, 0.10, 0.95)
    b = np.clip(base + cool * 1.2, 0.10, 0.95)
    # Add small dark specks (aggregate)
    specks = (rng.random((size, size)) > 0.997).astype(np.float32)
    r -= specks * 0.4
    g -= specks * 0.4
    b -= specks * 0.4
    arr = np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB").filter(ImageFilter.GaussianBlur(radius=0.3))
    return img


def _build_rough_plaster(size: int, seed: int) -> Image.Image:
    rng = _seeded_rng(seed)
    base = 0.78 + 0.10 * _fbm((size, size), rng, octaves=5, base_scale=96)
    base += 0.02 * (rng.random((size, size)) - 0.5)
    r = np.clip(base + 0.02, 0.20, 0.98)
    g = np.clip(base + 0.005, 0.20, 0.98)
    b = np.clip(base - 0.02, 0.20, 0.98)
    # Trowel streaks
    streaks = _fbm((size, size), rng, octaves=2, base_scale=8) * 0.06
    r -= streaks
    g -= streaks
    b -= streaks
    arr = np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")


def _build_fine_plaster(size: int, seed: int) -> Image.Image:
    rng = _seeded_rng(seed)
    base = 0.86 + 0.04 * _fbm((size, size), rng, octaves=3, base_scale=256)
    arr = np.stack([base, base * 0.995, base * 0.985], axis=-1)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")


def _build_painted(size: int, seed: int) -> Image.Image:
    rng = _seeded_rng(seed)
    # Warm white with very subtle texture so it doesn't look flat CG.
    base = 0.93 + 0.012 * _fbm((size, size), rng, octaves=2, base_scale=384)
    r = base
    g = base * 0.985
    b = base * 0.965
    arr = np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")


def _build_raw_wood(size: int, seed: int) -> Image.Image:
    rng = _seeded_rng(seed)
    # Streaky vertical grain
    cols = 0.55 + 0.18 * _fbm((size, 1), rng, octaves=4, base_scale=4)  # wide rings
    cols = np.repeat(cols, size, axis=1)
    rings = _fbm((size, size), rng, octaves=3, base_scale=16)
    base = cols * 0.7 + rings * 0.3
    base = 0.45 + 0.40 * base
    r = np.clip(base * 1.05, 0.08, 0.95)
    g = np.clip(base * 0.78, 0.05, 0.85)
    b = np.clip(base * 0.55, 0.04, 0.75)
    arr = np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")


def _build_painted_wood(size: int, seed: int) -> Image.Image:
    rng = _seeded_rng(seed)
    base = 0.78 + 0.02 * _fbm((size, size), rng, octaves=2, base_scale=256)
    arr = np.stack([base * 0.99, base * 1.00, base * 1.02], axis=-1)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB")


def _build_tile(size: int, seed: int) -> Image.Image:
    rng = _seeded_rng(seed)
    base = _fbm((size, size), rng, octaves=4, base_scale=64) * 0.20 + 0.72
    r = base * 0.97
    g = base * 0.99
    b = base * 1.02
    arr = np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
    # Draw a 4x4 tile grid with thin grout lines
    grout_period = size // 4
    grout_thickness = max(2, size // 256)
    arr = np.asarray(img).copy()
    for k in range(0, size + 1, grout_period):
        a, b_ = max(0, k - grout_thickness // 2), min(size, k + grout_thickness // 2)
        arr[a:b_, :, :] = (arr[a:b_, :, :] * 0.62).astype(np.uint8)
        arr[:, a:b_, :] = (arr[:, a:b_, :] * 0.62).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _build_glass(size: int, seed: int) -> Image.Image:
    rng = _seeded_rng(seed)
    n = _fbm((size, size), rng, octaves=2, base_scale=512) * 0.05 + 0.85
    r = n * 0.85
    g = n * 0.95
    b = n * 1.00
    arr = np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")


def _build_brick(size: int, seed: int) -> Image.Image:
    rng = _seeded_rng(seed)
    rows = 12
    cols = 6
    bw = size / cols
    bh = size / rows
    arr = np.full((size, size, 3), 0.55, dtype=np.float32)
    # Per-brick tints
    tints = rng.uniform(0.7, 1.0, size=(rows, cols, 3))
    for ry in range(rows):
        offset = (ry % 2) * (bw / 2)
        for cx in range(cols + 1):
            x_center = cx * bw + offset
            xs = int(x_center - bw / 2)
            xe = int(x_center + bw / 2)
            ys = int(ry * bh)
            ye = int((ry + 1) * bh)
            xs2, xe2 = max(0, xs), min(size, xe)
            ys2, ye2 = max(0, ys), min(size, ye)
            tint = tints[ry % rows, cx % cols]
            base = tint * np.array([0.62, 0.32, 0.25])
            arr[ys2:ye2, xs2:xe2, :] = base
    # Add mortar (darker borders)
    mortar = 0.18
    for ry in range(rows + 1):
        y = int(ry * bh)
        arr[max(0, y - 1):min(size, y + 1), :, :] = mortar
    arr = arr + 0.05 * (_fbm((size, size), rng, octaves=4, base_scale=64)[..., None] - 0.5)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")


def _build_ceiling(size: int, seed: int) -> Image.Image:
    rng = _seeded_rng(seed)
    base = 0.92 + 0.02 * _fbm((size, size), rng, octaves=2, base_scale=256)
    arr = np.stack([base, base, base * 0.99], axis=-1)
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="RGB")
    arr = np.asarray(img).copy()
    # 2x2 tile grid
    grid = size // 2
    for k in (0, grid, size):
        a = max(0, k - 1)
        b = min(size, k + 1)
        arr[a:b, :, :] = (arr[a:b, :, :] * 0.92).astype(np.uint8)
        arr[:, a:b, :] = (arr[:, a:b, :] * 0.92).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _build_construction_dust(size: int, seed: int) -> Image.Image:
    rng = _seeded_rng(seed)
    n = _fbm((size, size), rng, octaves=4, base_scale=128)
    a = np.clip(n * 0.30, 0.0, 1.0)
    arr = np.stack([np.full_like(a, 0.95), np.full_like(a, 0.93), np.full_like(a, 0.88), a], axis=-1)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGBA")


# ---------------------------------------------------------------------
# Library assembly
# ---------------------------------------------------------------------


_DEF: tuple[tuple[str, callable, dict], ...] = (
    ("raw_concrete",       _build_concrete,        {"tint": (0.95, 0.95, 0.95), "roughness": 0.92, "metallic": 0.0, "transparency": 0.0, "tile_meters": (1.5, 1.5)}),
    ("rough_plaster",      _build_rough_plaster,   {"tint": (1.00, 0.99, 0.97), "roughness": 0.88, "metallic": 0.0, "transparency": 0.0, "tile_meters": (1.2, 1.2)}),
    ("fine_plaster",       _build_fine_plaster,    {"tint": (1.00, 1.00, 0.99), "roughness": 0.78, "metallic": 0.0, "transparency": 0.0, "tile_meters": (1.0, 1.0)}),
    ("painted",            _build_painted,         {"tint": (1.00, 0.99, 0.97), "roughness": 0.55, "metallic": 0.0, "transparency": 0.0, "tile_meters": (1.0, 1.0)}),
    ("raw_wood",           _build_raw_wood,        {"tint": (1.00, 0.93, 0.80), "roughness": 0.70, "metallic": 0.0, "transparency": 0.0, "tile_meters": (1.0, 2.1)}),
    ("painted_wood",       _build_painted_wood,    {"tint": (0.92, 0.94, 0.95), "roughness": 0.45, "metallic": 0.0, "transparency": 0.0, "tile_meters": (1.0, 2.1)}),
    ("tile",               _build_tile,            {"tint": (1.00, 1.00, 1.00), "roughness": 0.30, "metallic": 0.0, "transparency": 0.0, "tile_meters": (0.6, 0.6)}),
    ("glass",              _build_glass,           {"tint": (0.80, 0.90, 1.00), "roughness": 0.12, "metallic": 0.0, "transparency": 0.55, "tile_meters": (1.4, 1.2)}),
    ("brick",              _build_brick,           {"tint": (1.00, 0.85, 0.80), "roughness": 0.92, "metallic": 0.0, "transparency": 0.0, "tile_meters": (1.4, 1.4)}),
    ("ceiling",            _build_ceiling,         {"tint": (0.99, 0.99, 0.97), "roughness": 0.85, "metallic": 0.0, "transparency": 0.0, "tile_meters": (1.2, 1.2)}),
    ("construction_dust",  _build_construction_dust,{"tint": (1.00, 0.97, 0.92), "roughness": 0.95, "metallic": 0.0, "transparency": 0.5, "tile_meters": (1.5, 1.5)}),
)


def build_material_library(*, seed: int, size: int = 512) -> dict[str, Material]:
    """Construct every material as a deterministic procedural texture.

    The seed is mixed with the material name to keep textures stable
    across runs but distinct from each other.
    """
    out: dict[str, Material] = {}
    for i, (name, fn, props) in enumerate(_DEF):
        sub_seed = seed * 1000 + i + 1
        img = fn(size, sub_seed)
        out[name] = Material(
            name=name,
            tint=tuple(float(x) for x in props["tint"]),
            roughness=float(props["roughness"]),
            metallic=float(props["metallic"]),
            transparency=float(props["transparency"]),
            tile_meters=tuple(float(x) for x in props["tile_meters"]),
            image=img,
        )
    return out


def save_material_samples(library: Mapping[str, Material], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, mat in library.items():
        p = out_dir / f"material_{name}.png"
        mat.image.save(p, format="PNG")
        written.append(p)
    return written


# Mapping of element category -> material when no explicit finishing
# string is given. Used as a fallback by the renderer.
DEFAULT_CATEGORY_MATERIAL: dict[str, str] = {
    "slab": "raw_concrete",
    "columns": "raw_concrete",
    "exterior_walls": "raw_concrete",
    "interior_walls": "raw_concrete",
    "windows": "glass",
    "doors": "raw_wood",
    "ceiling": "ceiling",
    "floor_finish": "tile",
    "baseboards": "painted_wood",
    "fixtures": "painted",
}


def material_for(category: str, finishing: str, library: Mapping[str, Material]) -> Material:
    """Pick the appropriate material for a category given its finishing.

    Priority:
        1. Specific category overrides (e.g. windows always use glass
           when finished, regardless of the configured finishing).
        2. Direct match on the finishing key.
        3. Fallback to the category default.
    """
    # Specific overrides that make physical sense
    if category == "windows":
        # Glass only appears once the window is "completed" (finishing != raw_concrete)
        if finishing in ("painted", "fine_plaster"):
            return library["glass"]
        if finishing == "rough_plaster":
            return library["rough_plaster"]
        return library["raw_concrete"]
    if category == "doors":
        if finishing in ("painted_wood", "painted"):
            return library["painted_wood"]
        if finishing in ("raw_wood",):
            return library["raw_wood"]
        return library["raw_concrete"]
    if category == "ceiling" and finishing in ("painted", "fine_plaster"):
        return library["ceiling"]
    if category == "floor_finish" and finishing == "tile":
        return library["tile"]
    if category == "baseboards":
        if finishing in ("painted", "painted_wood"):
            return library["painted_wood"]
        return library["raw_wood"]
    # Generic case
    if finishing in library:
        return library[finishing]
    return library[DEFAULT_CATEGORY_MATERIAL.get(category, "raw_concrete")]
