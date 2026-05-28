"""Ablation harness for pipeline-wide Cartesian-grid experiments.

This module is the thin, well-tested core of the ablation system. It does
*not* execute any stage. It only:

1. Reads a base YAML config and an *axes* spec (a flat dict of dotted-key →
   list-of-values).
2. Enumerates the Cartesian product of axis values into a list of cells.
3. For each cell, produces a deep-copied config dict with the axis values
   applied via :func:`apply_overlay`, and a stable cell name suitable for use
   as a directory.

The actual run script (`scripts/ablation_run.py`) is a small driver that
writes each overlay to disk and shells out to ``scripts/run_stage.py``. By
keeping the grid logic separate from the runner, the harness can be unit-
tested deterministically with no Open3D / COLMAP dependency.

The cell name is derived from a deterministic hash of the cell's overlay
contents so two runs of the same grid produce the same per-cell directories
even if axes are reordered. A short human-readable prefix is included so
operators can eyeball which cell is which.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import re
from dataclasses import dataclass, field
from typing import Any


_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class AblationCell:
    """One cell of the ablation grid."""

    name: str
    overlay: dict[str, Any]
    """Overlay applied to the base config: dotted-key → value."""

    @property
    def short_label(self) -> str:
        """Human-readable label like ``robust=tukey,corr=0.05,seed=42``."""
        parts = []
        for k in sorted(self.overlay):
            v = self.overlay[k]
            short_k = k.split(".")[-1]
            parts.append(f"{short_k}={v}")
        return ",".join(parts)


@dataclass(frozen=True)
class AblationGrid:
    """A Cartesian grid of ablation cells."""

    name: str
    axes: dict[str, list[Any]]
    cells: list[AblationCell] = field(default_factory=list)


def _normalize_axes(axes: dict[str, Any]) -> dict[str, list[Any]]:
    if not isinstance(axes, dict):
        raise TypeError("axes must be a dict of dotted_key -> list_of_values")
    out: dict[str, list[Any]] = {}
    for key, values in axes.items():
        if not isinstance(key, str) or "." not in key:
            raise ValueError(f"axis key must be a dotted YAML path, got {key!r}")
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"axis values must be a list, got {type(values).__name__} for {key!r}")
        if len(values) == 0:
            raise ValueError(f"axis {key!r} has no values")
        out[key] = list(values)
    return out


def _cell_name(name: str, overlay: dict[str, Any]) -> str:
    """Stable, filesystem-safe cell name.

    Composed of a short label (truncated) plus a 10-char hex hash so two cells
    that happen to truncate to the same prefix don't collide.
    """
    label_parts = []
    for k in sorted(overlay):
        v = overlay[k]
        label_parts.append(f"{k.split('.')[-1]}-{v}")
    label = "_".join(label_parts)
    label_safe = _NAME_SAFE_RE.sub("-", label)[:48]

    blob = name + "|" + "|".join(f"{k}={overlay[k]!r}" for k in sorted(overlay))
    digest = hashlib.blake2b(blob.encode("utf-8"), digest_size=5).hexdigest()
    return f"{label_safe}_{digest}" if label_safe else digest


def build_grid(name: str, axes: dict[str, Any]) -> AblationGrid:
    """Enumerate the Cartesian product of axes into an :class:`AblationGrid`."""
    norm = _normalize_axes(axes)
    keys = list(norm.keys())
    cells: list[AblationCell] = []
    for combo in itertools.product(*(norm[k] for k in keys)):
        overlay = {k: combo[i] for i, k in enumerate(keys)}
        cell_name = _cell_name(name, overlay)
        cells.append(AblationCell(name=cell_name, overlay=overlay))
    return AblationGrid(name=name, axes=norm, cells=cells)


def apply_overlay(base_cfg: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied config with ``overlay`` (dotted-keys) applied.

    The function never mutates ``base_cfg``. Intermediate dicts are created
    when the path passes through a *missing* key. If the path passes through
    a node that *exists but is not a dict*, a :class:`ValueError` is raised
    so silent corruption of an existing scalar is impossible.
    """
    cfg = copy.deepcopy(base_cfg)
    for dotted, value in overlay.items():
        parts = dotted.split(".")
        cur: Any = cfg
        for part in parts[:-1]:
            if not isinstance(cur, dict):
                raise ValueError(f"cannot apply overlay {dotted!r}: traversal hit a non-dict at {part!r}")
            if part not in cur:
                # Missing intermediate: create it.
                cur[part] = {}
            elif not isinstance(cur[part], dict):
                # Existing non-dict scalar: refuse to silently overwrite.
                raise ValueError(
                    f"cannot apply overlay {dotted!r}: traversal would overwrite "
                    f"existing non-dict value at {part!r}"
                )
            cur = cur[part]
        if not isinstance(cur, dict):
            raise ValueError(f"cannot apply overlay {dotted!r}: terminal parent is not a dict")
        cur[parts[-1]] = value
    return cfg


def grid_summary(grid: AblationGrid) -> dict[str, Any]:
    """Compact summary of a grid suitable for an ablation_summary.json header."""
    return {
        "name": grid.name,
        "axes": grid.axes,
        "cell_count": len(grid.cells),
        "cell_names": [c.name for c in grid.cells],
    }


__all__ = [
    "AblationCell",
    "AblationGrid",
    "apply_overlay",
    "build_grid",
    "grid_summary",
]
