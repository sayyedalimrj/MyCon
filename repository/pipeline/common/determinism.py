"""Centralized determinism utilities.

This module is the single point through which every stage should derive its
random state. It exists to fix the historical situation where multiple stages
hardcoded literal seeds (`42`, `9`, `75`) that ignored ``project.random_seed``
in the YAML config, making the pipeline only partially reproducible.

Usage
-----

>>> from pipeline.common.determinism import derived_seed, seeded_rng
>>> seed = derived_seed("stage_09_uncertainty", element_global_id="3MaIYBfdH8")
>>> rng = seeded_rng("stage_09_uncertainty", element_global_id="3MaIYBfdH8")

The ``derived_seed`` function takes a ``base_seed`` (which defaults to the
project seed read by :func:`project_seed`) and a series of label parts. Parts
are stringified and hashed to produce a deterministic 32-bit offset. The
returned seed is therefore stable across runs *and* distinct between callers.

The module deliberately avoids import-time side effects so importing it in a
test does not mutate global RNGs.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np

DEFAULT_PROJECT_SEED: int = 42
"""Project-wide default seed used when no config seed is supplied."""

_UINT32_MASK: int = 0xFFFFFFFF


def project_seed(cfg: Any | None = None, default: int = DEFAULT_PROJECT_SEED) -> int:
    """Return the project random seed read from a config dict-or-object.

    Accepts the same shape that :mod:`pipeline.common.config` produces. Falls
    back to ``default`` (32-bit) if the key is missing or invalid.
    """
    if cfg is None:
        return int(default)
    raw: Any
    if isinstance(cfg, dict):
        proj = cfg.get("project") or {}
        raw = proj.get("random_seed", default) if isinstance(proj, dict) else default
    else:
        proj = getattr(cfg, "project", None)
        raw = getattr(proj, "random_seed", default) if proj is not None else default
    try:
        return int(raw) & _UINT32_MASK
    except (TypeError, ValueError):
        return int(default)


def _stringify_part(part: Any) -> str:
    if isinstance(part, (str, int, float, bool)):
        return str(part)
    if isinstance(part, bytes):
        try:
            return part.decode("utf-8", errors="replace")
        except Exception:
            return repr(part)
    return repr(part)


def derived_seed(label: str, *parts: Any, base_seed: int | None = None, **kw_parts: Any) -> int:
    """Return a deterministic 32-bit seed derived from ``base_seed`` + label/parts.

    The hashing is BLAKE2b-128 truncated to 32 bits and XORed with the base
    seed, so changing ``base_seed`` shifts every derived seed in lockstep, while
    different ``label`` / ``parts`` produce uncorrelated streams.

    The function intentionally does **not** read any global state. Callers that
    want the project seed must pass it via ``base_seed`` (or rely on the
    :class:`DEFAULT_PROJECT_SEED`).
    """
    base = int(DEFAULT_PROJECT_SEED if base_seed is None else base_seed) & _UINT32_MASK
    h = hashlib.blake2b(digest_size=16)
    h.update(str(label).encode("utf-8", errors="replace"))
    for part in parts:
        h.update(b"\x1f")
        h.update(_stringify_part(part).encode("utf-8", errors="replace"))
    for key in sorted(kw_parts):
        h.update(b"\x1e")
        h.update(str(key).encode("utf-8", errors="replace"))
        h.update(b"=")
        h.update(_stringify_part(kw_parts[key]).encode("utf-8", errors="replace"))
    digest_int = int.from_bytes(h.digest()[:4], "big") & _UINT32_MASK
    return (base ^ digest_int) & _UINT32_MASK


def seeded_rng(label: str, *parts: Any, base_seed: int | None = None, **kw_parts: Any) -> np.random.Generator:
    """Return a numpy default_rng seeded from :func:`derived_seed`."""
    return np.random.default_rng(derived_seed(label, *parts, base_seed=base_seed, **kw_parts))


def set_global_determinism_envs(seed: int, *, deterministic_threads: bool = False) -> None:
    """Best-effort global determinism setup.

    Sets ``PYTHONHASHSEED`` (if not already set), seeds the stdlib :mod:`random`
    module, seeds numpy's legacy ``np.random`` generator, and optionally caps
    thread counts for deterministic BLAS / OpenMP.

    Returning early when ``PYTHONHASHSEED`` is already exported preserves the
    user's explicit choice. Thread caps are *opt-in* because they have a real
    performance cost.
    """
    seed_int = int(seed) & _UINT32_MASK
    os.environ.setdefault("PYTHONHASHSEED", str(seed_int))
    random.seed(seed_int)
    try:
        np.random.seed(seed_int)
    except Exception:
        pass
    if deterministic_threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, "1")


__all__ = [
    "DEFAULT_PROJECT_SEED",
    "project_seed",
    "derived_seed",
    "seeded_rng",
    "set_global_determinism_envs",
]
