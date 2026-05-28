"""Tests for ``pipeline.common.determinism``.

These tests pin the contract that the determinism module guarantees:
- :func:`derived_seed` is deterministic across calls.
- Different ``label`` / ``parts`` produce different seeds with overwhelming
  probability (we test 256 distinct labels for collision-freeness).
- Changing ``base_seed`` shifts every derived seed (so ``project.random_seed``
  in the YAML actually controls the run).
- :func:`project_seed` reads from both dict and attribute-style configs.
- :func:`set_global_determinism_envs` does not stomp an already-set
  ``PYTHONHASHSEED``.
"""

from __future__ import annotations

import os

import numpy as np

from pipeline.common.determinism import (
    DEFAULT_PROJECT_SEED,
    derived_seed,
    project_seed,
    seeded_rng,
    set_global_determinism_envs,
)


def test_derived_seed_is_deterministic() -> None:
    a = derived_seed("stage_09_uncertainty", "elem_001")
    b = derived_seed("stage_09_uncertainty", "elem_001")
    assert a == b


def test_derived_seed_changes_with_label_or_parts() -> None:
    base = derived_seed("stage_09_uncertainty", "elem_001")
    assert derived_seed("stage_09_other", "elem_001") != base
    assert derived_seed("stage_09_uncertainty", "elem_002") != base
    assert derived_seed("stage_09_uncertainty", "elem_001", extra="x") != base


def test_derived_seed_collision_free_for_256_labels() -> None:
    seeds = {derived_seed("stage_09_uncertainty", f"elem_{i:03d}") for i in range(256)}
    # Birthday paradox: 256 random 32-bit values have ≈0.0007% expected
    # collision rate. A single collision in 256 is a real bug.
    assert len(seeds) == 256


def test_derived_seed_shifts_with_base_seed() -> None:
    a = derived_seed("foo", "bar", base_seed=42)
    b = derived_seed("foo", "bar", base_seed=43)
    assert a != b


def test_seeded_rng_is_reproducible() -> None:
    rng_a = seeded_rng("stage_09_uncertainty", "elem_001", base_seed=42)
    rng_b = seeded_rng("stage_09_uncertainty", "elem_001", base_seed=42)
    assert np.array_equal(rng_a.standard_normal(64), rng_b.standard_normal(64))


def test_project_seed_reads_dict_config() -> None:
    cfg = {"project": {"random_seed": 1729}}
    assert project_seed(cfg) == 1729


def test_project_seed_reads_attribute_config() -> None:
    class _ProjectStub:
        random_seed = 7

    class _CfgStub:
        project = _ProjectStub()

    assert project_seed(_CfgStub()) == 7


def test_project_seed_falls_back_on_invalid() -> None:
    cfg = {"project": {"random_seed": "garbage"}}
    assert project_seed(cfg) == DEFAULT_PROJECT_SEED


def test_set_global_determinism_envs_preserves_existing_pythonhashseed() -> None:
    # Save and restore the env to keep the test session independent.
    original = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = "12345"
    try:
        set_global_determinism_envs(99)
        # The function uses setdefault, so 12345 must survive.
        assert os.environ["PYTHONHASHSEED"] == "12345"
    finally:
        if original is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = original
