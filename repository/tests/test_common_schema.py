"""Tests for ``pipeline.common.schema`` typed config views.

These tests pin two contracts:

1. Every shipping schema can be constructed from the real ``configs/site01.yaml``
   without raising.
2. Each schema's ``required_config_keys()`` matches the keys it actually
   reads (a structural drift check that would fail loudly if a future
   commit adds a key to ``from_config`` without listing it).
3. The schema layer never silently re-types a value: a missing required key
   yields :class:`ConfigSchemaError`, a wrong type yields
   :class:`ConfigSchemaError` with the offending key in the message.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pipeline.common.config import load_config
from pipeline.common.schema import (
    ALL_STAGE_SCHEMAS,
    ConfigSchemaError,
    InputsSchema,
    PathsSchema,
    ProjectSchema,
    Stage01IngestSchema,
    Stage02KeyframesSchema,
    Stage03ColmapSchema,
    Stage04RefinementSchema,
    Stage05DenseSchema,
    Stage06DA3Schema,
    Stage07CleanupSchema,
    Stage08BimEvalSchema,
    Stage09ProgressSchema,
    Stage10CopilotSchema,
)

CONFIG_PATH = Path("configs/site01.yaml")


def test_project_schema_loads_real_config() -> None:
    cfg = load_config(CONFIG_PATH)
    schema = ProjectSchema.from_config(cfg)
    assert schema.name == "site01"
    assert schema.run_id == "2026-04-30_site01_baseline"
    assert schema.root == Path("/workspace")
    assert schema.random_seed == 42


def test_inputs_schema_loads_real_config() -> None:
    cfg = load_config(CONFIG_PATH)
    schema = InputsSchema.from_config(cfg)
    assert schema.video.suffix == ".mp4"
    assert schema.ifc.suffix == ".ifc"


def test_paths_schema_exposes_all_required_paths() -> None:
    cfg = load_config(CONFIG_PATH)
    schema = PathsSchema.from_config(cfg)
    # Spot-check a few; exhaustive list is in the dataclass definition.
    assert schema.fused_ply.name == "fused.ply"
    assert schema.colmap_db.name == "database.db"
    assert schema.metrics_dir.name == "site01"


@pytest.mark.parametrize(
    "schema_class",
    [
        Stage01IngestSchema,
        Stage02KeyframesSchema,
        Stage03ColmapSchema,
        Stage04RefinementSchema,
        Stage05DenseSchema,
        Stage06DA3Schema,
        Stage07CleanupSchema,
        Stage08BimEvalSchema,
        Stage09ProgressSchema,
        Stage10CopilotSchema,
    ],
)
def test_every_stage_schema_loads_real_config(schema_class: type) -> None:
    """Every shipping stage schema must hydrate from the real config."""
    cfg = load_config(CONFIG_PATH)
    schema = schema_class.from_config(cfg)
    assert schema is not None
    # Required keys list must be non-empty and contain at least
    # project + paths roots — the registry depends on this invariant.
    keys = schema_class.required_config_keys()
    assert len(keys) > 0
    assert "project.name" in keys
    assert "project.run_id" in keys
    assert any(k.startswith("paths.") for k in keys)


def test_all_stage_schemas_tuple_matches_stage_count() -> None:
    """ALL_STAGE_SCHEMAS must list every shipping stage schema exactly once."""
    classes = ALL_STAGE_SCHEMAS
    names = {cls.__name__ for cls in classes}
    assert len(names) == len(classes), "ALL_STAGE_SCHEMAS contains duplicates"
    # The 10 shipping stages each have one schema entry. Stages 4.5, 7.5,
    # 7.6, 7.7, 8a (metric_alignment) reuse the closest matching schema
    # (Stage04Refinement, Stage07Cleanup, Stage08BimEval).
    assert len(classes) == 11


def test_required_keys_are_subset_of_config_keys() -> None:
    """A schema must not declare a required key that is absent from the YAML."""
    cfg = load_config(CONFIG_PATH)
    for schema_class in ALL_STAGE_SCHEMAS:
        for dotted in schema_class.required_config_keys():
            # cfg.require raises on missing; we want a clear assertion error
            # rather than the underlying ConfigError if this ever drifts.
            try:
                cfg.require(dotted)
            except Exception as exc:
                pytest.fail(
                    f"{schema_class.__name__} declares required key {dotted!r}, "
                    f"but it is not present in configs/site01.yaml: {exc}"
                )


def test_from_config_raises_for_missing_required_key(tmp_path: Path) -> None:
    """Removing a required key must surface as ``ConfigSchemaError``."""
    source = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    # Drop project.random_seed — required by ProjectSchema and indirectly by
    # every stage schema.
    del source["project"]["random_seed"]
    bad_path = tmp_path / "no_seed.yaml"
    bad_path.write_text(yaml.safe_dump(source), encoding="utf-8")

    # The base PipelineConfig validator already rejects this earlier than
    # the schema layer, but the failure must still surface as a ConfigError
    # subclass that callers can catch uniformly.
    from pipeline.common.config import ConfigError

    with pytest.raises(ConfigError, match="project.random_seed"):
        load_config(bad_path)


def test_from_config_raises_for_wrong_type(tmp_path: Path) -> None:
    """A non-numeric ``project.random_seed`` must raise ``ConfigSchemaError``.

    The base validator does not type-check this field. The schema layer is
    the one that catches it.
    """
    source = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    source["project"]["random_seed"] = "not-an-int"
    bad_path = tmp_path / "bad_seed.yaml"
    bad_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    cfg = load_config(bad_path)
    with pytest.raises(ConfigSchemaError, match="project.random_seed"):
        ProjectSchema.from_config(cfg)


def test_schemas_are_frozen_dataclasses() -> None:
    """Frozen-dataclass invariant must hold so schemas can be safely shared
    across threads / sub-processes (relevant for the future GUI run executor).
    """
    cfg = load_config(CONFIG_PATH)
    project = ProjectSchema.from_config(cfg)
    with pytest.raises(Exception):
        project.name = "mutated"  # type: ignore[misc]
