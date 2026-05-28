"""Tests for ``pipeline.common.registry``.

These tests pin the contract that every downstream consumer of the registry
relies on:

- Every stage descriptor in the canonical registry refers to a CLI module
  and callable that *actually exists*.
- Dependencies form a valid DAG; ``order`` is a topological ordering.
- ``required_config_keys`` is delegated to the typed schema view, so the
  registry cannot list a key the schema doesn't read.
- Capability tags belong to the documented set.
- ``to_dict()`` is JSON-round-trippable so the future GUI / API layer can
  consume it directly.
"""

from __future__ import annotations

import importlib
import json

import pytest

from pipeline.common.registry import (
    STAGE_REGISTRY,
    RegistryError,
    StageCapability,
    StageDescriptor,
    StageRegistry,
    build_default_registry,
)


def test_default_registry_size_is_canonical() -> None:
    # The shipping pipeline has 14 distinct ``run_*`` modules. Stage 8 has
    # two entries (metric_alignment + registration), so the registry has 15.
    assert len(STAGE_REGISTRY) == 15


def test_every_descriptor_has_unique_name() -> None:
    names = [d.name for d in STAGE_REGISTRY]
    assert len(names) == len(set(names))


def test_every_descriptor_has_resolvable_callable() -> None:
    """``descriptor.callable()`` must successfully load the in-process entry-point."""
    for d in STAGE_REGISTRY:
        fn = d.callable()
        assert callable(fn), f"{d.name}: {d.cli_module}.{d.callable_name} is not callable"


def test_every_descriptor_cli_module_is_importable() -> None:
    """``cli_module`` must be a real importable Python module."""
    for d in STAGE_REGISTRY:
        try:
            importlib.import_module(d.cli_module)
        except Exception as exc:  # pragma: no cover - failure path
            pytest.fail(f"{d.name}: cli_module {d.cli_module!r} not importable: {exc}")


def test_dependency_graph_is_valid_dag() -> None:
    STAGE_REGISTRY.validate_dependencies()
    ordered = STAGE_REGISTRY.topological_order()
    assert len(ordered) == len(STAGE_REGISTRY)


def test_capabilities_belong_to_documented_set() -> None:
    valid = StageCapability.all()
    for d in STAGE_REGISTRY:
        unknown = d.capabilities - valid
        assert not unknown, f"{d.name} has unknown capabilities {sorted(unknown)}"


def test_required_config_keys_delegate_to_schema() -> None:
    """``StageDescriptor.required_config_keys`` must return exactly what
    its bound schema class reports.
    """
    for d in STAGE_REGISTRY:
        descriptor_keys = tuple(d.required_config_keys())
        schema_keys = tuple(d.schema_class.required_config_keys())
        assert descriptor_keys == schema_keys


def test_inputs_and_outputs_are_dotted_yaml_keys() -> None:
    """Every input/output is a dotted YAML key; the GUI uses this assumption."""
    for d in STAGE_REGISTRY:
        for key in d.inputs + d.outputs:
            assert "." in key, f"{d.name}: {key!r} is not a dotted YAML key"


def test_to_dict_is_json_round_trippable() -> None:
    payload = STAGE_REGISTRY.to_dict()
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert isinstance(decoded, list)
    assert len(decoded) == len(STAGE_REGISTRY)
    sample = decoded[0]
    assert "name" in sample
    assert "required_config_keys" in sample
    assert isinstance(sample["required_config_keys"], list)


def test_cli_invocation_returns_runnable_argv() -> None:
    """``cli_invocation`` must produce a ``python3 -m`` argv that argparse
    will accept (we don't run it; we just check the shape).
    """
    d = STAGE_REGISTRY.get("stage_01_ingest")
    argv = d.cli_invocation("configs/site01.yaml")
    assert argv[:3] == ("python3", "-m", "pipeline.stage_01_ingest.run_ingest")
    assert "--config" in argv
    assert "configs/site01.yaml" in argv


def test_register_rejects_duplicate_names() -> None:
    fresh = build_default_registry()
    duplicate = StageDescriptor(
        name="stage_01_ingest",  # already in the registry
        order=999,
        title="dup",
        description="dup",
        cli_module="pipeline.stage_01_ingest.run_ingest",
        callable_name="run_ingest",
        schema_class=fresh.get("stage_01_ingest").schema_class,
    )
    with pytest.raises(RegistryError, match="already registered"):
        fresh.register(duplicate)


def test_register_rejects_unknown_capabilities() -> None:
    fresh = build_default_registry()
    bad = StageDescriptor(
        name="stage_xx_bogus",
        order=1000,
        title="bogus",
        description="bogus",
        cli_module="pipeline.stage_01_ingest.run_ingest",
        callable_name="run_ingest",
        schema_class=fresh.get("stage_01_ingest").schema_class,
        capabilities=frozenset({"not_a_real_capability"}),
    )
    with pytest.raises(RegistryError, match="unknown capabilities"):
        fresh.register(bad)


def test_get_unknown_stage_raises() -> None:
    with pytest.raises(RegistryError, match="Unknown stage"):
        STAGE_REGISTRY.get("stage_99_does_not_exist")


def test_validate_dependencies_detects_dangling_dependency() -> None:
    fresh = build_default_registry()
    schema_class = fresh.get("stage_01_ingest").schema_class
    bogus = StageDescriptor(
        name="stage_xx_dangling",
        order=999,
        title="dangling",
        description="dangling",
        cli_module="pipeline.stage_01_ingest.run_ingest",
        callable_name="run_ingest",
        schema_class=schema_class,
        dependencies=("stage_99_does_not_exist",),
    )
    fresh.register(bogus)
    with pytest.raises(RegistryError, match="depends on unknown stage"):
        fresh.validate_dependencies()


def test_topological_order_detects_order_violation() -> None:
    """If a stage's dependency has a *higher* order than itself, the
    topological-order method must catch it.
    """
    fresh = build_default_registry()
    schema_class = fresh.get("stage_01_ingest").schema_class
    bad = StageDescriptor(
        name="stage_xx_violator",
        order=5,  # earlier than stage_01_ingest (order 10)
        title="violator",
        description="violator",
        cli_module="pipeline.stage_01_ingest.run_ingest",
        callable_name="run_ingest",
        schema_class=schema_class,
        dependencies=("stage_01_ingest",),
    )
    fresh.register(bad)
    with pytest.raises(RegistryError, match="order violates DAG"):
        fresh.topological_order()
