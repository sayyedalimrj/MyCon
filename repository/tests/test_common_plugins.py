"""Tests for ``pipeline.common.plugins``.

Pinned contracts:

- The default registries contain the expected named plugins.
- Every registered factory produces a Protocol-conforming instance.
- The mock VLM backend is fully functional offline (no network).
- Adapters that defer to existing Stage 6 / Stage 10 implementations
  raise :class:`PluginError` with a helpful message when called via the
  plugin path *before* the Stage 6/10 routing refactor lands.
- Registration / lookup invariants (no duplicates, unknown name → error).
- ``to_dict`` is JSON-round-trippable for the future GUI / API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.common.plugins import (
    DEPTH_REGISTRY,
    VLM_REGISTRY,
    DepthProvider,
    DepthProviderRequest,
    DepthProviderResult,
    PluginError,
    PluginInfo,
    PluginRegistry,
    VLMBackend,
    VLMRequest,
    VLMResponse,
    build_default_plugin_registries,
)


def test_default_vlm_registry_has_three_named_backends() -> None:
    assert set(VLM_REGISTRY.names()) == {"mock", "ollama_local", "openai_compatible_local"}


def test_default_depth_registry_has_two_named_providers() -> None:
    assert set(DEPTH_REGISTRY.names()) == {"precomputed", "external_command"}


def test_every_vlm_factory_returns_protocol_conforming_instance() -> None:
    for info in VLM_REGISTRY:
        backend = info.factory()
        assert isinstance(backend, VLMBackend), f"{info.name} fails Protocol check"


def test_every_depth_factory_returns_protocol_conforming_instance() -> None:
    for info in DEPTH_REGISTRY:
        provider = info.factory()
        assert isinstance(provider, DepthProvider), f"{info.name} fails Protocol check"


def test_mock_vlm_answer_works_offline() -> None:
    backend = VLM_REGISTRY.get("mock").factory()
    req = VLMRequest(
        question="Is the wall complete?",
        system_prompt="strict",
        evidence_text="evidence",
        image_paths=(Path("a.jpg"), Path("b.jpg")),
    )
    res = backend.answer(req)
    assert isinstance(res, VLMResponse)
    assert res.provider == "mock"
    assert res.text  # non-empty
    # The mock must reflect the requested image count without ever opening files.
    assert res.raw_response["images"] == 2


def test_ollama_adapter_raises_until_routing_refactor_lands() -> None:
    """The adapter is registered for *discovery*, but until Stage 10 is
    refactored to consume the registry, calling ``answer`` via the plugin
    path must raise a clear :class:`PluginError` rather than appearing to
    work.
    """
    backend = VLM_REGISTRY.get("ollama_local").factory()
    with pytest.raises(PluginError, match="reserved for a later phase"):
        backend.answer(
            VLMRequest(question="q", system_prompt="s", evidence_text="e")
        )


def test_external_command_depth_adapter_raises_until_refactor() -> None:
    provider = DEPTH_REGISTRY.get("external_command").factory()
    with pytest.raises(PluginError, match="reserved for a later phase"):
        provider.materialize(DepthProviderRequest(image_dir=Path("/tmp"), output_dir=Path("/tmp")))


def test_precomputed_depth_provider_returns_not_configured_when_empty(tmp_path: Path) -> None:
    """The precomputed adapter is fully wired — it must return a structured
    ``not_configured`` result when no depth files are present, rather than
    raising.
    """
    images = tmp_path / "images"
    output = tmp_path / "depth"
    images.mkdir()
    output.mkdir()
    # Empty directories → no depth records → not_configured.
    provider = DEPTH_REGISTRY.get("precomputed").factory()
    result = provider.materialize(DepthProviderRequest(image_dir=images, output_dir=output))
    assert isinstance(result, DepthProviderResult)
    assert result.status == "not_configured"
    assert result.image_count == 0
    assert result.notes  # at least one note explaining why


def test_register_rejects_duplicate_name() -> None:
    fresh_vlm, _ = build_default_plugin_registries()
    duplicate = PluginInfo(
        name="mock",  # already present
        description="dup",
        factory=lambda: VLM_REGISTRY.get("mock").factory(),
    )
    with pytest.raises(PluginError, match="already registered"):
        fresh_vlm.register(duplicate)


def test_get_unknown_plugin_raises() -> None:
    with pytest.raises(PluginError, match="Unknown vlm_backend plugin"):
        VLM_REGISTRY.get("does_not_exist")


def test_registry_to_dict_is_json_round_trippable() -> None:
    payload = VLM_REGISTRY.to_dict()
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert isinstance(decoded, list)
    names = {entry["name"] for entry in decoded}
    assert names == set(VLM_REGISTRY.names())


def test_plugin_registry_iteration_is_ordered() -> None:
    """The order returned by iteration must match registration order so the
    GUI shows backends in a stable sequence.
    """
    names = list(VLM_REGISTRY.names())
    iterated = [info.name for info in VLM_REGISTRY]
    assert iterated == names


def test_factories_are_lazy() -> None:
    """Calling ``factory()`` twice must return distinct instances, so a GUI
    dispatching parallel requests does not share mutable state.
    """
    a = VLM_REGISTRY.get("mock").factory()
    b = VLM_REGISTRY.get("mock").factory()
    assert a is not b
