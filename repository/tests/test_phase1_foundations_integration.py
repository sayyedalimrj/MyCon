"""Integration test: schema + registry + provenance + plugins compose.

This file does not test any single foundation module; it tests that the four
modules together form the contract Phase 2 (REST API) and Phase 3 (GUI) will
consume. Specifically:

- For each registered stage, the typed schema view can be built from the
  real ``configs/site01.yaml``.
- The registry's ``required_config_keys`` for that stage all resolve in the
  loaded config.
- A provenance envelope can be built and attached to a synthetic stage report.
- Encoding the entire (stages × schemas × provenance × plugins) snapshot to
  JSON yields a payload the future GUI can consume directly.

If any of these breaks, the GUI integration in a later phase will discover
it earlier here, against a small synthetic test, instead of at deploy time.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.common.config import load_config
from pipeline.common.plugins import DEPTH_REGISTRY, VLM_REGISTRY
from pipeline.common.provenance import attach_provenance, current_provenance
from pipeline.common.registry import STAGE_REGISTRY

CONFIG_PATH = Path("configs/site01.yaml")


def test_every_registered_stage_has_a_buildable_schema() -> None:
    cfg = load_config(CONFIG_PATH)
    for descriptor in STAGE_REGISTRY:
        schema = descriptor.schema_class.from_config(cfg)
        assert schema is not None, f"{descriptor.name}: schema_class.from_config returned None"


def test_every_registered_stage_required_keys_resolve_in_real_config() -> None:
    cfg = load_config(CONFIG_PATH)
    for descriptor in STAGE_REGISTRY:
        for dotted in descriptor.required_config_keys():
            value = cfg.require(dotted)
            assert value is not None, f"{descriptor.name}: {dotted} resolved to None"


def test_provenance_envelope_attaches_to_synthetic_report_for_each_stage() -> None:
    cfg = load_config(CONFIG_PATH)
    for descriptor in STAGE_REGISTRY:
        envelope = current_provenance(
            cfg,
            stage=descriptor.name,
            artifact_name=(descriptor.report_basename or descriptor.name).replace(".json", ""),
        )
        report = {"stage": descriptor.name, "status": "complete", "elapsed_sec": 1.0}
        attach_provenance(report, envelope)
        assert report["provenance"]["stage"] == descriptor.name


def test_full_phase1_snapshot_is_json_round_trippable() -> None:
    """Encode the GUI-shaped payload and round-trip it through JSON to catch
    any non-serializable corner before Phase 2 builds an API around it.
    """
    cfg = load_config(CONFIG_PATH)
    envelope = current_provenance(cfg, stage="stage_09_progress", artifact_name="progress_summary")
    snapshot = {
        "stages": STAGE_REGISTRY.to_dict(),
        "vlm_backends": VLM_REGISTRY.to_dict(),
        "depth_providers": DEPTH_REGISTRY.to_dict(),
        "example_provenance": envelope.to_dict(),
    }
    encoded = json.dumps(snapshot)
    decoded = json.loads(encoded)
    assert len(decoded["stages"]) == len(STAGE_REGISTRY)
    assert {b["name"] for b in decoded["vlm_backends"]} == set(VLM_REGISTRY.names())
    assert {p["name"] for p in decoded["depth_providers"]} == set(DEPTH_REGISTRY.names())
    assert decoded["example_provenance"]["config_hash"] == envelope.config_hash


def test_provenance_changes_when_stage_or_artifact_changes() -> None:
    cfg = load_config(CONFIG_PATH)
    e1 = current_provenance(cfg, stage="stage_09_progress", artifact_name="progress_summary")
    e2 = current_provenance(cfg, stage="stage_08_bim_registration", artifact_name="registration_report")
    # Same config_hash (same cfg), but different stage/artifact.
    assert e1.config_hash == e2.config_hash
    assert e1.stage != e2.stage
    assert e1.artifact_name != e2.artifact_name


def test_registry_dependency_graph_lines_up_with_documented_order() -> None:
    """The hand-edited ``order`` field must produce a topologically-valid
    iteration order. This is the same check ``build_default_registry``
    performs at import; we re-assert it here so a future edit that breaks
    it is caught even if registry import itself silently succeeds.
    """
    ordered = STAGE_REGISTRY.topological_order()
    seen: set[str] = set()
    for descriptor in ordered:
        for dep in descriptor.dependencies:
            assert dep in seen, (
                f"{descriptor.name} (order={descriptor.order}) depends on {dep!r} "
                f"which has not yet been visited; topological order broken."
            )
        seen.add(descriptor.name)
