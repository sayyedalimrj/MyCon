"""Reject misleading Stage-7.5 evidence about nonexistent real model inference."""

import pytest
from pipeline.stage_07_5_vlm_qa.run_vlm_qa import run_vlm_qa

pytestmark = pytest.mark.lightweight

def test_stage_75_real_provider_does_not_silently_return_mock_qa():
    with pytest.raises(RuntimeError, match="deterministic QA only"):
        run_vlm_qa({"vlm_qa": {"provider": "ollama_local"}})

def test_stage_75_does_not_inherit_real_copilot_provider(monkeypatch):
    # Reaching the input gate means real Stage-10 provider was NOT inherited.
    from pipeline.stage_07_5_vlm_qa import run_vlm_qa as module
    def stop_at_input_gate(cfg):
        raise RuntimeError("input-gate-reached")
    monkeypatch.setattr(module, "stage75_paths", stop_at_input_gate)
    with pytest.raises(RuntimeError, match="input-gate-reached"):
        run_vlm_qa({"copilot": {"vlm": {"provider": "qwen_local"}}})
