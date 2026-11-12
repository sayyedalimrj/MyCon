"""Reject misleading Stage-7.5 evidence about nonexistent real model inference."""

import pytest
from pipeline.stage_07_5_vlm_qa.run_vlm_qa import run_vlm_qa

pytestmark = pytest.mark.lightweight

def test_stage_75_real_provider_does_not_silently_return_mock_qa():
    with pytest.raises(RuntimeError, match="deterministic QA only"):
        run_vlm_qa({"vlm_qa": {"provider": "ollama_local"}})

def test_stage_75_inherited_copilot_provider_is_not_implied_qa_model():
    with pytest.raises(RuntimeError, match="deterministic QA only"):
        run_vlm_qa({"copilot": {"vlm": {"provider": "qwen_local"}}})
