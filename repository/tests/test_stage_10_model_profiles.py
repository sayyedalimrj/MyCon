from __future__ import annotations

from pipeline.stage_10_copilot.model_profiles import recommend_profile


def test_a5000_profile_uses_selected_qwen3_vl_thinking_model() -> None:
    profile = recommend_profile("svc4", "Nvidia A5000", ram_gb=90, gpu_vram_gb=24)
    assert profile.recommended_provider == "ollama_local"
    assert profile.ollama_model == "qwen3-vl:8b-thinking"
    assert profile.hf_model == "Qwen/Qwen3-VL-8B-Thinking"


def test_3090_profile_uses_selected_qwen3_vl_thinking_model() -> None:
    profile = recommend_profile("svc1", "Nvidia 3090 Ti 24GB", ram_gb=58, gpu_vram_gb=24)
    assert profile.recommended_provider == "ollama_local"
    assert profile.ollama_model == "qwen3-vl:8b-thinking"
    assert profile.hf_model == "Qwen/Qwen3-VL-8B-Thinking"
