from __future__ import annotations

from pathlib import Path


def test_server_env_template_exists_and_uses_qwen_thinking_model() -> None:
    p = Path("env/server.env.example")
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "Qwen/Qwen3-VL-8B-Thinking" in text
    assert "qwen3-vl:8b-thinking" in text
    assert "DOWNLOAD_ON_SERVER_ONLY=true" in text
