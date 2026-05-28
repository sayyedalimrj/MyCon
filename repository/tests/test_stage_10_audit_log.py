from __future__ import annotations

import json
from pathlib import Path

from pipeline.stage_10_copilot.audit_log import build_copilot_audit_record, write_copilot_audit_record


def test_build_copilot_audit_record_contains_validation_and_model_metadata() -> None:
    cfg = {
        "project": {"root": "."},
        "copilot": {
            "vlm": {
                "model": "qwen3-vl:8b-thinking",
                "hf_model": "Qwen/Qwen3-VL-8B-Thinking",
            }
        },
    }
    answer = {
        "answer": "Direct answer: Do not accept.",
        "confidence": "low",
        "provider": "mock",
        "evidence_used": ["evidence.json"],
        "answer_validation": {"status": "pass", "passed": True, "failures": [], "warnings": []},
    }

    record = build_copilot_audit_record(
        cfg=cfg,
        request_payload={"question": "Can we accept?"},
        answer_payload=answer,
    )

    assert record["stage"] == "stage_10_copilot_audit"
    assert record["model"] == "qwen3-vl:8b-thinking"
    assert record["hf_model"] == "Qwen/Qwen3-VL-8B-Thinking"
    assert record["validation_passed"] is True
    assert record["request_payload"]["question"] == "Can we accept?"
