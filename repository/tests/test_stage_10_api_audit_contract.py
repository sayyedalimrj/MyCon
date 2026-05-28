from pathlib import Path


def test_stage10_api_writes_audit_record_contract() -> None:
    text = Path("pipeline/stage_10_copilot/api.py").read_text(encoding="utf-8")
    assert "write_copilot_audit_record" in text
    assert "copilot_audit_path" in text
    assert "copilot_audit_log_failed" in text
    assert "with audit persistence" in text
