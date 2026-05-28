from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _project_root(cfg: Any) -> Path:
    root = _cfg_get(cfg, "project.root", ".")
    return Path(str(root)).resolve()


def _audit_dir(cfg: Any) -> Path:
    raw = (
        _cfg_get(cfg, "copilot.paths.audit_dir")
        or _cfg_get(cfg, "copilot.audit_dir")
        or "runs/2026-04-30_site01_baseline/copilot/audit"
    )
    path = Path(str(raw))
    return path if path.is_absolute() else _project_root(cfg) / path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _short_hash(value: Any) -> str:
    blob = json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def build_copilot_audit_record(
    *,
    cfg: Any,
    request_payload: dict[str, Any] | None,
    answer_payload: dict[str, Any],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()

    provider = answer_payload.get("provider")
    confidence = answer_payload.get("confidence")
    validation = answer_payload.get("answer_validation") or {}

    return {
        "stage": "stage_10_copilot_audit",
        "created_at_utc": now,
        "provider": provider,
        "model": _cfg_get(cfg, "copilot.vlm.model") or _cfg_get(cfg, "model_cache.qwen_vlm.ollama_model"),
        "hf_model": _cfg_get(cfg, "copilot.vlm.hf_model") or _cfg_get(cfg, "model_cache.qwen_vlm.hf_model"),
        "confidence": confidence,
        "validation_status": validation.get("status"),
        "validation_passed": validation.get("passed"),
        "evidence_package_path": answer_payload.get("evidence_package_path"),
        "evidence_used": answer_payload.get("evidence_used") or [],
        "selected_element_id": answer_payload.get("selected_element_id"),
        "selected_activity_id": answer_payload.get("selected_activity_id"),
        "route": answer_payload.get("route"),
        "request_payload": _json_safe(request_payload or {}),
        "answer_payload": _json_safe(answer_payload),
    }


def write_copilot_audit_record(
    *,
    cfg: Any,
    request_payload: dict[str, Any] | None,
    answer_payload: dict[str, Any],
) -> Path:
    audit_dir = _audit_dir(cfg)
    audit_dir.mkdir(parents=True, exist_ok=True)

    record = build_copilot_audit_record(cfg=cfg, request_payload=request_payload, answer_payload=answer_payload)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = _short_hash({"request": request_payload or {}, "answer": answer_payload, "stamp": stamp})

    out = audit_dir / f"copilot_audit_{stamp}_{digest}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
