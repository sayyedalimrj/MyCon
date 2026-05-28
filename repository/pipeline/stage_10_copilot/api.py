"""Local Stage 10 Copilot API.

FastAPI is used when available. A stdlib HTTP server is also provided so the
backend remains testable without adding mandatory dependencies.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .evidence_builder import build_evidence_package
from .vlm_answer import answer_with_vlm
from .answer_validator import validate_copilot_answer_payload
from .audit_log import write_copilot_audit_record


def _ask_copilot_unvalidated(cfg: Any, payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise ValueError("question is required")
    package = build_evidence_package(
        cfg,
        question,
        element_global_id=payload.get("selected_element_id") or payload.get("element_global_id"),
        activity_id=payload.get("selected_activity_id") or payload.get("activity_id"),
        selected_bbox=payload.get("selected_bbox"),
        current_view=payload.get("current_view"),
        camera_pose=payload.get("camera_pose"),
        pointcloud_path=payload.get("pointcloud_path"),
        ifc_path=payload.get("ifc_path"),
        artifact_paths=payload.get("artifact_paths") or {},
    )
    answer = answer_with_vlm(cfg, package)
    return {
        "answer": answer.answer,
        "evidence_used": answer.evidence_used,
        "confidence": answer.confidence,
        "recommended_action": answer.recommended_action,
        "risks_or_uncertainty": answer.risks_or_uncertainty,
        "generated_view_paths": package.image_paths,
        "evidence_package_path": package.evidence_path,
        "selected_element_id": package.selected_element_id,
        "selected_activity_id": package.selected_activity_id,
        "route": package.route,
        "provider": answer.provider,
    }


def make_fastapi_app(cfg: Any) -> Any:
    try:
        from fastapi import FastAPI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FastAPI is not installed. Use run_stdlib_server or install fastapi/uvicorn.") from exc

    app = FastAPI(title="Construction Progress Copilot", version="0.1.0")

    @app.post("/ask")
    def ask(payload: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        return ask_copilot(cfg, payload)

    @app.get("/health")
    def health() -> dict[str, str]:  # type: ignore[override]
        return {"status": "ok"}

    return app


def run_stdlib_server(cfg: Any, host: str = "127.0.0.1", port: int = 8765) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _write(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._write(200, {"status": "ok"})
            else:
                self._write(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/ask":
                self._write(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._write(200, ask_copilot(cfg, payload))
            except Exception as exc:  # pragma: no cover
                self._write(400, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = HTTPServer((host, port), Handler)
    print(f"STAGE_10_API_READY http://{host}:{port}/ask")
    server.serve_forever()


def attach_answer_validation(response):
    """Attach deterministic evidence-policy validation to Stage 10 answers."""
    if not isinstance(response, dict):
        return response

    if "answer_validation" in response:
        return response

    validation = validate_copilot_answer_payload(response).to_dict()
    response["answer_validation"] = validation

    if not validation.get("passed", False):
        response["confidence"] = "low"

        risks = response.get("risks_or_uncertainty")
        if risks is None:
            risks = []
        elif isinstance(risks, str):
            risks = [risks]
        elif not isinstance(risks, list):
            risks = [str(risks)]

        risks.append("answer_validation_failed:" + ",".join(validation.get("failures", [])))
        response["risks_or_uncertainty"] = risks

    return response


def ask_copilot(*args, **kwargs):
    """Validated public Stage 10 copilot entrypoint with audit persistence."""
    response = _ask_copilot_unvalidated(*args, **kwargs)
    response = attach_answer_validation(response)

    cfg = kwargs.get("cfg")
    request_payload = kwargs.get("payload")

    if cfg is None and args:
        cfg = args[0]
    if request_payload is None and len(args) >= 2 and isinstance(args[1], dict):
        request_payload = args[1]

    if cfg is not None:
        try:
            audit_path = write_copilot_audit_record(
                cfg=cfg,
                request_payload=request_payload if isinstance(request_payload, dict) else {},
                answer_payload=response,
            )
            response["copilot_audit_path"] = str(audit_path)
        except Exception as exc:  # pragma: no cover
            risks = response.setdefault("risks_or_uncertainty", [])
            if isinstance(risks, str):
                risks = [risks]
            risks.append(f"copilot_audit_log_failed:{exc}")
            response["risks_or_uncertainty"] = risks

    return response
