"""Local/offline VLM clients for Stage 10 Construction Copilot.

This module intentionally supports only local or explicitly private endpoints.
The VLM is allowed to explain evidence, but it is not allowed to create metric
truth. Numeric facts must come from the deterministic evidence package.
"""
from __future__ import annotations

import base64
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any

from .config_access import cfg_get
from .evidence_builder import EvidencePackage


class LocalVLMError(RuntimeError):
    """Raised when a local VLM request cannot be completed."""


class LocalVLMConfigError(LocalVLMError):
    """Raised when VLM config violates local/offline safety rules."""


@dataclass(frozen=True)
class LocalVLMResult:
    text: str
    provider: str
    raw_response: dict[str, Any]


LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1", "host.docker.internal", "ollama"}


def _is_private_hostname(hostname: str) -> bool:
    host = hostname.strip().lower().strip("[]")
    if host in LOCAL_HOSTNAMES:
        return True
    try:
        ip = ip_address(host)
        return ip.is_loopback or ip.is_private
    except ValueError:
        return False


def validate_local_endpoint(endpoint: str, *, allow_private_lan: bool = False) -> str:
    """Validate endpoint is local/offline-safe and return the normalized URL."""
    url = endpoint.strip()
    if not url:
        raise LocalVLMConfigError("VLM endpoint is empty.")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LocalVLMConfigError(f"Invalid VLM endpoint: {endpoint!r}")
    host = parsed.hostname.lower()
    if host in LOCAL_HOSTNAMES:
        return url
    if allow_private_lan and _is_private_hostname(host):
        return url
    raise LocalVLMConfigError(
        "Refusing non-local VLM endpoint while copilot.vlm.local_only=true: "
        f"{endpoint!r}. Use localhost/127.0.0.1/host.docker.internal or set "
        "allow_private_lan only for an on-prem private server."
    )


def _read_json_response(request: urllib.request.Request, timeout_sec: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise LocalVLMError(f"Local VLM request failed: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LocalVLMError(f"Local VLM returned non-JSON response: {raw[:500]}") from exc
    if not isinstance(payload, dict):
        raise LocalVLMError("Local VLM response must be a JSON object.")
    return payload


def _existing_image_paths(package: EvidencePackage, max_images: int) -> list[Path]:
    paths: list[Path] = []
    for value in package.image_paths.values():
        path = Path(value)
        if path.exists() and path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            paths.append(path)
        if len(paths) >= max_images:
            break
    return paths


def _image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
    return f"data:{mime};base64,{_image_base64(path)}"


def _compact_json(value: Any, max_chars: int = 16000) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def build_evidence_prompt(package: EvidencePackage) -> str:
    """Create the text part sent to the local VLM."""
    evidence = package.to_dict()
    return (
        "You are answering a construction progress monitoring question using an evidence package.\n"
        "Strict rules:\n"
        "1. Do not invent measurements.\n"
        "2. Numeric claims must come only from the provided metrics JSON/CSV-derived evidence.\n"
        "3. Treat rendered images as visual evidence only, not metric truth.\n"
        "4. If Stage 8/9 artifacts are missing, say exactly which artifacts are missing.\n"
        "5. Answer in this structure: Direct answer; Evidence used; Metric facts; Visual observations; "
        "Confidence level; Risks or uncertainty; Recommended next action.\n\n"
        f"Question: {package.question}\n\n"
        "Evidence package JSON:\n"
        f"{_compact_json(evidence)}\n"
    )


def _ollama_endpoint(cfg: Any) -> str:
    endpoint = str(cfg_get(cfg, "copilot.vlm.endpoint", "http://host.docker.internal:11434/api/chat"))
    if endpoint.rstrip("/").endswith("/api/chat"):
        return endpoint
    return endpoint.rstrip("/") + "/api/chat"


def call_ollama_local(cfg: Any, package: EvidencePackage) -> LocalVLMResult:
    """Call a local Ollama VLM using /api/chat with base64 image inputs."""
    allow_lan = bool(cfg_get(cfg, "copilot.vlm.allow_private_lan", False))
    endpoint = validate_local_endpoint(_ollama_endpoint(cfg), allow_private_lan=allow_lan)
    model = str(cfg_get(cfg, "copilot.vlm.model", "qwen3-vl:8b"))
    timeout_sec = float(cfg_get(cfg, "copilot.vlm.timeout_sec", 180))
    max_images = int(cfg_get(cfg, "copilot.vlm.max_images", 4))
    images = [_image_base64(path) for path in _existing_image_paths(package, max_images=max_images)]
    prompt = build_evidence_prompt(package)
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a local offline Construction Copilot. Use only provided evidence."},
            {"role": "user", "content": prompt, "images": images},
        ],
        "options": {
            "temperature": float(cfg_get(cfg, "copilot.vlm.temperature", 0.1)),
            "num_predict": int(cfg_get(cfg, "copilot.vlm.max_tokens", 2048)),
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response = _read_json_response(request, timeout_sec)
    text = ""
    if isinstance(response.get("message"), dict):
        text = str(response["message"].get("content", ""))
    text = text or str(response.get("response") or response.get("text") or "")
    if not text.strip():
        raise LocalVLMError("Ollama returned an empty answer.")
    return LocalVLMResult(text=text, provider="ollama_local", raw_response=response)


def _openai_endpoint(cfg: Any) -> str:
    endpoint = str(cfg_get(cfg, "copilot.vlm.endpoint", "http://host.docker.internal:8000/v1/chat/completions"))
    if endpoint.rstrip("/").endswith("/chat/completions"):
        return endpoint
    return endpoint.rstrip("/") + "/v1/chat/completions"


def call_openai_compatible_local(cfg: Any, package: EvidencePackage) -> LocalVLMResult:
    """Call a local OpenAI-compatible multimodal server such as vLLM or LM Studio."""
    allow_lan = bool(cfg_get(cfg, "copilot.vlm.allow_private_lan", False))
    endpoint = validate_local_endpoint(_openai_endpoint(cfg), allow_private_lan=allow_lan)
    model = str(cfg_get(cfg, "copilot.vlm.model", "Qwen/Qwen3-VL-8B-Instruct"))
    timeout_sec = float(cfg_get(cfg, "copilot.vlm.timeout_sec", 180))
    max_images = int(cfg_get(cfg, "copilot.vlm.max_images", 4))
    content: list[dict[str, Any]] = [{"type": "text", "text": build_evidence_prompt(package)}]
    for path in _existing_image_paths(package, max_images=max_images):
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(path)}})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a local offline Construction Copilot. Use only provided evidence."},
            {"role": "user", "content": content},
        ],
        "temperature": float(cfg_get(cfg, "copilot.vlm.temperature", 0.1)),
        "max_tokens": int(cfg_get(cfg, "copilot.vlm.max_tokens", 2048)),
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response = _read_json_response(request, timeout_sec)
    choices = response.get("choices")
    text = ""
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        text = str(message.get("content") or choices[0].get("text") or "")
    text = text or str(response.get("answer") or response.get("response") or response.get("text") or "")
    if not text.strip():
        raise LocalVLMError("OpenAI-compatible local VLM returned an empty answer.")
    return LocalVLMResult(text=text, provider="openai_compatible_local", raw_response=response)
