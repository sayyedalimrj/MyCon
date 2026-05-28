from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def deep_get(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def normalize_base_from_chat_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    for suffix in ["/api/chat", "/api/generate"]:
        if endpoint.endswith(suffix):
            return endpoint[: -len(suffix)]
    return endpoint


def get_tags(base_url: str, timeout_sec: int) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/tags"
    with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local Ollama/Qwen connectivity without downloading models.")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout-sec", type=int, default=10)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if endpoint or selected model is unavailable.")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    endpoint = args.endpoint or deep_get(cfg, "copilot.vlm.endpoint", "http://host.docker.internal:11434/api/chat")
    model = args.model or deep_get(cfg, "copilot.vlm.model", "qwen3-vl:8b-thinking")
    base_url = normalize_base_from_chat_endpoint(endpoint)

    print("LOCAL_VLM_CONNECTION_CHECK")
    print(f"config={args.config}")
    print(f"endpoint={endpoint}")
    print(f"base_url={base_url}")
    print(f"selected_model={model}")

    try:
        tags = get_tags(base_url, args.timeout_sec)
        models = []
        for item in tags.get("models", []):
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if name:
                    models.append(str(name))
        present = any(m == model or m.startswith(model + ":") or model.startswith(m + ":") for m in models)

        print("ollama_reachable: true")
        print("available_models:", models)
        print("selected_model_present:", present)

        if args.strict and not present:
            raise SystemExit("SELECTED_MODEL_NOT_PRESENT_NO_DOWNLOAD_ATTEMPTED")

    except Exception as exc:
        print("ollama_reachable: false")
        print(f"error: {exc}")
        if args.strict:
            raise SystemExit(1) from exc

    print("LOCAL_VLM_CONNECTION_CHECK_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
