from __future__ import annotations

import argparse
import json
import os
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


def get_json(url: str, timeout: int = 10) -> dict[str, Any]:
    with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local/server model cache status without downloading anything.")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--ollama-endpoint", default=None)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    cfg = load_yaml(root / args.config)

    mc = cfg.get("model_cache", {}) if isinstance(cfg.get("model_cache", {}), dict) else {}
    paths = {
        "model_cache.root_dir": root / str(mc.get("root_dir", "model_cache")),
        "model_cache.ollama_models_dir": root / str(mc.get("ollama_models_dir", "model_cache/ollama")),
        "model_cache.hf_home": root / str(mc.get("hf_home", "model_cache/huggingface")),
        "model_cache.hf_hub_cache": root / str(mc.get("hf_hub_cache", "model_cache/huggingface/hub")),
        "model_cache.transformers_cache": root / str(mc.get("transformers_cache", "model_cache/huggingface/transformers")),
    }

    print("SERVER_MODEL_CACHE_STATUS")
    for key, p in paths.items():
        print(f"{key}: exists={p.exists()} size={p.stat().st_size if p.exists() and p.is_file() else 0} path={p}")

    ollama_model = deep_get(cfg, "model_cache.qwen_vlm.ollama_model", "qwen3-vl:8b-thinking")
    hf_model = deep_get(cfg, "model_cache.qwen_vlm.hf_model", "Qwen/Qwen3-VL-8B-Thinking")
    endpoint = args.ollama_endpoint or os.environ.get("OLLAMA_ENDPOINT") or deep_get(
        cfg, "model_cache.ollama.host_endpoint_for_server", "http://127.0.0.1:11434"
    )

    print(f"selected_ollama_model: {ollama_model}")
    print(f"selected_hf_model: {hf_model}")
    print(f"ollama_endpoint: {endpoint}")

    try:
        tags = get_json(endpoint)
        models = [m.get("name") or m.get("model") for m in tags.get("models", []) if isinstance(m, dict)]
        print("ollama_reachable: true")
        print("ollama_models:", models)
        print("selected_ollama_model_present:", any(str(x).split(":")[0] in ollama_model or ollama_model in str(x) for x in models))
    except Exception as exc:
        print("ollama_reachable: false")
        print(f"ollama_error: {exc}")

    print("SERVER_MODEL_CACHE_STATUS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
