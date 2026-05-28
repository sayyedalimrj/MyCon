# Qwen VLM Laptop-to-Server Plan

## Goal

Keep laptop development cheap and safe while preparing the server to download and reuse models later.

## Selected model

- Hugging Face: `Qwen/Qwen3-VL-8B-Thinking`
- Ollama: `qwen3-vl:8b-thinking`

## Laptop rule

On the laptop, the default config should remain safe:

- `copilot.vlm.provider: mock`
- model downloads disabled
- no mandatory Ollama dependency
- no server-only training dependency

## Server rule

On the server, apply a generated/server profile only after model cache preparation.

## Profile files

- `configs/local_qwen_vlm_profile.yaml`
- `configs/server_qwen_vlm_profile.yaml`

## Generate a laptop-safe Qwen config

    python3 scripts/apply_vlm_profile.py \
      --base configs/site01.yaml \
      --profile configs/local_qwen_vlm_profile.yaml \
      --output configs/site01_qwen_local.yaml \
      --force

## Generate a strict server Qwen config

    python3 scripts/apply_vlm_profile.py \
      --base configs/site01.yaml \
      --profile configs/server_qwen_vlm_profile.yaml \
      --output configs/site01_qwen_server.yaml \
      --force

## Check local endpoint without downloads

    python3 scripts/check_local_vlm_connection.py \
      --config configs/site01_qwen_local.yaml

## Server-only model download

    python3 scripts/server_prepare_model_cache.py \
      --config configs/site01.yaml \
      --provider both \
      --execute

## Important

The VLM is an evidence assistant. It must not override deterministic Stage 8/9 metrics.
