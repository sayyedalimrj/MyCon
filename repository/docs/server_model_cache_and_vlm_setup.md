# Server Model Cache and Local VLM Setup

This document defines the server-side model cache contract.

## Purpose

Laptop development should not download large VLM/3DGS models.

Server execution should download models once, store them in persistent cache directories, and reuse them for future runs.

## Selected VLM

- Hugging Face model: `Qwen/Qwen3-VL-8B-Thinking`
- Ollama model name: `qwen3-vl:8b-thinking`
- Current laptop-safe provider: `mock`
- Server provider target: `ollama_local`
- Optional later provider: `openai_compatible_local` through vLLM or equivalent

## Cache directories

The server should keep these directories on persistent storage:

- `model_cache/ollama`
- `model_cache/huggingface`
- `model_cache/huggingface/hub`
- `model_cache/huggingface/transformers`
- `model_cache/nerfstudio`

## Current laptop command: dry-run only

    python3 scripts/server_prepare_model_cache.py \
      --config configs/site01.yaml \
      --provider both

This command must not download anything.

## Server command: write env and download/pull

    python3 scripts/server_prepare_model_cache.py \
      --config configs/site01.yaml \
      --provider both \
      --execute

## Ollama-only server command

    python3 scripts/server_prepare_model_cache.py \
      --config configs/site01.yaml \
      --provider ollama \
      --execute

## Hugging Face-only server command

    python3 scripts/server_prepare_model_cache.py \
      --config configs/site01.yaml \
      --provider hf \
      --execute

## Status check

    python3 scripts/server_model_cache_status.py \
      --config configs/site01.yaml

## Important

Do not commit downloaded models, caches, generated viewer exports, run artifacts, or training outputs.

The local VLM must be treated as evidence assistant only. It must not override deterministic Stage 8/9 metrics.
