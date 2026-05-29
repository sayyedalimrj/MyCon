"""Google Colab integration package for the MyCon pipeline.

This package wraps the existing ``pipeline.stage_*`` runners and
``scripts/run_stage.py`` launcher with a Colab-friendly API:

- ``environment``: install system + Python dependencies in a Colab-safe order.
- ``drive``: mount Google Drive (resiliently) and create the persistent
  project tree, including persistent model/HF caches.
- ``sync``: resilient Drive sync (atomic writes, retries, remount detection,
  background mirror daemon) so caches survive disconnects.
- ``checkpoint``: portable checkpoint/resume manager (per-stage run state,
  output-verified completion, cross-device resume).
- ``models``: automated model/asset provisioning (COLMAP/ffmpeg checks,
  Ollama + Qwen-VL for a real local VLM, Hugging Face snapshots).
- ``config_manager``: clone ``configs/site01.yaml`` and apply an execution
  profile (colab_safe / colab_gpu / production) plus user overrides.
- ``log_capture``: thread-safe ring buffer for streaming subprocess logs to
  the Gradio UI.
- ``stage_runner``: invoke ``scripts/run_stage.py`` (and Stage 10 / Stage 11)
  with live log streaming, checkpoint/resume, per-stage retries, and
  per-stage memory cleanup.
- ``artifacts``: list, zip, and stage output files for download.
- ``cleanup``: gc + ``torch.cuda.empty_cache`` between heavy stages.
- ``ui``: Gradio Blocks UI that ties everything together.

The notebook itself only calls into this package; no pipeline imports happen
at notebook scope, so the UI can launch before all heavy deps are installed.
"""

from __future__ import annotations

__all__ = [
    "environment",
    "drive",
    "sync",
    "checkpoint",
    "models",
    "config_manager",
    "log_capture",
    "stage_runner",
    "artifacts",
    "cleanup",
    "ui",
]
