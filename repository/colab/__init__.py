"""Google Colab integration package for the MyCon pipeline.

This package wraps the existing ``pipeline.stage_*`` runners and
``scripts/run_stage.py`` launcher with a Colab-friendly API:

- ``environment``: install system + Python dependencies in a Colab-safe order.
- ``drive``: mount Google Drive and create the persistent project tree.
- ``config_manager``: clone ``configs/site01.yaml`` and override the small
  set of keys the user actually edits (project root, run id, video, IFC,
  schedule, plus a few safe Colab knobs).
- ``log_capture``: thread-safe ring buffer for streaming subprocess logs to
  the Gradio UI.
- ``stage_runner``: invoke ``scripts/run_stage.py`` (and Stage 10 / Stage 11)
  with live log streaming and per-stage memory cleanup.
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
    "config_manager",
    "log_capture",
    "stage_runner",
    "artifacts",
    "cleanup",
    "ui",
]
