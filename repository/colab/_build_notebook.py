"""Generate ``MyCon_Colab_Pipeline.ipynb`` deterministically.

This helper is run once at commit time so the .ipynb on disk is exact and
diff-friendly. It is not used at runtime — the notebook itself drives the
pipeline via the ``colab.*`` modules.

Usage:

    python3 colab/_build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + ("\n" if not line.endswith("\n") else "") for line in lines],
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + ("\n" if not line.endswith("\n") else "") for line in lines],
    }


CELLS: list[dict] = [
    md(
        "# MyCon — Colab Pipeline (3D reconstruction + BIM + VLM)",
        "",
        "Run the existing MyCon pipeline (Stages 1–11) on Google Colab with a Gradio UI.",
        "Outputs are written directly to **Google Drive** so a Colab disconnect does not lose work.",
        "",
        "## What this notebook does",
        "1. Mounts Google Drive and creates a persistent project tree.",
        "2. Installs system + Python dependencies in a Colab-safe order.",
        "3. Lets you upload a video (and optional IFC + schedule).",
        "4. Generates an effective YAML config from `configs/site01.yaml` with",
        "   Colab-safe defaults (smaller dense images, mock VLM, etc.).",
        "5. Runs the pipeline **stage by stage**, with live logs, memory cleanup",
        "   between heavy stages, and a status checkpoint after every stage.",
        "6. Exposes a Gradio UI with one tab per concern: Project & Inputs,",
        "   Run Pipeline, Artifacts & Downloads, Environment & Cleanup.",
        "",
        "**Tip:** For free Colab, keep the *Run Colab-safe default subset* button as your",
        "first run. It avoids COLMAP dense / DA3 / BIM stages that are likely to OOM.",
    ),
    md(
        "## 0. Runtime check (read this first)",
        "",
        "1. **Runtime → Change runtime type → GPU (T4)** before running heavy stages.",
        "2. **Runtime → Change runtime type → High-RAM** if Stage 5/7 OOM on standard.",
        "3. The notebook is safe to re-run cell by cell. Each cell is idempotent.",
    ),
    code(
        "# 1) Make sure we are running where we expect to run.",
        "import os, sys, platform, shutil, subprocess",
        "print('python   :', sys.version.split()[0])",
        "print('platform :', platform.platform())",
        "print('cwd      :', os.getcwd())",
        "print('is_colab :', 'google.colab' in sys.modules or os.path.exists('/content'))",
        "for tool in ('ffmpeg', 'colmap', 'git'):",
        "    print(f'{tool:7s}: {shutil.which(tool) or \"NOT FOUND\"}')",
        "try:",
        "    out = subprocess.check_output(['nvidia-smi', '-L'], text=True)",
        "    print('\\nGPU:\\n' + out)",
        "except Exception as exc:",
        "    print('\\nnvidia-smi:', exc)",
    ),
    md(
        "## 1. Clone the MyCon repository",
        "",
        "Edit the `REPO_URL` / `BRANCH` constants if you forked the project.",
        "If the repo is already cloned (you re-attached a Colab session), this cell is a no-op.",
    ),
    code(
        "import os, subprocess, sys, pathlib",
        "REPO_URL = 'https://github.com/sayyedalimrj/MyCon.git'",
        "BRANCH   = 'main'",
        "CLONE_DIR = pathlib.Path('/content/MyCon')",
        "if not CLONE_DIR.exists():",
        "    subprocess.check_call(['git', 'clone', '--depth', '1', '-b', BRANCH, REPO_URL, str(CLONE_DIR)])",
        "REPO_ROOT = CLONE_DIR / 'repository' if (CLONE_DIR / 'repository').is_dir() else CLONE_DIR",
        "print('REPO_ROOT =', REPO_ROOT)",
        "if str(REPO_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(REPO_ROOT))",
        "os.chdir(REPO_ROOT)",
    ),
    md(
        "## 2. Install dependencies (system + Python)",
        "",
        "Run this cell once per Colab session. It is idempotent and safe to re-run.",
        "Total install time: ~3–6 minutes the first time.",
    ),
    code(
        "from colab.environment import install_apt_packages, install_python_dependencies, validate_environment, format_validation",
        "from colab.log_capture import LogBuffer",
        "boot_log = LogBuffer(max_lines=4000)",
        "print('Installing apt packages (ffmpeg, colmap, ...)')",
        "apt_result = install_apt_packages(log=boot_log)",
        "print(' ->', apt_result)",
        "print('\\nInstalling Python dependencies (this may take a few minutes)...')",
        "py_results = install_python_dependencies(repo_root=REPO_ROOT, install_da3=True, install_ui=True, log=boot_log)",
        "for r in py_results:",
        "    print(f\"  [{ 'OK ' if r.ok else 'FAIL' }] { r.name } — { r.detail }\")",
    ),
    code(
        "# Probe every dependency layer.",
        "validation = validate_environment(log=boot_log)",
        "print(format_validation(validation))",
    ),
    md(
        "## 3. Mount Google Drive and create the project tree",
        "",
        "We persist all outputs under `MyDrive/MyCon_Colab/projects/<run_id>/`.",
        "If Drive is unavailable (rare), we fall back to `/content/MyCon_Colab/`.",
    ),
    code(
        "from colab.drive import mount_drive, setup_project_tree, DEFAULT_DRIVE_BASE",
        "from colab.config_manager import default_run_id",
        "RUN_ID = default_run_id()  # e.g. '2026-05-28_143012_colab'",
        "mount_drive(log=boot_log)",
        "PROJECT = setup_project_tree(run_id=RUN_ID, drive_base=DEFAULT_DRIVE_BASE, log=boot_log)",
        "for k, v in PROJECT.as_dict().items():",
        "    print(f'  {k:20s} = {v}')",
    ),
    md(
        "## 4. Launch the Gradio UI",
        "",
        "Click the public `share` URL printed below to use the UI from any browser.",
        "All actions are mirrored to logs on Drive at",
        "`<project_root>/runs/<run_id>/logs/<stage>.log`.",
        "",
        "If Gradio fails to start, run the cell in *Section 5* (Manual stage execution) instead —",
        "the notebook can drive the pipeline directly without the UI.",
    ),
    code(
        "from colab.ui import build_ui",
        "ui = build_ui(repo_root=REPO_ROOT, log=boot_log)",
        "ui.queue()",
        "_ = ui.launch(share=True, inline=True, prevent_thread_lock=True)",
    ),
    md(
        "## 5. Manual stage execution (fallback / advanced)",
        "",
        "The cells below replicate exactly what the Gradio UI does but from Python.",
        "Use them if Gradio is unavailable or you want to script a non-interactive run.",
    ),
    code(
        "# 5a) Build the effective YAML config and persist it on Drive.",
        "from colab.config_manager import build_effective_config, write_effective_config, validate_effective_config",
        "VIDEO_PATH    = None  # e.g. PROJECT.uploads_dir / 'site_walkthrough.mp4'",
        "IFC_PATH      = None  # e.g. PROJECT.uploads_dir / 'model.ifc'",
        "SCHEDULE_PATH = None  # e.g. PROJECT.uploads_dir / 'schedule.csv'",
        "USER_OVERRIDES = '''",
        "# Free-form YAML mapping merged on top of the safe defaults.",
        "# Example:",
        "# dense:",
        "#   max_image_size: 800",
        "# keyframes:",
        "#   max_frames_first_run: 60",
        "'''",
        "data = build_effective_config(",
        "    repo_root=REPO_ROOT,",
        "    project_root=PROJECT.project_root,",
        "    run_id=PROJECT.run_id,",
        "    project_name='colab_run',",
        "    video_path=VIDEO_PATH,",
        "    ifc_path=IFC_PATH,",
        "    schedule_path=SCHEDULE_PATH,",
        "    apply_safe_overrides=True,",
        "    user_overrides_yaml=USER_OVERRIDES,",
        "    log=boot_log,",
        ")",
        "CONFIG_PATH = write_effective_config(data=data, out_path=PROJECT.active_config_path, log=boot_log)",
        "ok, detail = validate_effective_config(config_path=CONFIG_PATH, repo_root=REPO_ROOT)",
        "print('config validates:', ok, '|', detail)",
        "print('config path     :', CONFIG_PATH)",
    ),
    code(
        "# 5b) Run a curated, Colab-safe subset of stages.",
        "from colab.stage_runner import run_stages, COLAB_SAFE_DEFAULT_KEYS",
        "results = run_stages(",
        "    spec_keys=list(COLAB_SAFE_DEFAULT_KEYS),",
        "    config_path=CONFIG_PATH,",
        "    repo_root=REPO_ROOT,",
        "    logs_dir=PROJECT.logs_dir,",
        "    reports_dir=PROJECT.reports_dir,",
        "    log=boot_log,",
        "    force=True,",
        "    log_level='INFO',",
        "    extra_kv={'question': 'Summarize available evidence.'},",
        "    stop_on_failure=True,",
        ")",
        "for r in results:",
        "    print(f\"  [{ 'OK ' if r.ok else 'FAIL' }] { r.key } rc={ r.return_code } { r.duration_sec:.1f }s\")",
    ),
    code(
        "# 5c) Run heavy SfM stages on demand. Requires GPU + good keyframe coverage.",
        "from colab.stage_runner import run_stages",
        "HEAVY_KEYS = ['stage_03_colmap', 'stage_04_refinement', 'stage_05_dense', 'stage_07_cleanup']",
        "# Uncomment to actually run:",
        "# results_heavy = run_stages(spec_keys=HEAVY_KEYS, config_path=CONFIG_PATH, repo_root=REPO_ROOT,",
        "#     logs_dir=PROJECT.logs_dir, reports_dir=PROJECT.reports_dir, log=boot_log, stop_on_failure=True)",
        "# for r in results_heavy:",
        "#     print(f\"[{'OK' if r.ok else 'FAIL'}] {r.key} rc={r.return_code} {r.duration_sec:.1f}s\")",
    ),
    md(
        "## 6. Inspect artifacts and build a download bundle",
    ),
    code(
        "from colab.artifacts import collect_artifacts, build_artifact_bundle",
        "entries = collect_artifacts(PROJECT.project_root)",
        "for e in entries[:60]:",
        "    print(f'  {e.category:16s} {e.bytes/1024:8.1f} KB  {e.relative_path}')",
        "if len(entries) > 60:",
        "    print(f'  ... ({len(entries)-60} more)')",
        "# Build a zip bundle on Drive:",
        "# bundle = build_artifact_bundle(project_root=PROJECT.project_root, exports_dir=PROJECT.exports_dir,",
        "#                                categories=['reports','exports_viewer','cleanup','dense','vlm_qa'])",
        "# print('bundle:', bundle)",
    ),
    md(
        "## 7. Memory cleanup utilities",
        "",
        "Always run `free_memory()` between heavy stages to release GPU VRAM.",
        "The Gradio UI invokes this automatically after every stage.",
    ),
    code(
        "from colab.cleanup import free_memory, disk_usage_summary",
        "print(free_memory(verbose=True))",
        "for path, info in disk_usage_summary([REPO_ROOT, PROJECT.project_root]).items():",
        "    print(f'  {path:60s} -> {info}')",
    ),
    md(
        "## 8. Colab caveats / known limitations",
        "",
        "- **Stage 5 (dense COLMAP)** can OOM on a free T4 with default settings.",
        "  Apply `dense.max_image_size: 800` (or smaller) in the config overrides.",
        "- **Stage 6 (DA3)** runs in `provider=precomputed` mode by default. To run a real",
        "  DA3 model in Colab, configure `da3.provider` and `da3.external_command` and",
        "  ensure the `requirements-da3.txt` deps are installed.",
        "- **Stage 8b / 9** require a real BIM IFC + matched scan; on free Colab they will",
        "  often log `skipped_insufficient_anchors` rather than register.",
        "- **Stage 10 (VLM ask)** uses the **mock** provider by default. Real Qwen 3-VL",
        "  needs Ollama or a vLLM endpoint reachable from the Colab runtime.",
        "- Free Colab disconnects after ~12 h. Because everything is on Drive, just",
        "  re-run cells 0–4 in a new session and continue.",
    ),
]


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
        "colab": {
            "provenance": [],
            "gpuType": "T4",
        },
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> int:
    out_path = Path(__file__).resolve().parents[1] / "MyCon_Colab_Pipeline.ipynb"
    out_path.write_text(json.dumps(NOTEBOOK, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
