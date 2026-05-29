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
        "Production-ready Colab runner for the MyCon pipeline (Stages 1–11).",
        "",
        "**Built for long, unattended runs on unstable Colab runtimes:**",
        "- Every artifact, checkpoint, cache, model, render, log and intermediate",
        "  result is written to **Google Drive** (with a resilient background sync",
        "  daemon for fast local caches).",
        "- **Checkpoint/resume**: progress is recorded after every stage. If Colab",
        "  disconnects, re-attach, re-run cells 1–4 with the **same Run ID**, and the",
        "  pipeline skips everything already finished and continues.",
        "- **Automated setup**: system packages, Python deps, COLMAP, ffmpeg, and an",
        "  optional real local VLM (Ollama + Qwen-VL) are installed for you.",
        "- **Portable**: because the whole project tree (config + state + artifacts)",
        "  lives on Drive, you can resume from another device or Drive account.",
        "",
        "## Execution profiles",
        "- `colab_safe`  — bounded memory, mock VLM, no training. Always works on a free T4.",
        "- `colab_gpu`   — full single-GPU pipeline (real dense/cleanup); real VLM if provisioned.",
        "- `production`  — server-grade settings (use an A100/L4 high-RAM runtime).",
    ),
    md(
        "## 0. Runtime check (read this first)",
        "",
        "1. **Runtime → Change runtime type → GPU (T4 or better)** before heavy stages.",
        "2. **High-RAM** helps Stage 5/7. Every cell below is idempotent and safe to re-run.",
    ),
    code(
        "import os, sys, platform, shutil, subprocess",
        "print('python   :', sys.version.split()[0])",
        "print('platform :', platform.platform())",
        "print('is_colab :', 'google.colab' in sys.modules or os.path.exists('/content'))",
        "for tool in ('ffmpeg', 'colmap', 'git'):",
        "    print(f'{tool:7s}: {shutil.which(tool) or \"NOT FOUND (installed in step 2)\"}')",
        "try:",
        "    print('\\nGPU:\\n' + subprocess.check_output(['nvidia-smi', '-L'], text=True))",
        "except Exception as exc:",
        "    print('\\nnvidia-smi:', exc, '(no GPU — pick a GPU runtime for heavy stages)')",
    ),
    md(
        "## 1. Clone the MyCon repository",
        "",
        "Edit `REPO_URL` / `BRANCH` if you forked the project. Re-attaching a session is a no-op.",
    ),
    code(
        "import os, subprocess, sys, pathlib",
        "REPO_URL = 'https://github.com/sayyedalimrj/MyCon.git'",
        "BRANCH   = 'main'",
        "CLONE_DIR = pathlib.Path('/content/MyCon')",
        "if not CLONE_DIR.exists():",
        "    subprocess.check_call(['git', 'clone', '--depth', '1', '-b', BRANCH, REPO_URL, str(CLONE_DIR)])",
        "else:",
        "    subprocess.call(['git', '-C', str(CLONE_DIR), 'pull', '--ff-only'])",
        "REPO_ROOT = CLONE_DIR / 'repository' if (CLONE_DIR / 'repository').is_dir() else CLONE_DIR",
        "print('REPO_ROOT =', REPO_ROOT)",
        "if str(REPO_ROOT) not in sys.path:",
        "    sys.path.insert(0, str(REPO_ROOT))",
        "os.chdir(REPO_ROOT)",
    ),
    md(
        "## 2. Install dependencies (system + Python)",
        "",
        "One idempotent call installs apt packages (ffmpeg, colmap, zstd, ...), pins",
        "`numpy<2`, installs `requirements-core` + `requirements-da3` + the UI deps,",
        "and validates every layer. ~3–6 min the first time; near-instant on re-run.",
    ),
    code(
        "from colab.environment import bootstrap_environment, format_validation",
        "from colab.log_capture import LogBuffer",
        "boot_log = LogBuffer(max_lines=6000)",
        "summary = bootstrap_environment(REPO_ROOT, install_apt=True, install_da3=True, install_ui=True, log=boot_log)",
        "print('environment ok:', summary.get('ok'))",
        "print('gpu:', summary.get('gpu'))",
        "for r in summary.get('python', []):",
        "    print(f\"  [{ 'OK ' if r['ok'] else 'FAIL' }] { r['name'] } — { r['detail'] }\")",
    ),
    md(
        "## 3. Mount Google Drive and create / resume the project tree",
        "",
        "All outputs live under `MyDrive/MyCon_Colab/projects/<RUN_ID>/`. Re-use the same",
        "`RUN_ID` to resume a previous run. A background daemon mirrors fast local caches",
        "(Hugging Face, models) onto Drive every couple of minutes.",
    ),
    code(
        "from colab.drive import setup_project_tree, remount_drive, DEFAULT_DRIVE_BASE",
        "from colab.sync import DriveSyncManager",
        "from colab.checkpoint import CheckpointManager",
        "from colab.config_manager import default_run_id",
        "",
        "# Set a fixed RUN_ID (e.g. 'site_a_2026') to resume across sessions/devices.",
        "RUN_ID = default_run_id()",
        "PROFILE = 'colab_gpu'  # 'colab_safe' | 'colab_gpu' | 'production'",
        "",
        "PROJECT = setup_project_tree(run_id=RUN_ID, drive_base=DEFAULT_DRIVE_BASE, log=boot_log)",
        "for k, v in PROJECT.as_dict().items():",
        "    print(f'  {k:20s} = {v}')",
        "",
        "# Resilient background Drive sync for local caches.",
        "SYNC = None",
        "if PROJECT.on_drive:",
        "    SYNC = DriveSyncManager(drive_mount=PROJECT.drive_mount, log=boot_log, interval=120.0)",
        "    SYNC.set_remount_callback(lambda: remount_drive(log=boot_log))",
        "    SYNC.register(PROJECT.local_hf_cache_dir, PROJECT.hf_cache_dir)",
        "    SYNC.start_periodic()",
    ),
    md(
        "## 4. Launch the Gradio UI (recommended)",
        "",
        "Use the public `share` URL to drive the pipeline from any browser. The UI has",
        "tabs for *Project & Inputs*, *Run Pipeline* (with **Resume** + retries + a",
        "**Run FULL pipeline** button), *Artifacts & Downloads*, and *Environment,",
        "Models & Cleanup* (one-click **Provision real local VLM**).",
        "",
        "Prefer scripting? Skip to *Section 5* for the headless API.",
    ),
    code(
        "from colab.ui import build_ui",
        "ui = build_ui(repo_root=REPO_ROOT, log=boot_log)",
        "ui.queue()",
        "_ = ui.launch(share=True, inline=True, prevent_thread_lock=True)",
    ),
    md(
        "## 5. Headless / scripted execution (no UI)",
        "",
        "The cells below do exactly what the UI does, from Python. Use them for",
        "fully unattended runs.",
    ),
    code(
        "# 5a) Write the effective config (profile + your overrides) onto Drive.",
        "from colab.config_manager import build_effective_config, write_effective_config, validate_effective_config",
        "VIDEO_PATH    = None  # e.g. PROJECT.uploads_dir / 'site_walkthrough.mp4' (copy your video there first)",
        "IFC_PATH      = None  # e.g. PROJECT.uploads_dir / 'model.ifc'",
        "SCHEDULE_PATH = None  # e.g. PROJECT.uploads_dir / 'schedule.csv'",
        "USER_OVERRIDES = '''",
        "# Free-form YAML merged on top of the profile. Example:",
        "# dense:",
        "#   max_image_size: 800",
        "'''",
        "data = build_effective_config(",
        "    repo_root=REPO_ROOT, project_root=PROJECT.project_root, run_id=PROJECT.run_id,",
        "    project_name=PROJECT.run_id, video_path=VIDEO_PATH, ifc_path=IFC_PATH,",
        "    schedule_path=SCHEDULE_PATH, profile=PROFILE, user_overrides_yaml=USER_OVERRIDES, log=boot_log,",
        ")",
        "CONFIG_PATH = write_effective_config(data=data, out_path=PROJECT.active_config_path, log=boot_log)",
        "ok, detail = validate_effective_config(config_path=CONFIG_PATH, repo_root=REPO_ROOT)",
        "print('config validates:', ok, '|', detail, '|', CONFIG_PATH)",
    ),
    code(
        "# 5b) (Optional) Provision a REAL local VLM (Ollama + Qwen-VL), cached on Drive.",
        "#      Skip this to keep the deterministic mock provider.",
        "from colab.models import provision_vlm, DEFAULT_VLM_MODEL",
        "VLM = provision_vlm(model=DEFAULT_VLM_MODEL, models_dir=PROJECT.model_cache_dir / 'ollama', log=boot_log)",
        "print(VLM)",
        "if VLM.ok:",
        "    data = build_effective_config(",
        "        repo_root=REPO_ROOT, project_root=PROJECT.project_root, run_id=PROJECT.run_id,",
        "        project_name=PROJECT.run_id, profile=PROFILE,",
        "        override_dict=VLM.data['config_overrides'], log=boot_log,",
        "    )",
        "    CONFIG_PATH = write_effective_config(data=data, out_path=PROJECT.active_config_path, log=boot_log)",
    ),
    code(
        "# 5c) Run the FULL pipeline with checkpoint/resume + per-stage retries.",
        "#      Re-running this cell after a disconnect resumes automatically.",
        "from colab.stage_runner import run_stages, FULL_PIPELINE_KEYS, COLAB_SAFE_DEFAULT_KEYS",
        "STAGES = list(FULL_PIPELINE_KEYS)  # or list(COLAB_SAFE_DEFAULT_KEYS) for a quick sanity run",
        "results = run_stages(",
        "    spec_keys=STAGES, config_path=CONFIG_PATH, repo_root=REPO_ROOT,",
        "    logs_dir=PROJECT.logs_dir, reports_dir=PROJECT.reports_dir, log=boot_log,",
        "    project_root=PROJECT.project_root, run_id=PROJECT.run_id,",
        "    resume=True, max_attempts=2, stop_on_failure=True,",
        "    hf_cache_dir=PROJECT.local_hf_cache_dir, sync_manager=SYNC,",
        "    extra_kv={'question': 'Summarize available evidence and progress.'},",
        ")",
        "for r in results:",
        "    tag = 'SKIP' if r.skipped else ('OK ' if r.ok else 'FAIL')",
        "    print(f'  [{tag}] {r.key} rc={r.return_code} {r.duration_sec:.1f}s attempts={r.attempts}')",
    ),
    md(
        "## 6. Inspect progress, artifacts and build a download bundle",
    ),
    code(
        "import json",
        "# Checkpoint manifest (per-stage status, survives disconnects):",
        "state = json.loads(PROJECT.run_state_path.read_text()) if PROJECT.run_state_path.exists() else {}",
        "for key, st in (state.get('stages') or {}).items():",
        "    print(f\"  {key:28s} {st.get('status'):10s} attempts={st.get('attempts')} {st.get('duration_sec',0):.0f}s\")",
        "",
        "from colab.artifacts import collect_artifacts, build_artifact_bundle",
        "entries = collect_artifacts(PROJECT.project_root)",
        "for e in entries[:60]:",
        "    print(f'  {e.category:16s} {e.bytes/1024:8.1f} KB  {e.relative_path}')",
        "# bundle = build_artifact_bundle(project_root=PROJECT.project_root, exports_dir=PROJECT.exports_dir,",
        "#                                categories=['reports','exports_viewer','cleanup','dense','vlm_qa'])",
        "# print('bundle:', bundle)",
    ),
    md(
        "## 7. Memory & sync utilities",
    ),
    code(
        "from colab.cleanup import free_memory, disk_usage_summary",
        "print(free_memory(verbose=True))",
        "if SYNC is not None:",
        "    print('sync flush:', SYNC.flush())",
        "for path, info in disk_usage_summary([REPO_ROOT, PROJECT.project_root]).items():",
        "    print(f'  {path} -> {info}')",
    ),
    md(
        "## 8. Resume / portability / caveats",
        "",
        "- **Resume after a disconnect:** re-run cells 1–4 (or 5a/5c) with the **same**",
        "  `RUN_ID`. Stages whose outputs already exist on Drive are skipped; the run",
        "  continues from the first incomplete stage.",
        "- **Resume on another device / Drive account:** copy (or share)",
        "  `MyDrive/MyCon_Colab/projects/<RUN_ID>/` to the new Drive, set the same",
        "  `RUN_ID`, and run. The checkpoint manifest + config travel with the folder.",
        "- **Stage 5 (dense)** on a free T4 can OOM at large `dense.max_image_size`;",
        "  use `colab_safe` or set `dense.max_image_size: 800` in USER_OVERRIDES.",
        "- **Stage 8b/9** need a real BIM IFC + matched scan; otherwise they exit 0 with",
        "  a `skipped` marker and the pipeline continues.",
        "- **Real VLM** needs the Ollama provision step (5b / Tab 4). Without it the",
        "  deterministic mock answers — the pipeline never blocks on the VLM.",
        "- Free Colab disconnects after ~12 h; everything is on Drive, so just resume.",
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
