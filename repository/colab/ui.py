"""Gradio Blocks UI for the MyCon Colab pipeline.

Designed for non-experts:

- Tab 1 — *Project & Inputs*: pick a run id, mount Drive, upload a video,
  optional IFC + schedule, persist everything to Drive and write the
  effective YAML config.
- Tab 2 — *Run Pipeline*: choose specific stages or the curated
  Colab-safe subset, run them with safe defaults, watch live logs.
- Tab 3 — *Artifacts & Downloads*: list all output files, build a zip
  bundle for Drive, and offer direct downloads.
- Tab 4 — *Environment & Cleanup*: install dependencies on demand,
  free GPU memory, and inspect disk usage.

The whole UI is built lazily inside ``build_ui`` so importing this module
does not require Gradio to already be installed (the install button on
Tab 4 can install it if needed; in practice ``environment.install_python_dependencies``
is run once before the UI launches).
"""

from __future__ import annotations

import os
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from colab import artifacts as _artifacts
from colab import cleanup as _cleanup
from colab import config_manager as _config_manager
from colab import drive as _drive
from colab import environment as _environment
from colab import stage_runner as _stage_runner
from colab.log_capture import LogBuffer


@dataclass
class UIState:
    """Mutable state held in a closure across Gradio callbacks."""

    repo_root: Path
    log: LogBuffer
    project_paths: Optional[_drive.ProjectPaths] = None
    config_path: Optional[Path] = None
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    last_run_thread: Optional[threading.Thread] = None
    stage_status: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers used by callbacks
# ---------------------------------------------------------------------------


def _validation_table(results: list[_environment.StepResult]) -> list[list[str]]:
    return [["OK" if r.ok else "FAIL", r.name, r.detail] for r in results]


def _stage_choice_labels() -> list[tuple[str, str]]:
    """Return (label, key) pairs for the Gradio CheckboxGroup."""
    return [(s.label, s.key) for s in _stage_runner.STAGE_CATALOG]


def _stage_status_table(state: UIState) -> list[list[str]]:
    rows: list[list[str]] = []
    for s in _stage_runner.STAGE_CATALOG:
        st = state.stage_status.get(s.key, "-")
        rows.append([s.key, s.label, st])
    return rows


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------


def build_ui(*, repo_root: Path, log: Optional[LogBuffer] = None):
    """Construct and return a Gradio Blocks app for the MyCon pipeline."""
    import gradio as gr  # imported lazily so the package is optional at import time

    repo_root = Path(repo_root).resolve()
    log = log or LogBuffer(max_lines=4000)
    state = UIState(repo_root=repo_root, log=log)

    log.banner("Gradio UI starting")
    log.append(f"[ui] repo_root = {repo_root}")

    # ---------- Tab 1 callbacks ----------

    def cb_setup_project(
        run_id: str,
        drive_base: str,
        mount_drive_now: bool,
        apply_safe_defaults: bool,
        copy_demo_schedule: bool,
    ):
        try:
            run_id = run_id.strip() or _config_manager.default_run_id()
            if mount_drive_now:
                _drive.mount_drive(log=log)
            paths = _drive.setup_project_tree(
                run_id=run_id,
                drive_base=drive_base.strip() or _drive.DEFAULT_DRIVE_BASE,
                log=log,
            )
            state.project_paths = paths

            # Build a starter config (no inputs yet — those land in cb_save_inputs).
            data = _config_manager.build_effective_config(
                repo_root=repo_root,
                project_root=paths.project_root,
                run_id=paths.run_id,
                project_name=run_id,
                apply_safe_overrides=apply_safe_defaults,
                user_overrides_yaml=None,
                log=log,
            )
            cfg_path = _config_manager.write_effective_config(
                data=data, out_path=paths.active_config_path, log=log
            )
            state.config_path = cfg_path

            if copy_demo_schedule:
                _config_manager.make_demo_assets(
                    repo_root=repo_root, project_root=paths.project_root, log=log
                )
            summary = (
                f"project_root: {paths.project_root}\n"
                f"run_id: {paths.run_id}\n"
                f"active_config: {cfg_path}\n"
                f"uploads_dir: {paths.uploads_dir}\n"
                f"reports_dir: {paths.reports_dir}\n"
            )
            return summary, log.text(tail=400)
        except Exception as exc:
            log.append(f"[ui] setup failed: {exc}\n{traceback.format_exc()}")
            return f"ERROR: {exc}", log.text(tail=400)

    def cb_save_inputs(
        video_file,
        ifc_file,
        schedule_file,
        user_overrides_yaml: str,
        apply_safe_defaults: bool,
    ):
        try:
            if state.project_paths is None:
                return (
                    "ERROR: run 'Initialize project' on Tab 1 first.",
                    log.text(tail=400),
                )
            paths = state.project_paths
            video_path = None
            ifc_path = None
            schedule_path = None
            if video_file is not None:
                video_path = _drive.stage_upload_to_drive(
                    src=video_file.name if hasattr(video_file, "name") else video_file,
                    dest_dir=paths.uploads_dir,
                    log=log,
                )
            if ifc_file is not None:
                ifc_path = _drive.stage_upload_to_drive(
                    src=ifc_file.name if hasattr(ifc_file, "name") else ifc_file,
                    dest_dir=paths.uploads_dir,
                    log=log,
                )
            if schedule_file is not None:
                schedule_path = _drive.stage_upload_to_drive(
                    src=schedule_file.name if hasattr(schedule_file, "name") else schedule_file,
                    dest_dir=paths.uploads_dir,
                    log=log,
                )

            data = _config_manager.build_effective_config(
                repo_root=repo_root,
                project_root=paths.project_root,
                run_id=paths.run_id,
                project_name=paths.run_id,
                video_path=video_path,
                ifc_path=ifc_path,
                schedule_path=schedule_path,
                apply_safe_overrides=apply_safe_defaults,
                user_overrides_yaml=user_overrides_yaml,
                log=log,
            )
            cfg_path = _config_manager.write_effective_config(
                data=data, out_path=paths.active_config_path, log=log
            )
            state.config_path = cfg_path
            ok, detail = _config_manager.validate_effective_config(
                config_path=cfg_path, repo_root=repo_root
            )
            log.append(f"[config] validation: {ok} ({detail})")

            summary = (
                f"video: {video_path}\n"
                f"ifc:   {ifc_path}\n"
                f"schedule: {schedule_path}\n"
                f"active_config: {cfg_path}\n"
                f"validation: {'OK' if ok else 'FAIL'} - {detail}\n"
            )
            return summary, log.text(tail=400)
        except Exception as exc:
            log.append(f"[ui] save inputs failed: {exc}\n{traceback.format_exc()}")
            return f"ERROR: {exc}", log.text(tail=400)

    # ---------- Tab 2 callbacks ----------

    def _run_in_thread(spec_keys: list[str], force: bool, log_level: str, question: str):
        if state.config_path is None or state.project_paths is None:
            log.append("[run] ERROR: configure Tab 1 first")
            return
        state.cancel_flag.clear()
        state.stage_status = {k: "queued" for k in spec_keys}
        try:
            _stage_runner.run_stages(
                spec_keys=spec_keys,
                config_path=state.config_path,
                repo_root=repo_root,
                logs_dir=state.project_paths.logs_dir,
                reports_dir=state.project_paths.reports_dir,
                log=log,
                force=force,
                log_level=log_level,
                extra_kv={"question": question or "Summarize the available evidence."},
                cancel_flag=state.cancel_flag,
                on_status=lambda key, st: state.stage_status.__setitem__(key, st),
            )
        except Exception as exc:
            log.append(f"[run] uncaught exception: {exc}\n{traceback.format_exc()}")

    def cb_start_run(selected_keys, force, log_level, question):
        if not selected_keys:
            return "ERROR: select at least one stage", log.text(tail=400), _stage_status_table(state)
        if state.last_run_thread is not None and state.last_run_thread.is_alive():
            return (
                "A run is already in progress. Click 'Cancel current run' first.",
                log.text(tail=400),
                _stage_status_table(state),
            )
        thread = threading.Thread(
            target=_run_in_thread,
            args=(list(selected_keys), bool(force), str(log_level), str(question or "")),
            daemon=True,
        )
        state.last_run_thread = thread
        thread.start()
        return (
            f"Started run with {len(selected_keys)} stage(s). Check the live log below.",
            log.text(tail=400),
            _stage_status_table(state),
        )

    def cb_run_safe_default(force, log_level, question):
        return cb_start_run(list(_stage_runner.COLAB_SAFE_DEFAULT_KEYS), force, log_level, question)

    def cb_cancel_run():
        state.cancel_flag.set()
        log.append("[ui] cancellation requested")
        return "Cancellation requested. Current subprocess will be terminated.", log.text(tail=400)

    def cb_refresh_logs():
        return log.text(tail=600), _stage_status_table(state)

    # ---------- Tab 3 callbacks ----------

    def cb_list_artifacts():
        if state.project_paths is None:
            return [["-", "Configure Tab 1 first", "-"]], None
        entries = _artifacts.collect_artifacts(state.project_paths.project_root)
        rows = _artifacts.to_table_rows(entries)
        if not rows:
            rows = [["-", "(no artifacts yet)", "-"]]
        return rows, None

    def cb_build_bundle(selected_categories: list[str]):
        if state.project_paths is None:
            return None, "Configure Tab 1 first.", log.text(tail=400)
        try:
            bundle = _artifacts.build_artifact_bundle(
                project_root=state.project_paths.project_root,
                exports_dir=state.project_paths.exports_dir,
                categories=selected_categories or None,
            )
            log.append(f"[artifacts] bundle: {bundle}")
            return str(bundle), f"Bundle ready: {bundle}", log.text(tail=400)
        except Exception as exc:
            log.append(f"[artifacts] bundle failed: {exc}\n{traceback.format_exc()}")
            return None, f"ERROR: {exc}", log.text(tail=400)

    # ---------- Tab 4 callbacks ----------

    def cb_install_apt():
        result = _environment.install_apt_packages(log=log)
        return f"{result.name}: {'OK' if result.ok else 'FAIL'} ({result.detail})", log.text(tail=400)

    def cb_install_python(install_da3: bool):
        results = _environment.install_python_dependencies(
            repo_root=repo_root, install_da3=install_da3, install_ui=True, log=log
        )
        text = "\n".join(
            f"{'OK' if r.ok else 'FAIL'} {r.name}: {r.detail}" for r in results
        )
        return text, log.text(tail=400)

    def cb_validate_env():
        results = _environment.validate_environment(log=log)
        return _validation_table(results), log.text(tail=400)

    def cb_free_memory():
        summary = _cleanup.free_memory(verbose=False)
        return "\n".join(f"{k}: {v}" for k, v in summary.items()), log.text(tail=400)

    def cb_disk_usage():
        roots: list[Path] = [repo_root]
        if state.project_paths is not None:
            roots.append(state.project_paths.project_root)
        usage = _cleanup.disk_usage_summary(roots)
        rows = []
        for path, info in usage.items():
            rows.append([path, str(info.get("exists", "")), str(info.get("mb", info.get("error", "-"))), str(info.get("files", "-"))])
        return rows, log.text(tail=400)

    # ---------- Layout ----------

    with gr.Blocks(title="MyCon Pipeline — Colab", analytics_enabled=False) as ui:
        gr.Markdown(
            "# MyCon — 3D Reconstruction + BIM + VLM (Colab UI)\n"
            "Run the pipeline stage-by-stage on a Colab GPU runtime. Outputs are "
            "written directly to Google Drive so a Colab disconnect does not lose work."
        )

        with gr.Tab("1. Project & Inputs"):
            with gr.Row():
                with gr.Column():
                    run_id_in = gr.Textbox(
                        label="Run ID",
                        value=_config_manager.default_run_id(),
                        info="Used as the project subfolder on Drive and the pipeline run_id.",
                    )
                    drive_base_in = gr.Textbox(
                        label="Drive base path (under MyDrive)",
                        value=_drive.DEFAULT_DRIVE_BASE,
                        info="Output tree will live under <Drive>/" + _drive.DEFAULT_DRIVE_BASE + "/projects/<run_id>/",
                    )
                    mount_in = gr.Checkbox(label="Mount Google Drive now", value=True)
                    safe_defaults_in = gr.Checkbox(
                        label="Apply Colab-safe pipeline defaults",
                        value=True,
                        info="Caps dense.max_image_size to 1024, mocks VLM, etc.",
                    )
                    copy_demo_in = gr.Checkbox(
                        label="Copy demo schedule fixture (Stage 11 sanity run)",
                        value=False,
                    )
                    init_btn = gr.Button("Initialize project on Drive", variant="primary")
                with gr.Column():
                    project_summary = gr.Textbox(label="Project summary", lines=8, interactive=False)

            gr.Markdown("### Upload inputs")
            with gr.Row():
                video_in = gr.File(label="Video (mp4/mov/...)", file_types=[".mp4", ".mov", ".mkv", ".avi"])
                ifc_in = gr.File(label="IFC (optional)", file_types=[".ifc"])
                schedule_in = gr.File(label="Schedule CSV (optional)", file_types=[".csv"])
            user_overrides_in = gr.Code(
                label="Optional config overrides (YAML mapping)",
                language="yaml",
                value="# Example:\n# dense:\n#   max_image_size: 800\n# keyframes:\n#   max_frames_first_run: 60\n",
            )
            save_btn = gr.Button("Save inputs & write effective config", variant="primary")
            inputs_summary = gr.Textbox(label="Inputs summary", lines=6, interactive=False)

            init_btn.click(
                cb_setup_project,
                inputs=[run_id_in, drive_base_in, mount_in, safe_defaults_in, copy_demo_in],
                outputs=[project_summary, gr.Textbox(visible=False)],
            )
            save_btn.click(
                cb_save_inputs,
                inputs=[video_in, ifc_in, schedule_in, user_overrides_in, safe_defaults_in],
                outputs=[inputs_summary, gr.Textbox(visible=False)],
            )

        with gr.Tab("2. Run Pipeline"):
            stage_choices = [(s.label, s.key) for s in _stage_runner.STAGE_CATALOG]
            stage_select = gr.CheckboxGroup(
                choices=stage_choices,
                label="Stages to run (in catalog order)",
                value=list(_stage_runner.COLAB_SAFE_DEFAULT_KEYS),
            )
            with gr.Row():
                force_in = gr.Checkbox(label="Force overwrite outputs", value=True)
                log_level_in = gr.Dropdown(
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    value="INFO",
                    label="Log level",
                )
                question_in = gr.Textbox(
                    label="Stage 10 question (used only if stage_10_copilot is selected)",
                    value="Summarize the available evidence and any progress signal.",
                )
            with gr.Row():
                run_btn = gr.Button("Run selected stages", variant="primary")
                safe_run_btn = gr.Button("Run Colab-safe default subset")
                cancel_btn = gr.Button("Cancel current run", variant="stop")

            run_status = gr.Textbox(label="Run status", lines=2, interactive=False)
            stage_status_table = gr.Dataframe(
                headers=["key", "label", "status"],
                value=_stage_status_table(state),
                label="Stage status",
                interactive=False,
            )
            log_box = gr.Textbox(label="Live log (auto-refreshes)", lines=22, interactive=False)
            refresh_btn = gr.Button("Refresh logs")

            run_btn.click(
                cb_start_run,
                inputs=[stage_select, force_in, log_level_in, question_in],
                outputs=[run_status, log_box, stage_status_table],
            )
            safe_run_btn.click(
                cb_run_safe_default,
                inputs=[force_in, log_level_in, question_in],
                outputs=[run_status, log_box, stage_status_table],
            )
            cancel_btn.click(cb_cancel_run, outputs=[run_status, log_box])
            refresh_btn.click(cb_refresh_logs, outputs=[log_box, stage_status_table])

            # Live polling timer (every 3 s) so logs and status update without
            # the user clicking "Refresh".
            try:
                timer = gr.Timer(3.0)
                timer.tick(cb_refresh_logs, outputs=[log_box, stage_status_table])
            except Exception:  # pragma: no cover - older gradio without Timer
                pass

        with gr.Tab("3. Artifacts & Downloads"):
            list_btn = gr.Button("Refresh artifact list", variant="primary")
            artifact_table = gr.Dataframe(
                headers=["category", "relative_path", "size"],
                value=[["-", "(no project yet)", "-"]],
                label="Artifacts in project_root",
                interactive=False,
            )
            categories_in = gr.CheckboxGroup(
                choices=list(_artifacts.ARTIFACT_PATTERNS.keys()),
                label="Categories to include in zip bundle",
                value=["reports", "exports_viewer", "cleanup", "dense", "vlm_qa"],
            )
            bundle_btn = gr.Button("Build zip bundle on Drive")
            bundle_status = gr.Textbox(label="Bundle status", lines=2, interactive=False)
            bundle_file = gr.File(label="Latest bundle (download)", interactive=False)

            list_btn.click(cb_list_artifacts, outputs=[artifact_table, gr.Textbox(visible=False)])
            bundle_btn.click(
                cb_build_bundle,
                inputs=[categories_in],
                outputs=[bundle_file, bundle_status, gr.Textbox(visible=False)],
            )

        with gr.Tab("4. Environment & Cleanup"):
            gr.Markdown(
                "Use this tab if the notebook environment looks broken, "
                "between heavy stages to free GPU memory, or to install the "
                "DA3/transformers stack on demand."
            )
            with gr.Row():
                apt_btn = gr.Button("Install apt packages (ffmpeg, colmap, ...)")
                py_btn = gr.Button("Install Python deps (core + UI)")
                py_da3_btn = gr.Button("Install Python deps (core + UI + DA3)")
            env_text = gr.Textbox(label="Last install/validation output", lines=10, interactive=False)
            env_table = gr.Dataframe(
                headers=["status", "name", "detail"],
                value=[["-", "(not validated)", "-"]],
                label="Environment validation",
                interactive=False,
            )
            with gr.Row():
                validate_btn = gr.Button("Validate environment", variant="primary")
                free_btn = gr.Button("Free GPU/CPU memory now")
                disk_btn = gr.Button("Show disk usage")
            disk_table = gr.Dataframe(
                headers=["path", "exists", "MB", "files"],
                value=[["-", "-", "-", "-"]],
                label="Disk usage",
                interactive=False,
            )

            apt_btn.click(cb_install_apt, outputs=[env_text, gr.Textbox(visible=False)])
            py_btn.click(
                lambda: cb_install_python(False),
                outputs=[env_text, gr.Textbox(visible=False)],
            )
            py_da3_btn.click(
                lambda: cb_install_python(True),
                outputs=[env_text, gr.Textbox(visible=False)],
            )
            validate_btn.click(cb_validate_env, outputs=[env_table, gr.Textbox(visible=False)])
            free_btn.click(cb_free_memory, outputs=[env_text, gr.Textbox(visible=False)])
            disk_btn.click(cb_disk_usage, outputs=[disk_table, gr.Textbox(visible=False)])

        gr.Markdown(
            "*Tip: keep this tab open while a run is in progress. Logs are also "
            "mirrored to ``runs/<run_id>/logs/<stage>.log`` on Drive.*"
        )

    return ui


def launch_ui(*, repo_root: Path, share: bool = True, log: Optional[LogBuffer] = None):
    """Convenience entry-point used directly from the notebook."""
    ui = build_ui(repo_root=repo_root, log=log)
    ui.queue()
    return ui.launch(share=share, inline=False, prevent_thread_lock=True)
