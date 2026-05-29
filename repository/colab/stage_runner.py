"""Stage orchestration for the Colab pipeline (UI + notebook).

We wrap the canonical ``scripts/run_stage.py`` launcher (and the
``stage_10_copilot.run_ask`` and ``stage_11_schedule_variance.run_schedule_variance``
entry points it does not handle) so the notebook/UI can:

- Show a curated, ordered stage catalog with safe defaults *and* a full
  end-to-end pipeline ordering.
- Run one stage, a hand-picked subset, or the whole pipeline.
- Stream live stdout/stderr into a log buffer (and to a per-stage log file
  on Drive).
- Run gc + ``torch.cuda.empty_cache`` between heavy stages to keep VRAM low.
- **Resume**: skip stages that already completed (verified against their
  declared output artifacts on Drive) via :class:`colab.checkpoint.CheckpointManager`.
- **Retry**: re-run a failed stage up to ``max_attempts`` times with backoff.
- **Persist progress**: the checkpoint manifest is written atomically after
  every state transition so a Colab disconnect mid-run is fully recoverable.
- **Sync**: flush a :class:`colab.sync.DriveSyncManager` after each stage so
  local caches are mirrored onto Drive at stage boundaries.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from colab import cleanup as _cleanup
from colab.checkpoint import CheckpointManager
from colab.log_capture import LogBuffer


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


@dataclass
class StageSpec:
    """Description of an executable pipeline stage."""

    key: str  # short id, e.g. "stage_03_colmap"
    label: str  # human label for the UI
    runner: str  # "run_stage" or "module"
    target: str  # stage name for run_stage.py OR module path
    needs_video: bool = False
    needs_bim: bool = False
    needs_schedule: bool = False
    heavy: bool = False
    server_only: bool = False
    optional: bool = True
    description: str = ""
    extra_args: list[str] = field(default_factory=list)
    # Glob patterns (relative to project_root) that should exist after a
    # successful run. Used by the resume logic to confirm a stage marked "ok"
    # still has its artifacts on Drive before skipping it.
    outputs: list[str] = field(default_factory=list)
    # Stage keys this stage logically depends on (soft, informational).
    deps: list[str] = field(default_factory=list)
    # Whether a "complete" result can be reused on resume. Stage 10 (ask) is
    # a query, not a build step, so it is always re-runnable.
    resumable: bool = True


# Order matches the documented pipeline. Heavy/optional flags come from
# scripts/run_pipeline_plan.py.
STAGE_CATALOG: list[StageSpec] = [
    StageSpec(
        key="stage_01_ingest",
        label="Stage 1 — Video ingest & frame quality",
        runner="run_stage",
        target="stage_01_ingest",
        needs_video=True,
        heavy=False,
        optional=False,
        description="Normalize the uploaded video to CFR and score every frame.",
        outputs=["data/normalized/*.mp4", "data/normalized/*metadata*.json"],
    ),
    StageSpec(
        key="stage_02_keyframes",
        label="Stage 2 — Adaptive keyframe selection",
        runner="run_stage",
        target="stage_02_keyframes",
        heavy=False,
        optional=False,
        description="Select keyframes, build the manifest CSV and contact sheet.",
        outputs=["data/frames/key/*manifest*.csv", "data/frames/key/**/*.csv"],
        deps=["stage_01_ingest"],
    ),
    StageSpec(
        key="stage_03_colmap",
        label="Stage 3 — COLMAP sparse SfM (heavy)",
        runner="run_stage",
        target="stage_03_colmap",
        heavy=True,
        description="ALIKED+LightGlue (or SIFT fallback) features, sparse mapping.",
        outputs=["data/sparse/**/cameras.bin", "data/sparse/**/*.bin"],
        deps=["stage_02_keyframes"],
    ),
    StageSpec(
        key="stage_04_refinement",
        label="Stage 4 — Sparse bundle adjustment",
        runner="run_stage",
        target="stage_04_refinement",
        heavy=True,
        description="Final bundle adjustment of the sparse model.",
        outputs=["data/sparse_refined/**/*.bin"],
        deps=["stage_03_colmap"],
    ),
    StageSpec(
        key="stage_04_5_cams_gs",
        label="Stage 4.5 — CAMS-GS / 3DGS prepare (no training)",
        runner="run_stage",
        target="stage_04_5_cams_gs",
        heavy=False,
        description="Prepare Nerfstudio dataset; never trains on Colab by default.",
        outputs=["runs/**/reports/cams_gs_prepare_summary.json"],
        deps=["stage_04_refinement"],
    ),
    StageSpec(
        key="stage_05_dense",
        label="Stage 5 — COLMAP dense (very heavy, GPU strongly recommended)",
        runner="run_stage",
        target="stage_05_dense",
        heavy=True,
        description="PatchMatch stereo + fusion. Cap dense.max_image_size on Colab.",
        outputs=["data/dense/**/fused.ply"],
        deps=["stage_04_refinement"],
    ),
    StageSpec(
        key="stage_06_da3_assist",
        label="Stage 6 — DA3 depth assist (skip-safe)",
        runner="run_stage",
        target="stage_06_da3_assist",
        heavy=True,
        description="Optional DA3 depth assist. Provider=precomputed by default in Colab.",
        outputs=["runs/**/reports/da3_summary.json"],
        deps=["stage_05_dense"],
    ),
    StageSpec(
        key="stage_07_cleanup",
        label="Stage 7 — Cleanup, mesh, plane extraction",
        runner="run_stage",
        target="stage_07_cleanup",
        heavy=True,
        description="Open3D cleanup, mesh + plane extraction on the cleaned cloud.",
        outputs=["data/clean/**/*.ply", "runs/**/reports/cleanup_summary.json"],
        deps=["stage_05_dense"],
    ),
    StageSpec(
        key="stage_07_5_vlm_qa",
        label="Stage 7.5 — VLM QA (mock-safe)",
        runner="run_stage",
        target="stage_07_5_vlm_qa",
        heavy=False,
        description="Pre-BIM visual QA. Uses mock VLM unless a real one is provisioned.",
        outputs=["runs/**/reports/vlm_qa_summary.json", "data/vlm_qa/**/*.json"],
        deps=["stage_02_keyframes"],
    ),
    StageSpec(
        key="stage_07_6_viewer_export",
        label="Stage 7.6 — Viewer export package",
        runner="run_stage",
        target="stage_07_6_viewer_export",
        heavy=False,
        description="Pack viewer-friendly artifacts into exports/viewer/.",
        outputs=["exports/viewer/**/*"],
        deps=["stage_02_keyframes"],
    ),
    StageSpec(
        key="stage_07_7_cams_gs_evidence",
        label="Stage 7.7 — CAMS-GS evidence (visual only)",
        runner="run_stage",
        target="stage_07_7_cams_gs_evidence",
        heavy=False,
        description="Optional 3DGS evidence; not metric truth.",
        outputs=["runs/**/reports/cams_gs_evidence_summary.json"],
        deps=["stage_04_5_cams_gs"],
    ),
    StageSpec(
        key="stage_08_metric_alignment",
        label="Stage 8a — Metric alignment (anchors required)",
        runner="run_stage",
        target="stage_08_metric_alignment",
        heavy=False,
        server_only=True,
        description="Skipped if no metric or visual anchors are available.",
        outputs=["runs/**/reports/metric_alignment_report.json"],
        deps=["stage_07_cleanup"],
    ),
    StageSpec(
        key="stage_08_bim_registration",
        label="Stage 8b — BIM registration (heavy, server-only)",
        runner="run_stage",
        target="stage_08_bim_registration",
        needs_bim=True,
        heavy=True,
        server_only=True,
        description="Coarse + ICP scan-to-BIM registration.",
        outputs=["runs/**/reports/registration_report.json", "data/bim/aligned/**/*.json"],
        deps=["stage_07_cleanup"],
    ),
    StageSpec(
        key="stage_09_progress",
        label="Stage 9 — Progress metrics",
        runner="run_stage",
        target="stage_09_progress",
        needs_bim=True,
        heavy=False,
        server_only=True,
        description="Progress only meaningful when Stage 8 quality is defensible.",
        outputs=["data/bim/metrics/**/element_metrics.csv", "data/bim/metrics/**/*.json"],
        deps=["stage_08_bim_registration"],
    ),
    StageSpec(
        key="stage_10_copilot",
        label="Stage 10 — Copilot ask (mock-safe)",
        runner="module",
        target="pipeline.stage_10_copilot.run_ask",
        heavy=False,
        description="Ask the local Copilot a question. Requires --question (UI provides it).",
        resumable=False,
    ),
    StageSpec(
        key="stage_11_schedule_variance",
        label="Stage 11 — Schedule variance (laptop-safe)",
        runner="module",
        target="pipeline.stage_11_schedule_variance.run_schedule_variance",
        needs_schedule=True,
        heavy=False,
        description="Schedule + element metrics -> activity progress + variance.",
        outputs=["runs/**/reports/schedule_variance.json", "runs/**/reports/dashboard_summary.json"],
        deps=["stage_09_progress"],
    ),
]


STAGES_BY_KEY: dict[str, StageSpec] = {s.key: s for s in STAGE_CATALOG}

# Default selection when the user clicks "Run Colab-safe pipeline".
COLAB_SAFE_DEFAULT_KEYS: tuple[str, ...] = (
    "stage_01_ingest",
    "stage_02_keyframes",
    "stage_07_5_vlm_qa",
    "stage_07_6_viewer_export",
    "stage_11_schedule_variance",
)

# Full end-to-end reconstruction + BIM + VLM ordering. Stages that lack
# inputs (no BIM, no anchors) exit 0 with a "skipped" marker, so running the
# whole list is safe; the heavy SfM/dense stages need a GPU runtime.
FULL_PIPELINE_KEYS: tuple[str, ...] = (
    "stage_01_ingest",
    "stage_02_keyframes",
    "stage_03_colmap",
    "stage_04_refinement",
    "stage_04_5_cams_gs",
    "stage_05_dense",
    "stage_06_da3_assist",
    "stage_07_cleanup",
    "stage_07_5_vlm_qa",
    "stage_07_6_viewer_export",
    "stage_07_7_cams_gs_evidence",
    "stage_08_metric_alignment",
    "stage_08_bim_registration",
    "stage_09_progress",
    "stage_11_schedule_variance",
)


def outputs_for(key: str) -> Optional[list[str]]:
    """Return the declared output globs for ``key`` (or None if unknown)."""
    spec = STAGES_BY_KEY.get(key)
    if spec is None:
        return None
    return list(spec.outputs)


def order_stages(keys: list[str]) -> list[str]:
    """Return ``keys`` reordered to match the canonical full-pipeline order.

    Unknown keys keep their relative order at the end. This makes a hand
    picked subset run in a sensible dependency order even if the UI passed
    them in checkbox order.
    """
    rank = {k: i for i, k in enumerate(FULL_PIPELINE_KEYS)}
    extra = [k for k in keys if k not in rank]
    known = [k for k in keys if k in rank]
    known.sort(key=lambda k: rank[k])
    return known + extra


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class StageRunResult:
    key: str
    started_at: str
    finished_at: str
    duration_sec: float
    return_code: int
    ok: bool
    log_file: str
    command: list[str]
    attempts: int = 1
    skipped: bool = False
    note: str = ""


def _build_command(
    *, spec: StageSpec, config_path: Path, repo_root: Path, force: bool, log_level: str,
    extra_kv: Optional[dict[str, str]] = None,
) -> list[str]:
    config_path = Path(config_path)
    if spec.runner == "run_stage":
        cmd = [
            sys.executable,
            str(Path(repo_root) / "scripts" / "run_stage.py"),
            spec.target,
            "--config",
            str(config_path),
            "--log-level",
            log_level,
        ]
        if force:
            cmd.append("--force")
    elif spec.runner == "module":
        cmd = [
            sys.executable,
            "-m",
            spec.target,
            "--config",
            str(config_path),
            "--log-level",
            log_level,
        ]
        if force:
            cmd.append("--force")
        # Stage 10 needs a question, Stage 11 has its own special args.
        if spec.target.endswith("stage_10_copilot.run_ask"):
            question = (extra_kv or {}).get("question") or "Summarize the available evidence."
            cmd.extend(["--question", question, "--json"])
        if spec.target.endswith("stage_11_schedule_variance.run_schedule_variance"):
            # The Stage 11 runner does NOT accept --config. Build proper CLI.
            cmd = _build_stage_11_command(
                config_path=config_path,
                repo_root=repo_root,
                extra_kv=extra_kv or {},
            )
    else:
        raise ValueError(f"unknown runner type: {spec.runner}")
    cmd.extend(spec.extra_args)
    return cmd


def _build_stage_11_command(
    *, config_path: Path, repo_root: Path, extra_kv: dict[str, str]
) -> list[str]:
    """Stage 11 uses CSV/JSON paths directly. We resolve them from the config."""
    import yaml as _yaml

    with Path(config_path).open("r", encoding="utf-8") as fh:
        cfg = _yaml.safe_load(fh) or {}

    project_root = Path(cfg.get("project", {}).get("root") or Path.cwd())
    run_id = cfg.get("project", {}).get("run_id", "default")
    inputs = cfg.get("inputs", {})
    paths = cfg.get("paths", {})
    copilot_paths = (cfg.get("copilot", {}) or {}).get("paths", {})

    schedule_csv = (
        Path(extra_kv.get("schedule_csv") or inputs.get("schedule") or "")
    )
    if not schedule_csv.is_absolute():
        schedule_csv = project_root / schedule_csv

    mapping_csv = Path(
        extra_kv.get("mapping_csv")
        or copilot_paths.get("activity_progress_csv")
        or "data/bim/design/bim_schedule_mapping.csv"
    )
    if not mapping_csv.is_absolute():
        mapping_csv = project_root / mapping_csv

    element_metrics_csv = Path(
        extra_kv.get("element_metrics_csv")
        or copilot_paths.get("element_metrics_csv")
        or paths.get("metrics_dir", "data/bim/metrics") + "/element_metrics.csv"
    )
    if not element_metrics_csv.is_absolute():
        element_metrics_csv = project_root / element_metrics_csv

    out_dir = project_root / "runs" / run_id / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    activity_json = out_dir / "activity_progress.json"
    variance_json = out_dir / "schedule_variance.json"
    dashboard_json = out_dir / "dashboard_summary.json"

    data_date = extra_kv.get("data_date_utc") or _dt.date.today().isoformat()

    return [
        sys.executable,
        "-m",
        "pipeline.stage_11_schedule_variance.run_schedule_variance",
        "--schedule-csv",
        str(schedule_csv),
        "--mapping-csv",
        str(mapping_csv),
        "--element-metrics-csv",
        str(element_metrics_csv),
        "--activity-progress-json",
        str(activity_json),
        "--schedule-variance-json",
        str(variance_json),
        "--dashboard-summary-json",
        str(dashboard_json),
        "--data-date-utc",
        data_date,
    ]


def _stream_subprocess(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log: LogBuffer,
    log_file: Path,
    cancel_flag: Optional[threading.Event] = None,
) -> int:
    log.append(f"$ cd {cwd}")
    log.append("$ " + " ".join(command))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as sink:
        sink.write("\n" + "=" * 72 + "\n")
        sink.write(f"# {_dt.datetime.now().isoformat()}\n")
        sink.write("$ " + " ".join(command) + "\n")
        sink.flush()

        try:
            proc = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            msg = f"[error] cannot launch: {exc}"
            log.append(msg)
            sink.write(msg + "\n")
            return 127

        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                log.append(line)
                sink.write(line + "\n")
                if cancel_flag is not None and cancel_flag.is_set():
                    proc.terminate()
                    log.append("[run] cancellation requested; terminating subprocess")
                    sink.write("[run] cancellation requested; terminating subprocess\n")
                    break
        except Exception as exc:
            log.append(f"[error] log streaming exception: {exc}")
        proc.wait()
        return int(proc.returncode or 0)


def _make_env(repo_root: Path, *, hf_cache_dir: Optional[Path] = None) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(Path(repo_root).resolve())
    env["PYTHONPATH"] = (
        pythonpath + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else pythonpath
    )
    # COLMAP needs an offscreen Qt platform on Colab.
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    if hf_cache_dir is not None:
        env["HF_HOME"] = str(hf_cache_dir)
        env["HUGGINGFACE_HUB_CACHE"] = str(Path(hf_cache_dir) / "hub")
    return env


def run_stage(
    *,
    spec: StageSpec,
    config_path: Path,
    repo_root: Path,
    logs_dir: Path,
    log: LogBuffer,
    force: bool = True,
    log_level: str = "INFO",
    cleanup_after: bool = True,
    extra_kv: Optional[dict[str, str]] = None,
    cancel_flag: Optional[threading.Event] = None,
    on_status: Optional[Callable[[str, str], None]] = None,
    hf_cache_dir: Optional[Path] = None,
    max_attempts: int = 1,
    retry_backoff_sec: float = 10.0,
) -> StageRunResult:
    """Run a single stage (with optional retries) and stream logs into ``log``."""
    started = _dt.datetime.now()
    log.banner(f"RUN {spec.label}")
    if on_status is not None:
        on_status(spec.key, "running")

    log_file = Path(logs_dir) / f"{spec.key}.log"
    cmd = _build_command(
        spec=spec,
        config_path=config_path,
        repo_root=Path(repo_root),
        force=force,
        log_level=log_level,
        extra_kv=extra_kv,
    )
    env = _make_env(repo_root, hf_cache_dir=hf_cache_dir)

    rc = 1
    attempts = 0
    for attempt in range(1, max(1, max_attempts) + 1):
        attempts = attempt
        if attempt > 1:
            log.append(
                f"[run] retry {attempt}/{max_attempts} for {spec.key} "
                f"after {retry_backoff_sec:.0f}s backoff"
            )
            time.sleep(retry_backoff_sec)
        rc = _stream_subprocess(
            command=cmd,
            cwd=Path(repo_root),
            env=env,
            log=log,
            log_file=log_file,
            cancel_flag=cancel_flag,
        )
        if rc == 0:
            break
        if cancel_flag is not None and cancel_flag.is_set():
            break

    finished = _dt.datetime.now()
    if cleanup_after:
        summary = _cleanup.free_memory()
        log.append(f"[cleanup] {summary}")

    ok = rc == 0
    log.append(f"[run] {spec.key} rc={rc} ok={ok} attempts={attempts}")
    if on_status is not None:
        on_status(spec.key, "ok" if ok else "fail")

    return StageRunResult(
        key=spec.key,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        duration_sec=(finished - started).total_seconds(),
        return_code=rc,
        ok=ok,
        log_file=str(log_file),
        command=cmd,
        attempts=attempts,
    )


def run_stages(
    *,
    spec_keys: list[str],
    config_path: Path,
    repo_root: Path,
    logs_dir: Path,
    reports_dir: Path,
    log: LogBuffer,
    force: bool = True,
    log_level: str = "INFO",
    extra_kv: Optional[dict[str, str]] = None,
    stop_on_failure: bool = True,
    cancel_flag: Optional[threading.Event] = None,
    on_status: Optional[Callable[[str, str], None]] = None,
    # --- production additions ---
    project_root: Optional[Path] = None,
    run_id: Optional[str] = None,
    checkpoint: Optional[CheckpointManager] = None,
    resume: bool = True,
    max_attempts: int = 1,
    retry_backoff_sec: float = 10.0,
    reorder: bool = True,
    hf_cache_dir: Optional[Path] = None,
    sync_manager: object = None,
) -> list[StageRunResult]:
    """Run multiple stages in order, with checkpoint/resume + retries.

    When ``resume`` is True and a :class:`CheckpointManager` is available
    (either passed in or constructed from ``project_root``/``run_id``), stages
    whose recorded status is terminal-ok *and* whose declared output artifacts
    still exist are skipped. Progress is persisted after every transition, and
    a registered :class:`colab.sync.DriveSyncManager` is flushed after each
    stage so local caches reach Drive at stage boundaries.
    """
    results: list[StageRunResult] = []

    # Resolve a checkpoint manager.
    mgr = checkpoint
    if mgr is None and project_root is not None and run_id is not None:
        try:
            mgr = CheckpointManager(
                project_root=Path(project_root),
                run_id=run_id,
                state_path=Path(reports_dir) / "run_state.json",
                config_path=Path(config_path),
            )
        except Exception as exc:  # never let checkpointing break a run
            log.append(f"[run] checkpoint init failed ({exc}); continuing without resume")
            mgr = None

    ordered_keys = order_stages(list(spec_keys)) if reorder else list(spec_keys)

    # Compute resume plan.
    if mgr is not None and resume:
        plan = mgr.plan_resume(ordered_keys, outputs_for=outputs_for, resume=True)
        log.append(plan.describe())
        for key in plan.to_skip:
            if on_status is not None:
                on_status(key, "skipped(complete)")
    else:
        plan = None

    # Legacy human-readable status file (superseded by run_state.json).
    legacy_status = Path(reports_dir) / "colab_run_status.json"

    for key in ordered_keys:
        spec = STAGES_BY_KEY.get(key)
        if spec is None:
            log.append(f"[run] unknown stage: {key} (skipping)")
            continue
        if cancel_flag is not None and cancel_flag.is_set():
            log.append("[run] cancelled before stage start")
            break

        # Skip already-complete stages (resume).
        if (
            mgr is not None
            and resume
            and spec.resumable
            and mgr.is_complete(key, output_patterns=outputs_for(key))
        ):
            log.append(f"[run] skip {key}: already complete (resume)")
            results.append(
                StageRunResult(
                    key=key,
                    started_at=_dt.datetime.now().isoformat(),
                    finished_at=_dt.datetime.now().isoformat(),
                    duration_sec=0.0,
                    return_code=0,
                    ok=True,
                    log_file="",
                    command=[],
                    skipped=True,
                    note="resume:already_complete",
                )
            )
            continue

        # Soft dependency warning.
        _warn_unmet_deps(spec, ordered_keys, mgr, log)

        if mgr is not None:
            mgr.mark_running(key)

        result = run_stage(
            spec=spec,
            config_path=config_path,
            repo_root=repo_root,
            logs_dir=logs_dir,
            log=log,
            force=force,
            log_level=log_level,
            extra_kv=extra_kv,
            cancel_flag=cancel_flag,
            on_status=on_status,
            hf_cache_dir=hf_cache_dir,
            max_attempts=max_attempts,
            retry_backoff_sec=retry_backoff_sec,
        )
        results.append(result)

        # Update checkpoint.
        if mgr is not None:
            if result.ok:
                mgr.mark_ok(
                    key,
                    duration_sec=result.duration_sec,
                    return_code=result.return_code,
                    output_patterns=outputs_for(key),
                )
            else:
                mgr.mark_failed(
                    key,
                    duration_sec=result.duration_sec,
                    return_code=result.return_code,
                    error=f"rc={result.return_code} after {result.attempts} attempt(s)",
                )

        # Flush local caches to Drive at the stage boundary.
        if sync_manager is not None:
            try:
                sync_manager.flush()
            except Exception as exc:  # pragma: no cover
                log.append(f"[run] sync flush error: {exc}")

        # Legacy checkpoint (best-effort).
        try:
            legacy_status.parent.mkdir(parents=True, exist_ok=True)
            legacy_status.write_text(
                json.dumps(
                    {
                        "config_path": str(config_path),
                        "results": [r.__dict__ for r in results],
                        "updated_at": _dt.datetime.now().isoformat(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            log.append(f"[run] failed to write legacy status file: {exc}")

        if not result.ok and stop_on_failure:
            log.append(f"[run] stopping pipeline because {key} failed")
            break

    return results


def _warn_unmet_deps(
    spec: StageSpec,
    run_keys: list[str],
    mgr: Optional[CheckpointManager],
    log: LogBuffer,
) -> None:
    """Emit a soft warning when a stage's dependencies are not satisfied."""
    if not spec.deps:
        return
    for dep in spec.deps:
        in_run = dep in run_keys
        done = mgr is not None and mgr.is_complete(dep, output_patterns=outputs_for(dep))
        if not in_run and not done:
            log.append(
                f"[run] note: {spec.key} usually follows {dep}, which is "
                "neither in this run nor already complete; the stage may "
                "skip or run with reduced inputs."
            )
