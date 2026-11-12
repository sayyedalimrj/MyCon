"""Execute notebook cells with stubbed external services to detect Run-All NameErrors.

This is a CPU/stdlib simulation: actual Colab Drive, GPU, pip, network and
external binaries are deliberately replaced.  It MUST NOT claim field readiness.
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest
from colab._build_notebook import NOTEBOOK

pytestmark = pytest.mark.lightweight
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _source(i):
    return "".join(NOTEBOOK["cells"][i]["source"])


@pytest.mark.parametrize("mode", ["ui", "headless"])
def test_run_all_control_flow_without_network_gpu_or_drive(monkeypatch, tmp_path, mode):
    import colab.drive as drive
    import colab.environment as env
    import colab.sync as sync
    import colab.ui as ui
    import colab.stage_runner as stage_runner
    import colab.cleanup as cleanup

    # Keep ALL writes within pytest's temporary folder.
    original_setup = drive.setup_project_tree

    def fake_setup(**kwargs):
        d = original_setup(
            run_id=kwargs["run_id"], auto_mount=False,
            drive_mount=tmp_path / "no-google-drive",
            fallback_root=tmp_path / "drive_project",
            local_scratch=tmp_path / "cache",
        )
        d.on_drive = True  # pretend our temp project is a confirmed Drive
        return d

    monkeypatch.setattr(drive, "setup_project_tree", fake_setup)
    monkeypatch.setattr(env, "bootstrap_environment", lambda *a, **kw: {
        "ok": True, "gpu": {"available": "simulated"},
        "python": [{"name": "mock-install", "ok": True, "detail": "no network"}]})
    monkeypatch.setattr(env, "is_colab", lambda: True)
    monkeypatch.setattr(cleanup, "free_memory", lambda **kw: {"status": "mock"})
    monkeypatch.setattr(cleanup, "disk_usage_summary", lambda ps: {})
    monkeypatch.setattr(stage_runner, "run_stages",
                        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Unrequested heavy stage started")))

    launch = []
    class FakeUI:
        def queue(self): return self
        def launch(self, **kwargs):
            launch.append(kwargs)
            return ("mock app", "mock url")
    monkeypatch.setattr(ui, "build_ui", lambda **kwargs: FakeUI())

    class FakeSync:
        def __init__(self, **kwargs): pass
        def set_remount_callback(self, f): pass
        def register(self, *args): pass
        def start_periodic(self): pass
        def flush(self): return "mock"
    monkeypatch.setattr(sync, "DriveSyncManager", FakeSync)

    ns = {"__name__": "__main__", "REPO_ROOT": ROOT,
          "os": os, "sys": sys, "pathlib": pathlib}
    cells = [2, 6, 8, 10, 12, 13, 15, 17, 19]
    for i in cells:
        source = _source(i)
        if i == 8:
            source = source.replace("EXECUTION_MODE = 'ui'", f"EXECUTION_MODE = {mode!r}")
        exec(compile(source, f"notebook_cell_{i}", "exec"), ns)

    assert ns["RUN_HEADLESS_PIPELINE"] is False
    assert ns["ENABLE_REAL_VLM"] is False
    if mode == "ui":
        assert len(launch) == 1
        assert launch[0]["share"] is True
        username, password = launch[0]["auth"]
        assert username == "mycon" and len(password) > 20
        assert not ns["PROJECT"].active_config_path.exists()
    else:
        assert not launch
        assert ns["CONFIG_PATH"].exists()
        import yaml
        cfg = yaml.safe_load(ns["CONFIG_PATH"].read_text())
        assert cfg["cleanup"]["comparison_ready_run_id"] == "geom-ready-mycon_site_01"
        assert cfg["cleanup"]["comparison_ready_enabled"] is True


def test_colab_gradio_ui_builds_with_installed_version():
    gradio = pytest.importorskip("gradio")
    from colab.log_capture import LogBuffer
    from colab.ui import build_ui
    app = build_ui(repo_root=ROOT, log=LogBuffer(max_lines=50))
    assert app is not None
