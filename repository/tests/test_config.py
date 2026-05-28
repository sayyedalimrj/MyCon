from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pipeline.common.config import ConfigError, load_config, required_config_keys
from pipeline.common.paths import input_path, output_path


def test_site01_config_loads() -> None:
    cfg = load_config(Path("configs/site01.yaml"))
    assert cfg.project_name == "site01"
    assert cfg.run_id == "2026-04-30_site01_baseline"
    assert str(cfg.project_root) == "/workspace"


def test_all_required_keys_present() -> None:
    cfg = load_config(Path("configs/site01.yaml"))
    for dotted_key in required_config_keys():
        assert cfg.require(dotted_key) is not None


def test_resolve_paths_are_root_relative() -> None:
    cfg = load_config(Path("configs/site01.yaml"))
    assert input_path(cfg, "video") == Path("/workspace/data/raw/site01.mp4")
    assert output_path(cfg, "quality_csv") == Path("/workspace/data/normalized/site01_frame_quality.csv")


def test_missing_required_key_fails_fast(tmp_path: Path) -> None:
    source = yaml.safe_load(Path("configs/site01.yaml").read_text(encoding="utf-8"))
    del source["paths"]["quality_csv"]
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ConfigError, match="paths.quality_csv"):
        load_config(cfg_path)


def test_windows_root_is_rejected(tmp_path: Path) -> None:
    source = yaml.safe_load(Path("configs/site01.yaml").read_text(encoding="utf-8"))
    source["project"]["root"] = r"C:\project"
    cfg_path = tmp_path / "bad_windows.yaml"
    cfg_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ConfigError, match="Linux/container path"):
        load_config(cfg_path)
