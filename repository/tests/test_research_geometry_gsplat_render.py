"""CPU-only preflight tests; GPU execution deliberately NOT invoked."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.stage_04_5_cams_gs.research_gsplat_render import (
    inspect_render_command, render_from_checkpoint, validate_camera_path,
)


def _camera(viewmat=None):
    return {
        "width": 320, "height": 240, "fx": 300, "fy": 300,
        "cx": 160, "cy": 120, "view_id": "withheld_pose",
        "world_to_camera_4x4": np.eye(4).tolist() if viewmat is None else viewmat,
        "is_training_image": False,
    }


def test_camera_contract_and_no_input_or_output_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "original_ckpt.pt"
    checkpoint.write_bytes(b"fake_checkpoint")
    cameras = tmp_path / "original_camera_path.json"
    cameras.write_text(json.dumps({"frame_convention": "colmap_opencv_world_to_camera",
                                   "cameras": [_camera()]}), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in (checkpoint, cameras)}
    result = render_from_checkpoint(project_root=tmp_path, run_id="geom-render-001",
                                    checkpoint=checkpoint, camera_json=cameras)
    assert result["render_executed"] is False
    assert result["training_executed"] is False
    assert result["camera_count"] == 1
    assert result["novel_view_holdout_verified"] is False
    assert {p.name: p.read_bytes() for p in (checkpoint, cameras)} == before
    assert not (tmp_path / "runs").exists()


def test_camera_path_rejects_reflections_and_unbounded_inputs() -> None:
    with pytest.raises(ValueError, match="frame_convention"):
        validate_camera_path({"cameras": [_camera()]})
    bad = np.eye(4)
    bad[0, 0] = -1
    with pytest.raises(ValueError, match="proper rigid rotation"):
        validate_camera_path({"frame_convention": "colmap_opencv_world_to_camera",
                              "cameras": [_camera(bad.tolist())]})
    with pytest.raises(ValueError, match="1 to 60"):
        validate_camera_path({"frame_convention": "colmap_opencv_world_to_camera",
                              "cameras": []})
    with pytest.raises(ValueError, match="Invalid intrinsic"):
        validate_camera_path({"frame_convention": "colmap_opencv_world_to_camera",
                              "cameras": [{**_camera(), "fx": -1}]})


pytestmark = pytest.mark.lightweight
