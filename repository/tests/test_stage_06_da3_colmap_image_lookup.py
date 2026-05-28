from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.stage_06_da3_assist.colmap_model import ColmapTextModel, ImagePose, image_by_name


def test_image_by_name_shared_helper_removes_duplicate_private_helpers() -> None:
    pose = ImagePose(
        image_id=1,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0]),
        tvec=np.asarray([0.0, 0.0, 0.0]),
        camera_id=1,
        name="frame.jpg",
        xys=np.zeros((0, 2)),
        point3d_ids=np.zeros((0,), dtype=int),
    )
    model = ColmapTextModel(cameras={}, images={1: pose}, points3d={})
    assert image_by_name(model)["frame.jpg"] is pose

    assert "def _image_by_name" not in Path("pipeline/stage_06_da3_assist/depth_fusion.py").read_text(encoding="utf-8")
    assert "def _image_by_name" not in Path("pipeline/stage_06_da3_assist/depth_alignment.py").read_text(encoding="utf-8")
