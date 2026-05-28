"""Video and image-sequence exporter.

We use ``imageio`` with the bundled ``imageio-ffmpeg`` binary to encode
H.264 MP4. Each stage produces:

* ``stage_<id>.mp4`` - the final smartphone-style video.
* ``stage_<id>_clean.mp4`` - the renderer output before smartphone
  post-processing (useful for ablation / debugging).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import numpy as np


def write_mp4(frames: Iterable[np.ndarray], out_path: Path, fps: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.imwrite(
        out_path,
        np.stack(list(frames), axis=0),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    return out_path


def write_frames_dir(frames: Iterable[np.ndarray], out_dir: Path, prefix: str = "frame") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, fr in enumerate(frames):
        p = out_dir / f"{prefix}_{i:05d}.png"
        iio.imwrite(p, fr)
        written.append(p)
    return written
