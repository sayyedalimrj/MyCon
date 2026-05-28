"""Render AI-readable evidence views for the Construction Copilot.

This first implementation is deliberately conservative: it creates deterministic
PNG evidence cards even when Open3D offscreen rendering is unavailable. The file
contract is stable, so the renderer can later be upgraded to true scan/BIM/overlay
rendering without changing the Copilot API.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config_access import cfg_get, resolve_path
from .context_resolver import CopilotContext

try:  # OpenCV is already part of the core image in previous stages.
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RenderedViews:
    status: str
    image_paths: dict[str, str]
    metadata_path: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)[:80]


def _write_card(path: Path, *, title: str, lines: list[str], width: int = 1280, height: int = 720) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is None:
        path.write_bytes(b"PNG_RENDERER_UNAVAILABLE")
        return
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (width - 30, height - 30), (70, 70, 70), 3)
    cv2.putText(image, title, (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (40, 40, 40), 3, cv2.LINE_AA)
    y = 160
    for line in lines[:16]:
        cv2.putText(image, line[:110], (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 35, 35), 2, cv2.LINE_AA)
        y += 38
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write rendered evidence image: {path}")


def render_views(cfg: Any, context: CopilotContext, *, question: str, requested_views: list[str]) -> RenderedViews:
    render_root = resolve_path(cfg, cfg_get(cfg, "copilot.paths.render_dir", "runs/copilot/renders"), required=True)
    assert render_root is not None
    render_root.mkdir(parents=True, exist_ok=True)

    token = _safe_name(question.lower()) or "question"
    image_paths: dict[str, str] = {}
    warnings = list(context.warnings)

    scan_lines = [
        f"Question: {question}",
        f"Point cloud: {context.pointcloud_path or 'not provided'}",
        f"View: {context.current_view}",
        f"Selected element: {context.element_global_id or 'none'}",
        f"Selected activity: {context.activity_id or 'none'}",
        "Renderer mode: evidence card placeholder; upgradeable to Open3D offscreen render.",
    ]
    bim_lines = [
        f"IFC/BIM: {context.ifc_path or 'not provided'}",
        f"Selected element: {context.element_global_id or 'none'}",
        "BIM geometry must come from deterministic IfcOpenShell/Open3D tools.",
    ]
    overlay_lines = [
        "Overlay evidence package placeholder.",
        "Use Stage 8/9 aligned scan-vs-BIM artifacts when available.",
        "VLM must not infer measurements from this image alone.",
    ]
    heatmap_lines = [
        "Deviation heatmap placeholder.",
        "Numeric deviation facts are read from metric JSON/CSV only.",
    ]

    images = {
        "scan_view": ("Scan View", scan_lines),
        "bim_view": ("BIM Reference View", bim_lines),
        "overlay_view": ("Scan/BIM Overlay", overlay_lines),
        "deviation_heatmap": ("Deviation Heatmap", heatmap_lines),
    }
    for name, (title, lines) in images.items():
        path = render_root / f"{token}_{name}.png"
        _write_card(path, title=title, lines=lines)
        image_paths[name] = path.as_posix()

    metadata = {
        "question": question,
        "requested_views": requested_views,
        "context": context.to_dict(),
        "image_paths": image_paths,
        "warnings": warnings,
    }
    metadata_path = render_root / f"{token}_render_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return RenderedViews("ok", image_paths, metadata_path.as_posix(), warnings)
