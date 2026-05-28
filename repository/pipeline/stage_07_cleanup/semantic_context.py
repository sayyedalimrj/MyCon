from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config_access import bool_value, cfg_get, list_value
from .io_utils import write_json_atomic


def _read_json(path: Path) -> Any | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists() or path.stat().st_size <= 0:
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def _extract_detection_classes(record: dict[str, Any]) -> list[str]:
    detections = record.get("detections", record.get("objects", []))
    classes: list[str] = []
    if isinstance(detections, list):
        for det in detections:
            if isinstance(det, dict):
                value = det.get("class", det.get("name", det.get("label", "")))
                if value:
                    classes.append(str(value).strip().lower())
            elif det:
                classes.append(str(det).strip().lower())
    return classes


def summarise_yolo_context(cfg: Any, detections_jsonl: Path, summary_json: Path) -> dict[str, Any]:
    if not bool_value(cfg_get(cfg, "cleanup.semantics_enabled", True)) or not bool_value(cfg_get(cfg, "cleanup.yolo_enabled", True)):
        summary = {"status": "disabled"}
        write_json_atomic(summary_json, summary)
        return summary
    records = _iter_jsonl(detections_jsonl)
    transient = {str(v).strip().lower() for v in list_value(cfg_get(cfg, "cleanup.yolo_transient_classes", []), [])}
    class_counter: Counter[str] = Counter()
    transient_frame_count = 0
    max_transient_count = 0
    for rec in records:
        classes = _extract_detection_classes(rec)
        class_counter.update(classes)
        transient_count = sum(1 for c in classes if c in transient)
        if transient_count > 0:
            transient_frame_count += 1
        max_transient_count = max(max_transient_count, transient_count)
    summary = {
        "status": "ok" if records else "missing",
        "detections_jsonl": detections_jsonl.as_posix(),
        "record_count": len(records),
        "class_counts": dict(sorted(class_counter.items())),
        "transient_classes": sorted(transient),
        "transient_frame_count": transient_frame_count,
        "transient_frame_ratio": float(transient_frame_count / len(records)) if records else 0.0,
        "max_transient_count_per_frame": max_transient_count,
        "note": "Stage 7 consumes optional YOLO reports only; model inference and weights remain external to the pipeline baseline.",
    }
    write_json_atomic(summary_json, summary)
    return summary


def summarise_vlm_context(cfg: Any, vlm_json: Path, summary_json: Path) -> dict[str, Any]:
    if not bool_value(cfg_get(cfg, "cleanup.semantics_enabled", True)) or not bool_value(cfg_get(cfg, "cleanup.vlm_enabled", True)):
        summary = {"status": "disabled"}
        write_json_atomic(summary_json, summary)
        return summary
    payload = _read_json(vlm_json)
    if not isinstance(payload, dict):
        summary = {
            "status": "missing",
            "vlm_scene_report_json": vlm_json.as_posix(),
            "note": "Optional VLM report was not found. Cleanup proceeds without semantic scene priors.",
        }
        write_json_atomic(summary_json, summary)
        return summary
    keys = ["site_conditions", "construction_elements", "risks", "cleanup_hints", "bim_alignment_hints", "summary"]
    summary = {"status": "ok", "vlm_scene_report_json": vlm_json.as_posix()}
    for key in keys:
        if key in payload:
            summary[key] = payload[key]
    write_json_atomic(summary_json, summary)
    return summary


def build_semantic_context(cfg: Any, paths: dict[str, Path]) -> dict[str, Any]:
    yolo = summarise_yolo_context(cfg, paths["yolo_detections_jsonl"], paths["yolo_summary_json"])
    vlm = summarise_vlm_context(cfg, paths["vlm_scene_report_json"], paths["vlm_summary_json"])
    return {"yolo": yolo, "vlm": vlm}
