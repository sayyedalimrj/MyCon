"""Route natural-language construction questions to evidence tools."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any


class QueryCategory(str, Enum):
    VISUAL_QUESTION = "visual_question"
    METRIC_QUESTION = "metric_question"
    BIM_QUESTION = "bim_question"
    SCHEDULE_QUESTION = "schedule_question"
    PROGRESS_QUESTION = "progress_question"
    CAPTURE_QUALITY_QUESTION = "capture_quality_question"
    GENERAL_EXPLANATION = "general_explanation"


@dataclass(frozen=True)
class RoutedQuery:
    question: str
    category: QueryCategory
    needs_visuals: bool
    needs_metrics: bool
    needs_bim: bool
    needs_schedule: bool
    needs_progress: bool
    needs_capture_quality: bool
    requested_views: list[str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        return data


_VISUAL_TERMS = {
    "see", "show", "view", "image", "visual", "overlay", "heatmap", "render", "scan", "point cloud",
    "تصویر", "دید", "نمایش", "ابرنقطه", "اورلی", "هیت مپ", "رندر",
}
_METRIC_TERMS = {
    "deviation", "coverage", "distance", "mean", "max", "metric", "threshold", "confidence", "error",
    "انحراف", "پوشش", "متریک", "میانگین", "حداکثر", "اعتماد", "خطا",
}
_BIM_TERMS = {"bim", "ifc", "element", "wall", "column", "beam", "slab", "globalid", "مدل", "دیوار", "ستون", "تیر", "المان"}
_SCHEDULE_TERMS = {"schedule", "activity", "delay", "planned", "actual", "زمان", "برنامه", "تاخیر", "فعالیت"}
_PROGRESS_TERMS = {"progress", "complete", "completion", "executed", "accept", "پیشرفت", "کامل", "اجرا", "قبول", "تحویل"}
_CAPTURE_TERMS = {"capture", "video", "reconstruct", "registration", "low confidence", "undercovered", "کیفیت", "ویدیو", "بازسازی", "ثبت"}


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _views_from_question(text: str) -> list[str]:
    views: list[str] = []
    for name in ["front", "back", "left", "right", "top"]:
        if name in text:
            views.append(name)
    if "top" not in views and ("plan" in text or "floor plan" in text or "نقشه" in text):
        views.append("top")
    if not views:
        views = ["front", "top"]
    return views


def route_query(question: str) -> RoutedQuery:
    """Classify a user question and decide what evidence should be built."""
    text = question.strip().lower()
    visual = _contains_any(text, _VISUAL_TERMS)
    metric = _contains_any(text, _METRIC_TERMS)
    bim = _contains_any(text, _BIM_TERMS)
    schedule = _contains_any(text, _SCHEDULE_TERMS)
    progress = _contains_any(text, _PROGRESS_TERMS)
    capture = _contains_any(text, _CAPTURE_TERMS)

    if schedule:
        category = QueryCategory.SCHEDULE_QUESTION
    elif progress:
        category = QueryCategory.PROGRESS_QUESTION
    elif bim:
        category = QueryCategory.BIM_QUESTION
    elif metric:
        category = QueryCategory.METRIC_QUESTION
    elif capture:
        category = QueryCategory.CAPTURE_QUALITY_QUESTION
    elif visual:
        category = QueryCategory.VISUAL_QUESTION
    else:
        category = QueryCategory.GENERAL_EXPLANATION

    needs_visuals = visual or bim or progress or metric or capture
    needs_metrics = metric or bim or progress or capture
    needs_bim = bim or progress
    needs_schedule = schedule or progress
    needs_progress = progress or schedule
    needs_capture_quality = capture or metric

    rationale = "keyword_route:" + ",".join(
        flag for flag, active in {
            "visual": visual,
            "metric": metric,
            "bim": bim,
            "schedule": schedule,
            "progress": progress,
            "capture": capture,
        }.items() if active
    )
    if rationale == "keyword_route:":
        rationale = "keyword_route:general"

    return RoutedQuery(
        question=question,
        category=category,
        needs_visuals=needs_visuals,
        needs_metrics=needs_metrics,
        needs_bim=needs_bim,
        needs_schedule=needs_schedule,
        needs_progress=needs_progress,
        needs_capture_quality=needs_capture_quality,
        requested_views=_views_from_question(text),
        rationale=rationale,
    )
