"""Per-stage element selection.

For each stage we map the configured ``completion`` ratio to a concrete
deterministic *subset* of elements. The selection rule for each
category is simple and reproducible:

    keep round(completion * len(elements)) elements,
    in their natural order.

Because the natural order is itself deterministic, the same subset is
produced every time. This module also attaches the configured
``finishing`` string to every kept element so the renderer/IFC builder
can pick the right material.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .layout import CATEGORIES, Element
from .scene_spec import StageSpec


@dataclass(frozen=True)
class StagedElement:
    element: Element
    completion: float
    finishing: str
    status: str  # "likely_completed" | "partially_observed" | "not_evidenced"


def _status_for_completion(c: float) -> str:
    """Map a completion ratio to the canonical Stage 9 status string.

    The mapping deliberately mirrors the one in
    ``pipeline/stage_11_schedule_variance/activity_rollup.py``::

        likely_completed   = fully built
        partially_observed = partly built / scaffolding visible
        not_evidenced      = not built or not visible
    """
    if c >= 0.9999:
        return "likely_completed"
    if c >= 0.25:
        return "partially_observed"
    return "not_evidenced"


def select_for_stage(
    stage: StageSpec,
    elements_by_cat: dict[str, list[Element]],
) -> list[StagedElement]:
    """Return the deterministic set of elements present at this stage."""
    out: list[StagedElement] = []
    for cat in CATEGORIES:
        info = stage.elements.get(cat, {"completion": 0.0, "finishing": "raw_concrete"})
        completion = float(info.get("completion", 0.0))
        finishing = str(info.get("finishing", "raw_concrete"))
        bucket = elements_by_cat.get(cat, [])
        n_keep = int(round(completion * len(bucket)))
        # Stable selection: take the first n_keep elements in the order
        # they appear in the layout (which is itself deterministic).
        kept = bucket[:n_keep]
        # The "status" we emit depends on whether THIS specific element
        # is fully built (kept) or not (skipped). We still want to
        # report the un-kept elements in element_metrics so the rest of
        # the pipeline can compute progress; we mark them not_evidenced
        # when kept = 0 of N, partially_observed when 0 < kept < N, and
        # likely_completed when kept = N. Per-element granularity is
        # better than per-category granularity for the rest of the
        # pipeline, so individual kept elements get likely_completed
        # and individual skipped elements get not_evidenced.
        for e in kept:
            out.append(StagedElement(
                element=e,
                completion=1.0,
                finishing=finishing,
                status="likely_completed",
            ))
        # If the category is partly built (0 < c < 1) then the *missing*
        # half is reported as partially_observed (some scaffolding is
        # visible on site). If the category is 0% complete the missing
        # elements are reported as not_evidenced.
        per_elem_missing_status = (
            "partially_observed" if 0.0 < completion < 1.0 else "not_evidenced"
        )
        for e in bucket[n_keep:]:
            out.append(StagedElement(
                element=e,
                completion=0.0,
                finishing=finishing,
                status=per_elem_missing_status,
            ))
    return out


def kept_only(staged: Iterable[StagedElement]) -> list[StagedElement]:
    """Return only the elements that are physically present at this stage."""
    return [s for s in staged if s.completion >= 0.5]
