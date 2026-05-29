"""Stage controller — filters elements by construction stage.

Given the full layout (all elements, all categories) and a stage spec
from ``scene.yaml``, this module decides:

- Which elements are **visible** at a given stage.
- What **finishing** material each visible element should display.

The logic is simple: for each category defined in the stage's
``elements`` dict, if ``completion >= threshold`` the element is kept.
A completion of 0.0 hides the entire category. A completion of 1.0
shows all elements.  Fractional completions (e.g. 0.7 for partial
masonry) show a proportional subset of elements in that category.

The finishing string is attached to each kept element as
``element.metadata["finishing"]`` so the renderer / material library
can look it up.
"""

from __future__ import annotations

from typing import Any, Mapping

from .layout import Element
from .scene_spec import StageSpec


# Minimum completion for an element to be visible.
_VISIBLE_THRESHOLD = 0.01


def select_for_stage(
    stage: StageSpec,
    elements_by_category: dict[str, list[Element]],
) -> list[Element]:
    """Return the elements visible at the given stage.

    Parameters
    ----------
    stage:
        The stage spec (from scene.yaml), containing the ``elements``
        dict with ``{category: {completion, finishing}}``.
    elements_by_category:
        The full layout grouped by category (output of
        ``layout.elements_by_category``).

    Returns
    -------
    A flat list of :class:`StagedElement` instances that are visible at
    this stage.  Each element has its ``metadata["finishing"]``
    updated to the stage-specific finishing value.
    """
    kept: list[StagedElement] = []

    for category, cat_elements in elements_by_category.items():
        stage_info = stage.elements.get(category)
        if stage_info is None:
            continue

        completion = float(stage_info.get("completion", 0.0))
        finishing = str(stage_info.get("finishing", "none"))

        if completion < _VISIBLE_THRESHOLD:
            continue

        if finishing == "none":
            continue

        n_total = len(cat_elements)
        n_keep = max(1, int(round(completion * n_total)))
        n_keep = min(n_keep, n_total)

        for elem in cat_elements[:n_keep]:
            updated_meta = dict(elem.metadata)
            updated_meta["finishing"] = finishing
            updated_meta["stage_completion"] = completion
            updated_elem = Element(
                id=elem.id,
                ifc_global_id=elem.ifc_global_id,
                name=elem.name,
                category=elem.category,
                box_min=elem.box_min,
                box_max=elem.box_max,
                metadata=updated_meta,
            )
            kept.append(StagedElement(updated_elem))

    return kept


class StagedElement:
    """Backward-compatible wrapper for the old IFC builder.

    The old code expected ``s.element`` to access the underlying Element.
    Now that select_for_stage returns Elements directly, this wrapper
    just delegates attribute access.
    """

    def __init__(self, element: Element):
        self.element = element

    @property
    def category(self) -> str:
        return self.element.category

    @property
    def finishing(self) -> str:
        return self.element.metadata.get("finishing", "raw_concrete")

    @property
    def completion(self) -> float:
        return float(self.element.metadata.get("stage_completion", 1.0))

    @property
    def status(self) -> str:
        c = self.completion
        if c >= 1.0:
            return "complete"
        elif c >= 0.5:
            return "partial"
        else:
            return "started"


def kept_only(staged_elements) -> list:
    """Unwrap StagedElements or pass Elements through (backward compat).

    If the input is already a list of Elements, wraps them in
    StagedElement so old callers (ifc_builder) that do `s.element`
    still work.
    """
    if not staged_elements:
        return []
    if isinstance(staged_elements[0], StagedElement):
        return staged_elements
    # Wrap plain Elements
    return [StagedElement(e) for e in staged_elements]
