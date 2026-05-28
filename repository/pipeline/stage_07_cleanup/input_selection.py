from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_access import bool_value, input_candidates


class Stage7InputError(RuntimeError):
    """Raised when Stage 7 cannot find a valid point cloud input."""


@dataclass(frozen=True)
class SelectedInput:
    path: Path
    source: str
    reason: str
    size_bytes: int


def _source_name(path: Path) -> str:
    text = path.as_posix().lower()
    if "/da3/" in text or "da3_assisted" in text:
        return "da3_assisted"
    if "/dense/" in text or "fused" in text:
        return "dense_fused"
    return "custom"


def select_input_cloud(cfg: Any) -> SelectedInput:
    candidates = input_candidates(cfg)
    checked: list[str] = []
    for path in candidates:
        checked.append(path.as_posix())
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return SelectedInput(
                path=path,
                source=_source_name(path),
                reason="first_existing_candidate",
                size_bytes=path.stat().st_size,
            )
    if bool_value(getattr(cfg, "get", lambda k, d=None: d)("cleanup.fail_if_missing_input", True)):
        raise Stage7InputError("No valid Stage 7 input point cloud found. Checked: " + ", ".join(checked))
    raise Stage7InputError("Stage 7 input is missing and fail_if_missing_input=false is not supported for cleanup execution.")
