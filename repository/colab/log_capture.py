"""Thread-safe append-only log buffer used by the Gradio UI.

The Gradio app polls ``LogBuffer.text()`` on a timer to refresh its log
panel, while ``stage_runner`` writes to the buffer from a background
thread. We deliberately keep the implementation small and dependency-free
so it works on a fresh Colab kernel before any pip installs.
"""

from __future__ import annotations

import datetime as _dt
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Iterable


class LogBuffer:
    """Bounded, thread-safe text log buffer with optional file mirroring."""

    def __init__(self, max_lines: int = 4000, mirror_path: Path | None = None) -> None:
        self._max_lines = int(max_lines)
        self._lines: Deque[str] = deque(maxlen=self._max_lines)
        self._lock = threading.Lock()
        self._mirror_path = mirror_path
        if self._mirror_path is not None:
            self._mirror_path.parent.mkdir(parents=True, exist_ok=True)

    # ----- write API -----

    def append(self, line: str) -> None:
        line = line.rstrip("\n")
        if not line:
            return
        with self._lock:
            self._lines.append(line)
            if self._mirror_path is not None:
                try:
                    with self._mirror_path.open("a", encoding="utf-8") as handle:
                        handle.write(line + "\n")
                except OSError:
                    # Never let log mirroring break the pipeline.
                    pass

    def append_many(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.append(line)

    def banner(self, title: str) -> None:
        bar = "=" * 72
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.append(bar)
        self.append(f"{ts} | {title}")
        self.append(bar)

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()

    # ----- read API -----

    def text(self, tail: int | None = None) -> str:
        with self._lock:
            data = list(self._lines)
        if tail is not None and tail > 0:
            data = data[-tail:]
        return "\n".join(data)

    def __len__(self) -> int:
        with self._lock:
            return len(self._lines)
