"""Simple inline progress bar (no external dependencies).

Usage::

    from synthetic_floor.progress import ProgressBar

    with ProgressBar(total=100, label="rendering") as pb:
        for i in range(100):
            do_work()
            pb.update(1)

This prints something like::

    rendering: [====================] 100/100 (100.0%) 12.3s

If stdout is not a TTY (e.g. piped to a log file) it falls back to
printing a progress line every 10% so logs stay readable.
"""

from __future__ import annotations

import sys
import time


class ProgressBar:
    """Minimal progress bar that works in Colab, terminals, and log files."""

    def __init__(self, total: int, label: str = "", width: int = 30):
        self.total = max(1, total)
        self.label = label
        self.width = width
        self.current = 0
        self._start = time.time()
        self._last_print_pct = -1
        self._is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def update(self, n: int = 1) -> None:
        self.current = min(self.total, self.current + n)
        pct = int(100 * self.current / self.total)
        elapsed = time.time() - self._start

        if self._is_tty:
            # Inline overwrite on terminals / Colab notebook
            filled = int(self.width * self.current / self.total)
            bar = "=" * filled + "-" * (self.width - filled)
            line = f"\r{self.label}: [{bar}] {self.current}/{self.total} ({pct}%) {elapsed:.1f}s"
            sys.stdout.write(line)
            sys.stdout.flush()
            if self.current >= self.total:
                sys.stdout.write("\n")
                sys.stdout.flush()
        else:
            # Log-file mode: print every 10% (or at 100%)
            decile = pct // 10
            if decile > self._last_print_pct or self.current >= self.total:
                self._last_print_pct = decile
                print(f"{self.label}: {self.current}/{self.total} ({pct}%) {elapsed:.1f}s", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        # Ensure we print a final newline if TTY and not already done
        if self._is_tty and self.current < self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()
