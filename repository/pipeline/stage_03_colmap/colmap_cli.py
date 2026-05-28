"""COLMAP command execution utilities."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


class ColmapExecutionError(RuntimeError):
    """Raised when a COLMAP command fails."""


@dataclass(slots=True)
class CommandRecord:
    name: str
    command: list[str]
    returncode: int
    elapsed_sec: float
    stdout_tail: list[str]
    started_at_unix: float
    ended_at_unix: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ColmapRunner:
    """Run COLMAP commands with consistent logging and headless defaults."""

    def __init__(
        self,
        executable: str = "colmap",
        logger: logging.Logger | None = None,
        qt_qpa_platform: str = "offscreen",
        extra_env: dict[str, str] | None = None,
        tail_lines: int = 240,
    ) -> None:
        self.executable = executable
        self.logger = logger or logging.getLogger(__name__)
        self.qt_qpa_platform = qt_qpa_platform
        self.extra_env = extra_env or {}
        self.tail_lines = tail_lines
        self.history: list[CommandRecord] = []

    def ensure_available(self) -> Path:
        path = shutil.which(self.executable)
        if not path:
            raise ColmapExecutionError(
                f"COLMAP executable not found in PATH: {self.executable}. "
                "Verify Docker image construction-core-dev:latest before running Stage 3."
            )
        return Path(path)

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", self.qt_qpa_platform)
        env.setdefault("LC_ALL", "C")
        env.setdefault("LANG", "C")
        env.update(self.extra_env)
        return env

    def run(self, args: Sequence[str], name: str, check: bool = True) -> CommandRecord:
        self.ensure_available()
        command = [self.executable, *map(str, args)]
        self.logger.info("Running COLMAP command [%s]: %s", name, " ".join(command))
        start = time.time()
        tail: deque[str] = deque(maxlen=self.tail_lines)
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=self._env(),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            stripped = line.rstrip("\n")
            tail.append(stripped)
            if stripped:
                self.logger.info("[colmap:%s] %s", name, stripped)
        returncode = proc.wait()
        end = time.time()
        record = CommandRecord(
            name=name,
            command=command,
            returncode=returncode,
            elapsed_sec=end - start,
            stdout_tail=list(tail),
            started_at_unix=start,
            ended_at_unix=end,
        )
        self.history.append(record)
        if check and returncode != 0:
            tail_text = "\n".join(record.stdout_tail[-80:])
            raise ColmapExecutionError(
                f"COLMAP command failed [{name}] with exit code {returncode}.\n"
                f"Command: {' '.join(command)}\n"
                f"Output tail:\n{tail_text}"
            )
        self.logger.info("COLMAP command [%s] finished in %.3fs", name, record.elapsed_sec)
        return record

    def history_as_dicts(self) -> list[dict[str, object]]:
        return [item.to_dict() for item in self.history]
