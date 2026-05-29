#!/usr/bin/env python3
"""Run every smoke test in this folder.

Usage::

    PYTHONPATH=examples/synthetic_floor_7stage/src \\
        python3 examples/synthetic_floor_7stage/tests/run_all.py

Designed to be runnable in CI without external services. Tests that
need optional dependencies (Pillow, imageio) skip themselves
gracefully when the deps are missing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "src"))

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(here), pattern="test_*.py", top_level_dir=str(here))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
