from __future__ import annotations


HEAVY_MARKERS = {"geometry", "server", "vlm", "colmap", "slow"}


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    import pytest

    for item in items:
        existing = {mark.name for mark in item.iter_markers()}

        if "lightweight" in existing:
            continue

        if existing.intersection(HEAVY_MARKERS):
            continue

        item.add_marker(pytest.mark.lightweight)
