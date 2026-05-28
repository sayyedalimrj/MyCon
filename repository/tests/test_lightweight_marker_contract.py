from __future__ import annotations

from pathlib import Path


def test_conftest_auto_marks_unclassified_tests_as_lightweight() -> None:
    text = Path("tests/conftest.py").read_text(encoding="utf-8")
    assert "pytest_collection_modifyitems" in text
    assert "item.add_marker(pytest.mark.lightweight)" in text
    assert "HEAVY_MARKERS" in text
