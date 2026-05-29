"""Tests for the 3 detailed example scenes (office / loft / warehouse).

Verifies the richer geometry (foundation, concrete beams, exterior site,
window frames, final floor finish) and — crucially — that the construction
sequence is *logical*: doors and windows are only installed late, the final
floor finish appears only at the end, etc.

IFC checks run only when ifcopenshell is importable (it is on Colab / in the
Docker images); otherwise they are skipped so the suite stays laptop-safe.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CFG = HERE.parent / "config"
sys.path.insert(0, str(SRC))

from synthetic_floor.scene_spec import load_scene_spec  # noqa: E402
from synthetic_floor.layout import build_layout, elements_by_category  # noqa: E402
from synthetic_floor.stage_controller import select_for_stage, kept_only  # noqa: E402

CONFIGS = ["scene_office.yaml", "scene_loft.yaml", "scene_warehouse.yaml"]


def _cats_at_stage(spec, by_cat, stage_id):
    staged = select_for_stage(spec.stages[stage_id - 1], by_cat)
    return {s.element.category for s in kept_only(staged)}


class TestDetailedExamples(unittest.TestCase):
    def test_all_configs_load_with_seven_stages(self):
        for cfg in CONFIGS:
            spec = load_scene_spec(CFG / cfg)
            self.assertEqual(len(spec.stages), 7, cfg)

    def test_detailed_categories_present(self):
        for cfg in CONFIGS:
            spec = load_scene_spec(CFG / cfg)
            cats = set(elements_by_category(build_layout(spec)))
            for need in ("site_ground", "foundation", "beams", "window_frame", "floor_finish"):
                self.assertIn(need, cats, f"{cfg}: missing {need}")

    def test_construction_sequence_is_logical(self):
        for cfg in CONFIGS:
            spec = load_scene_spec(CFG / cfg)
            by = elements_by_category(build_layout(spec))
            c1 = _cats_at_stage(spec, by, 1)
            c3 = _cats_at_stage(spec, by, 3)
            c4 = _cats_at_stage(spec, by, 4)
            c5 = _cats_at_stage(spec, by, 5)
            c6 = _cats_at_stage(spec, by, 6)
            c7 = _cats_at_stage(spec, by, 7)

            # Stage 1: only earthwork + foundation (no superstructure).
            self.assertIn("foundation", c1, cfg)
            self.assertIn("site_ground", c1, cfg)
            for absent in ("slab", "columns", "beams", "north_wall", "windows", "door"):
                self.assertNotIn(absent, c1, f"{cfg}: {absent} should not exist at stage 1")

            # Beams (the concrete frame) appear by stage 3.
            self.assertIn("beams", c3, cfg)

            # Walls by stage 4, but doors/windows are NOT installed yet.
            self.assertIn("north_wall", c4, cfg)
            for absent in ("windows", "window_frame", "door"):
                self.assertNotIn(absent, c4, f"{cfg}: {absent} installed too early (stage 4)")
            for absent in ("windows", "window_frame", "door"):
                self.assertNotIn(absent, c5, f"{cfg}: {absent} installed too early (stage 5)")

            # Doors + windows (+ frames) first appear together at stage 6.
            for present in ("windows", "window_frame", "door"):
                self.assertIn(present, c6, f"{cfg}: {present} missing at stage 6")

            # The final (different) floor finish appears only at the very end.
            self.assertNotIn("floor_finish", c6, cfg)
            self.assertIn("floor_finish", c7, cfg)
            self.assertIn("ceiling_lights", c7, cfg)

    def test_kept_elements_monotonic_until_finishes(self):
        # The number of placed elements should grow as the building goes up.
        for cfg in CONFIGS:
            spec = load_scene_spec(CFG / cfg)
            by = elements_by_category(build_layout(spec))
            counts = [
                len(kept_only(select_for_stage(spec.stages[i], by)))
                for i in range(7)
            ]
            # Stages 1..6 are strictly increasing (substructure -> frame ->
            # walls -> services -> openings).
            for a, b in zip(counts[:5], counts[1:6]):
                self.assertLess(a, b, f"{cfg}: counts not increasing: {counts}")

    def test_ifc_classes_when_ifcopenshell_available(self):
        try:
            import ifcopenshell  # noqa: F401
            import scipy  # noqa: F401  (trimesh OBJ export dependency)
        except Exception:
            self.skipTest("ifcopenshell/scipy not installed")
        import tempfile
        from dataclasses import replace
        from synthetic_floor.scene_spec import OutputPaths
        from synthetic_floor.ifc_builder import write_stage_ifc

        for cfg in CONFIGS:
            spec = load_scene_spec(CFG / cfg)
            by = elements_by_category(build_layout(spec))
            tmp = Path(tempfile.mkdtemp())
            out = OutputPaths(root=tmp, bim=tmp, mesh=tmp, renders=tmp, video=tmp,
                              depth=tmp, segmentation=tmp, camera=tmp, manifests=tmp, logs=tmp)
            spec2 = replace(spec, output=out)
            staged7 = select_for_stage(spec.stages[6], by)
            ifc_path = write_stage_ifc(spec2, 7, staged7, tmp / "s7.ifc")
            model = ifcopenshell.open(str(ifc_path))
            self.assertEqual(model.schema, "IFC4", cfg)
            self.assertGreater(len(model.by_type("IfcBeam")), 0, cfg)
            self.assertGreater(len(model.by_type("IfcColumn")), 0, cfg)
            self.assertGreater(len(model.by_type("IfcWindow")), 0, cfg)
            self.assertGreater(len(model.by_type("IfcDoor")), 0, cfg)
            self.assertGreater(len(model.by_type("IfcFooting")), 0, cfg)


if __name__ == "__main__":
    unittest.main()
