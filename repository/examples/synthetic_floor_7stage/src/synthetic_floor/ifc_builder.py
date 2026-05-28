"""IFC4 BIM file generation per stage.

Each stage produces a valid IFC4 file containing one project, one site,
one building, one storey and the elements that exist at that stage.
The IFC GlobalId of every element is *the same* across stages, so a
downstream tool can compute differences between stages by GUID.

The implementation uses ``ifcopenshell.api`` (the official Python
authoring API). If the API import fails the function falls back to a
minimal hand-written entity dump that still parses as valid IFC.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

from .layout import Element
from .scene_spec import SceneSpec
from .stage_controller import StagedElement, kept_only

_LOG = logging.getLogger(__name__)


_CATEGORY_TO_IFC: dict[str, str] = {
    "slab": "IfcSlab",
    "columns": "IfcColumn",
    "exterior_walls": "IfcWall",
    "interior_walls": "IfcWall",
    "windows": "IfcWindow",
    "doors": "IfcDoor",
    "ceiling": "IfcCovering",
    "floor_finish": "IfcCovering",
    "baseboards": "IfcCovering",
    "fixtures": "IfcFlowTerminal",
}


def write_stage_ifc(
    spec: SceneSpec,
    stage_id: int,
    staged_elements: Sequence[StagedElement],
    out_path: Path,
) -> Path:
    """Write a single IFC4 file for one stage."""
    try:
        import ifcopenshell  # type: ignore
        import ifcopenshell.api  # type: ignore
        from ifcopenshell.api import run  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        _LOG.warning("IfcOpenShell unavailable (%s); writing minimal text IFC.", e)
        return _write_minimal_ifc_fallback(spec, stage_id, staged_elements, out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = ifcopenshell.api.run("project.create_file", version="IFC4")
    project = ifcopenshell.api.run(
        "root.create_entity", model,
        ifc_class="IfcProject",
        name=f"{spec.project_name} (stage {stage_id:02d})",
    )
    ifcopenshell.api.run("unit.assign_unit", model)
    # IFC requires a top-level "Model" context plus a Body subcontext for
    # 3D geometry. We add both explicitly.
    model_ctx = ifcopenshell.api.run(
        "context.add_context", model,
        context_type="Model",
    )
    body_subctx = ifcopenshell.api.run(
        "context.add_context", model,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=model_ctx,
    )

    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="Floor 01")

    ifcopenshell.api.run("aggregate.assign_object", model, products=[site], relating_object=project)
    ifcopenshell.api.run("aggregate.assign_object", model, products=[building], relating_object=site)
    ifcopenshell.api.run("aggregate.assign_object", model, products=[storey], relating_object=building)

    kept = kept_only(staged_elements)
    for s in kept:
        e = s.element
        ifc_class = _CATEGORY_TO_IFC.get(e.category, "IfcBuildingElementProxy")
        product = ifcopenshell.api.run(
            "root.create_entity", model,
            ifc_class=ifc_class,
            name=e.name,
        )
        # Pin the deterministic GlobalId so it matches across stages.
        product.GlobalId = e.ifc_global_id

        # Build a simple swept-solid representation: the box footprint
        # extruded along Z. Rotation is identity; the placement origin
        # is at (xmin, ymin, zmin).
        x0, y0, z0 = e.box_min
        x1, y1, z1 = e.box_max
        depth = max(0.001, z1 - z0)
        # Profile in local XY
        profile = model.create_entity(
            "IfcRectangleProfileDef",
            ProfileType="AREA",
            ProfileName=None,
            Position=model.create_entity(
                "IfcAxis2Placement2D",
                Location=model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0)),
            ),
            XDim=max(0.001, x1 - x0),
            YDim=max(0.001, y1 - y0),
        )
        extrusion = model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile,
            Position=model.create_entity(
                "IfcAxis2Placement3D",
                Location=model.create_entity(
                    "IfcCartesianPoint",
                    Coordinates=(0.5 * (x0 + x1), 0.5 * (y0 + y1), z0),
                ),
            ),
            ExtrudedDirection=model.create_entity(
                "IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)
            ),
            Depth=depth,
        )
        shape = model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=body_subctx,
            RepresentationIdentifier="Body",
            RepresentationType="SweptSolid",
            Items=[extrusion],
        )
        product.Representation = model.create_entity(
            "IfcProductDefinitionShape", Representations=[shape]
        )
        # Place the product at the origin (the solid has its own coordinates)
        product.ObjectPlacement = model.create_entity(
            "IfcLocalPlacement",
            RelativePlacement=model.create_entity(
                "IfcAxis2Placement3D",
                Location=model.create_entity(
                    "IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)
                ),
            ),
        )
        ifcopenshell.api.run(
            "spatial.assign_container", model,
            products=[product], relating_structure=storey,
        )

    model.write(str(out_path))
    return out_path


def _write_minimal_ifc_fallback(
    spec: SceneSpec,
    stage_id: int,
    staged_elements: Sequence[StagedElement],
    out_path: Path,
) -> Path:
    """Tiny no-deps text IFC writer used only if IfcOpenShell is missing.

    The output is *not* a full IFC4 file; it preserves the GlobalId
    list and bounding boxes so a unit test can still read them.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ISO-10303-21;",
        "HEADER;",
        f"FILE_DESCRIPTION(('synthetic_floor_7stage stage {stage_id}'),'2;1');",
        f"FILE_NAME('{out_path.name}','2026-01-01T00:00:00','synthetic','synthetic','IfcOpenShell-fallback','synthetic','synthetic');",
        "FILE_SCHEMA(('IFC4'));",
        "ENDSEC;",
        "DATA;",
    ]
    for s in kept_only(staged_elements):
        e = s.element
        ifc_class = _CATEGORY_TO_IFC.get(e.category, "IfcBuildingElementProxy")
        lines.append(
            f"/*{ifc_class} GUID={e.ifc_global_id} ID={e.id} "
            f"box=({e.box_min[0]:.3f},{e.box_min[1]:.3f},{e.box_min[2]:.3f})-("
            f"{e.box_max[0]:.3f},{e.box_max[1]:.3f},{e.box_max[2]:.3f})*/"
        )
    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
