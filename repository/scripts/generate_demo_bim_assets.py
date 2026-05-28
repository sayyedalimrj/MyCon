from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import ifcopenshell
    import ifcopenshell.guid
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "IfcOpenShell is required to generate the demo IFC. "
        "Run this script inside the core Docker container."
    ) from exc


@dataclass
class DemoElement:
    global_id: str
    name: str
    ifc_type: str
    activity_id: str
    quantity_type: str
    quantity_value: float
    weight: float


def gid() -> str:
    return ifcopenshell.guid.new()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def p3(f: Any, xyz: tuple[float, float, float]) -> Any:
    return f.create_entity("IfcCartesianPoint", Coordinates=[float(xyz[0]), float(xyz[1]), float(xyz[2])])


def p2(f: Any, xy: tuple[float, float]) -> Any:
    return f.create_entity("IfcCartesianPoint", Coordinates=[float(xy[0]), float(xy[1])])


def direction(f: Any, values: tuple[float, float, float]) -> Any:
    return f.create_entity("IfcDirection", DirectionRatios=[float(values[0]), float(values[1]), float(values[2])])


def axis2placement3d(
    f: Any,
    location: tuple[float, float, float],
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ref_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> Any:
    return f.create_entity(
        "IfcAxis2Placement3D",
        Location=p3(f, location),
        Axis=direction(f, axis),
        RefDirection=direction(f, ref_direction),
    )


def axis2placement2d(f: Any) -> Any:
    return f.create_entity(
        "IfcAxis2Placement2D",
        Location=p2(f, (0.0, 0.0)),
        RefDirection=f.create_entity("IfcDirection", DirectionRatios=[1.0, 0.0]),
    )


def local_placement(
    f: Any,
    relative_to: Any,
    location: tuple[float, float, float],
    angle_rad: float = 0.0,
) -> Any:
    ref = (math.cos(angle_rad), math.sin(angle_rad), 0.0)
    return f.create_entity(
        "IfcLocalPlacement",
        PlacementRelTo=relative_to,
        RelativePlacement=axis2placement3d(f, location, ref_direction=ref),
    )


def make_rectangular_prism(
    f: Any,
    context: Any,
    spatial_placement: Any,
    ifc_type: str,
    name: str,
    x_dim: float,
    y_dim: float,
    z_dim: float,
    base_center: tuple[float, float, float],
    angle_rad: float,
    predefined_type: str | None,
) -> tuple[Any, DemoElement]:
    profile = f.create_entity(
        "IfcRectangleProfileDef",
        ProfileType="AREA",
        ProfileName=f"{name}_profile",
        Position=axis2placement2d(f),
        XDim=float(x_dim),
        YDim=float(y_dim),
    )

    solid = f.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=axis2placement3d(f, (0.0, 0.0, 0.0)),
        ExtrudedDirection=direction(f, (0.0, 0.0, 1.0)),
        Depth=float(z_dim),
    )

    shape_rep = f.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[solid],
    )

    product_shape = f.create_entity(
        "IfcProductDefinitionShape",
        Name=f"{name}_shape",
        Representations=[shape_rep],
    )

    placement = local_placement(f, spatial_placement, base_center, angle_rad=angle_rad)
    element_gid = gid()

    kwargs = {
        "GlobalId": element_gid,
        "Name": name,
        "ObjectPlacement": placement,
        "Representation": product_shape,
    }
    if predefined_type is not None:
        kwargs["PredefinedType"] = predefined_type

    element = f.create_entity(ifc_type, **kwargs)

    if ifc_type == "IfcSlab":
        quantity_type = "area_m2"
        quantity_value = x_dim * y_dim
        activity_id = "A100"
    elif ifc_type == "IfcWall":
        quantity_type = "area_m2"
        quantity_value = x_dim * z_dim
        activity_id = "A200"
    elif ifc_type == "IfcColumn":
        quantity_type = "volume_m3"
        quantity_value = x_dim * y_dim * z_dim
        activity_id = "A300"
    else:
        quantity_type = "count"
        quantity_value = 1.0
        activity_id = "A999"

    record = DemoElement(
        global_id=element_gid,
        name=name,
        ifc_type=ifc_type,
        activity_id=activity_id,
        quantity_type=quantity_type,
        quantity_value=float(quantity_value),
        weight=1.0,
    )
    return element, record


def build_ifc(ifc_path: Path) -> list[DemoElement]:
    f = ifcopenshell.file(schema="IFC4")

    world = axis2placement3d(f, (0.0, 0.0, 0.0))
    context = f.create_entity(
        "IfcGeometricRepresentationContext",
        ContextIdentifier="Model",
        ContextType="Model",
        CoordinateSpaceDimension=3,
        Precision=1.0e-5,
        WorldCoordinateSystem=world,
    )

    units = f.create_entity(
        "IfcUnitAssignment",
        Units=[
            f.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE"),
            f.create_entity("IfcSIUnit", UnitType="AREAUNIT", Name="SQUARE_METRE"),
            f.create_entity("IfcSIUnit", UnitType="VOLUMEUNIT", Name="CUBIC_METRE"),
        ],
    )

    project = f.create_entity("IfcProject", GlobalId=gid(), Name="Demo Metric BIM Project")
    project.RepresentationContexts = [context]
    project.UnitsInContext = units

    site_placement = local_placement(f, None, (0.0, 0.0, 0.0))
    building_placement = local_placement(f, site_placement, (0.0, 0.0, 0.0))
    storey_placement = local_placement(f, building_placement, (0.0, 0.0, 0.0))

    site = f.create_entity("IfcSite", GlobalId=gid(), Name="Demo Site", ObjectPlacement=site_placement)
    building = f.create_entity("IfcBuilding", GlobalId=gid(), Name="Demo Building", ObjectPlacement=building_placement)
    storey = f.create_entity(
        "IfcBuildingStorey",
        GlobalId=gid(),
        Name="Level 01",
        ObjectPlacement=storey_placement,
        Elevation=0.0,
    )

    f.create_entity("IfcRelAggregates", GlobalId=gid(), RelatingObject=project, RelatedObjects=[site])
    f.create_entity("IfcRelAggregates", GlobalId=gid(), RelatingObject=site, RelatedObjects=[building])
    f.create_entity("IfcRelAggregates", GlobalId=gid(), RelatingObject=building, RelatedObjects=[storey])

    elements = []
    records: list[DemoElement] = []

    specs = [
        # ifc_type, name, x_dim, y_dim, z_dim, base_center, angle_rad, predefined_type
        ("IfcSlab", "DEMO_SLAB_LEVEL_01", 12.0, 8.0, 0.25, (6.0, 4.0, -0.25), 0.0, "FLOOR"),
        ("IfcWall", "DEMO_WALL_SOUTH", 12.0, 0.20, 3.20, (6.0, -0.10, 0.0), 0.0, "STANDARD"),
        ("IfcWall", "DEMO_WALL_NORTH", 12.0, 0.20, 3.20, (6.0, 8.10, 0.0), 0.0, "STANDARD"),
        ("IfcWall", "DEMO_WALL_WEST", 8.0, 0.20, 3.20, (-0.10, 4.0, 0.0), math.pi / 2.0, "STANDARD"),
        ("IfcWall", "DEMO_WALL_EAST", 8.0, 0.20, 3.20, (12.10, 4.0, 0.0), math.pi / 2.0, "STANDARD"),
        ("IfcColumn", "DEMO_COLUMN_C1", 0.40, 0.40, 3.20, (2.0, 2.0, 0.0), 0.0, "COLUMN"),
        ("IfcColumn", "DEMO_COLUMN_C2", 0.40, 0.40, 3.20, (10.0, 2.0, 0.0), 0.0, "COLUMN"),
        ("IfcColumn", "DEMO_COLUMN_C3", 0.40, 0.40, 3.20, (2.0, 6.0, 0.0), 0.0, "COLUMN"),
        ("IfcColumn", "DEMO_COLUMN_C4", 0.40, 0.40, 3.20, (10.0, 6.0, 0.0), 0.0, "COLUMN"),
    ]

    for spec in specs:
        element, record = make_rectangular_prism(
            f=f,
            context=context,
            spatial_placement=storey_placement,
            ifc_type=spec[0],
            name=spec[1],
            x_dim=spec[2],
            y_dim=spec[3],
            z_dim=spec[4],
            base_center=spec[5],
            angle_rad=spec[6],
            predefined_type=spec[7],
        )
        elements.append(element)
        records.append(record)

    f.create_entity(
        "IfcRelContainedInSpatialStructure",
        GlobalId=gid(),
        RelatingStructure=storey,
        RelatedElements=elements,
    )

    ensure_parent(ifc_path)
    f.write(str(ifc_path))
    return records


def write_schedule(path: Path) -> None:
    ensure_parent(path)
    rows = [
        {
            "activity_id": "A100",
            "activity_name": "Level 01 slab completed",
            "planned_start": "2026-05-01",
            "planned_finish": "2026-05-03",
            "planned_weight": "0.30",
        },
        {
            "activity_id": "A200",
            "activity_name": "Level 01 walls completed",
            "planned_start": "2026-05-04",
            "planned_finish": "2026-05-10",
            "planned_weight": "0.45",
        },
        {
            "activity_id": "A300",
            "activity_name": "Level 01 columns completed",
            "planned_start": "2026-05-02",
            "planned_finish": "2026-05-06",
            "planned_weight": "0.25",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_element_activity_map(path: Path, records: list[DemoElement]) -> None:
    ensure_parent(path)
    rows = []
    for record in records:
        row = asdict(record)
        row["notes"] = "synthetic_demo_bim_for_pipeline_validation"
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metric_anchor_files(anchor_path: Path, distance_path: Path) -> None:
    ensure_parent(anchor_path)
    anchors = [
        {"anchor_id": "A", "description": "floor southwest corner", "bim_x_m": 0.0, "bim_y_m": 0.0, "bim_z_m": 0.0},
        {"anchor_id": "B", "description": "floor southeast corner", "bim_x_m": 12.0, "bim_y_m": 0.0, "bim_z_m": 0.0},
        {"anchor_id": "C", "description": "floor northeast corner", "bim_x_m": 12.0, "bim_y_m": 8.0, "bim_z_m": 0.0},
        {"anchor_id": "D", "description": "floor northwest corner", "bim_x_m": 0.0, "bim_y_m": 8.0, "bim_z_m": 0.0},
        {"anchor_id": "COL1", "description": "column C1 center base", "bim_x_m": 2.0, "bim_y_m": 2.0, "bim_z_m": 0.0},
        {"anchor_id": "COL2", "description": "column C2 center base", "bim_x_m": 10.0, "bim_y_m": 2.0, "bim_z_m": 0.0},
        {"anchor_id": "COL3", "description": "column C3 center base", "bim_x_m": 2.0, "bim_y_m": 6.0, "bim_z_m": 0.0},
        {"anchor_id": "COL4", "description": "column C4 center base", "bim_x_m": 10.0, "bim_y_m": 6.0, "bim_z_m": 0.0},
    ]

    with anchor_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "anchor_id",
            "description",
            "bim_x_m",
            "bim_y_m",
            "bim_z_m",
            "scan_x",
            "scan_y",
            "scan_z",
            "use_for_scale",
            "use_for_registration",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for anchor in anchors:
            row = dict(anchor)
            row.update({"scan_x": "", "scan_y": "", "scan_z": "", "use_for_scale": "true", "use_for_registration": "true"})
            writer.writerow(row)

    distances = [
        {"distance_id": "AB", "anchor_a": "A", "anchor_b": "B", "distance_m": 12.0},
        {"distance_id": "AD", "anchor_a": "A", "anchor_b": "D", "distance_m": 8.0},
        {"distance_id": "AC_DIAGONAL", "anchor_a": "A", "anchor_b": "C", "distance_m": round(math.sqrt(12.0**2 + 8.0**2), 6)},
        {"distance_id": "COL1_COL2", "anchor_a": "COL1", "anchor_b": "COL2", "distance_m": 8.0},
        {"distance_id": "COL1_COL3", "anchor_a": "COL1", "anchor_b": "COL3", "distance_m": 4.0},
    ]

    with distance_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(distances[0].keys()))
        writer.writeheader()
        writer.writerows(distances)


def write_reference_primitives(path: Path) -> None:
    ensure_parent(path)
    payload = {
        "coordinate_system": {
            "units": "meters",
            "x_axis": "building length",
            "y_axis": "building width",
            "z_axis": "vertical",
            "origin": "southwest floor corner",
        },
        "matching_strategy": {
            "production_transform": "Sim3 uniform scale + rotation + translation",
            "diagnostic_only": "axis-wise scale may be estimated to detect reconstruction distortion but must not be silently used as final metric truth",
            "preferred_features": ["floor_plane", "wall_planes", "column_centers", "known_distances"],
        },
        "planes": [
            {"id": "floor_z0", "type": "floor", "normal": [0, 0, 1], "point": [0, 0, 0]},
            {"id": "wall_south", "type": "wall", "normal": [0, -1, 0], "point": [6, 0, 1.6]},
            {"id": "wall_north", "type": "wall", "normal": [0, 1, 0], "point": [6, 8, 1.6]},
            {"id": "wall_west", "type": "wall", "normal": [-1, 0, 0], "point": [0, 4, 1.6]},
            {"id": "wall_east", "type": "wall", "normal": [1, 0, 0], "point": [12, 4, 1.6]},
        ],
        "column_centers": [
            {"id": "COL1", "center_base": [2, 2, 0], "size": [0.4, 0.4, 3.2]},
            {"id": "COL2", "center_base": [10, 2, 0], "size": [0.4, 0.4, 3.2]},
            {"id": "COL3", "center_base": [2, 6, 0], "size": [0.4, 0.4, 3.2]},
            {"id": "COL4", "center_base": [10, 6, 0], "size": [0.4, 0.4, 3.2]},
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_manifest(path: Path, records: list[DemoElement]) -> None:
    ensure_parent(path)
    payload = {
        "asset_type": "synthetic_demo_bim",
        "purpose": "pipeline_validation_only",
        "is_real_project_bim": False,
        "ifc_schema": "IFC4",
        "units": "meters",
        "element_count": len(records),
        "elements": [asdict(r) for r in records],
        "warnings": [
            "This IFC is synthetic and must not be used as a real project design model.",
            "Use this file to validate Stage 8/Stage 9 file contracts, IfcOpenShell parsing, registration, and reporting.",
            "Final metric evaluation should use a real IFC or a validated project-specific BIM model.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_ifc(ifc_path: Path) -> dict[str, int]:
    model = ifcopenshell.open(str(ifc_path))
    return {
        "IfcProject": len(model.by_type("IfcProject")),
        "IfcSite": len(model.by_type("IfcSite")),
        "IfcBuilding": len(model.by_type("IfcBuilding")),
        "IfcBuildingStorey": len(model.by_type("IfcBuildingStorey")),
        "IfcSlab": len(model.by_type("IfcSlab")),
        "IfcWall": len(model.by_type("IfcWall")),
        "IfcColumn": len(model.by_type("IfcColumn")),
        "IfcElement": len(model.by_type("IfcElement")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate demo IFC/BIM assets for Stage 8/9 pipeline validation.")
    parser.add_argument("--output-dir", default="data/bim/design")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    ifc_path = out / "model.ifc"
    schedule_path = out / "schedule.csv"
    element_map_path = out / "element_activity_map.csv"
    anchors_path = out / "metric_anchors.csv"
    distances_path = out / "known_distances.csv"
    primitives_path = out / "reference_primitives.json"
    manifest_path = out / "demo_bim_manifest.json"

    outputs = [
        ifc_path,
        schedule_path,
        element_map_path,
        anchors_path,
        distances_path,
        primitives_path,
        manifest_path,
    ]

    existing = [p for p in outputs if p.exists() and p.stat().st_size > 0]
    if existing and not args.force:
        raise SystemExit(
            "Refusing to overwrite existing BIM demo assets without --force:\n"
            + "\n".join(str(p) for p in existing)
        )

    records = build_ifc(ifc_path)
    write_schedule(schedule_path)
    write_element_activity_map(element_map_path, records)
    write_metric_anchor_files(anchors_path, distances_path)
    write_reference_primitives(primitives_path)
    write_manifest(manifest_path, records)

    counts = validate_ifc(ifc_path)

    print("DEMO_BIM_ASSETS_OK")
    print(f"ifc={ifc_path}")
    print(f"schedule={schedule_path}")
    print(f"element_activity_map={element_map_path}")
    print(f"metric_anchors={anchors_path}")
    print(f"known_distances={distances_path}")
    print(f"reference_primitives={primitives_path}")
    print(f"manifest={manifest_path}")
    print(f"counts={json.dumps(counts, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
