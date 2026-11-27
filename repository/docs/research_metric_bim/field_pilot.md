# D — Field measurement and IFC pilot (2026-11-27)

Status: **candidate only**, not a surveyed result. Run from repository/ on D branch. No edits to Stage8, original capture, geometry or IFC. No merge.

## Field collection — mandatory for independent scale

| Input | What to provide |
|---|---|
| Source | A-reviewed comparison_ready_geometry.v1 JSON; source reconstruction frame, SHA256-linked geometry files; fixed scan/reconstruction revision and session. Do not select points from screenshots or Gaussian centres alone. |
| Fit spans | At least 3 independently measured physical spans in different locations/directions, preferably long room spans (wall-to-wall, height, column edges), not solely small block dimensions. |
| Holdouts | At least 2 separately measured spans reserved before fitting; no common physical source group or physical endpoint ID with other spans. Prefer different instrument/operator or repeated independent survey session when practical. |
| Each span | Stable endpoint IDs, corresponding XYZ in original reconstruction units, actual length (m), 1-sigma instrument+measurement uncertainty (m), 1-sigma reconstructed span uncertainty (reconstruction units, positive), instrument ID, method, measurement date/time with timezone, operator, reviewer and evidence ID. Include photos/sketches/measurement logs and SHA256 receipt. |
| Nominal size | Catalogue block/profile lengths optional; never substitute for actual measured size. Record nominal and actual separately. |
| Tolerance | Project engineer specifies a positive diagnostic absolute length tolerance in metres BEFORE examining holdouts. It is not a certified tolerance or uncertainty. |

Collect long spans in three directions where practical; identify EXACT same physical endpoints in field and source geometry; document endpoints from surveyed marks or visible stable features, not hidden wall intersections. Use separate IDs for physical endpoints even across BIM and length paths. Do not rename the same physical feature to evade independence. Record calibration certificate, repeated observations, temperature/sag/laser alignment as applicable, survey reference and photo. Use an independent reviewer. A file hash proves byte identity, not accuracy or authorship.

Survey JSON extends `mycon.length_references.v1` with `geometry_manifest_sha256` and per-reviewed-actual record `endpoint_ids`, `measured_at`, `instrument`, `measurement_method`, `operator`, `field_evidence:{path,sha256}`. Retain proposed nominal records with approval proposed; not counted. Field CLI rejects duplicate physical groups/endpoint IDs, missing receipts, fewer than 3 fit/2 holdouts, inconsistent fit and mixed frames. Existing standalone core remains available for lower-readiness computational experiments.

## IFC path (optional, independent of length-only workflow)

1. Provide **original IFC2x3/IFC4/IFC4x3** (actual parser support depends on geometry), revision/date and list of relevant physical element GlobalIds (walls/slabs/columns). Specify engineering coordinate frame, project units, datum and any CRS/map conversion separately. The inventory uses object placement world coordinates **in SI metres**, not geodetic coordinates; do not multiply by unit scale twice. Do not use design model dimensions as field truth. Install optional dependency in separate venv: `python -m pip install "numpy>=1.26,<2" "ifcopenshell==0.8.4"`.
2. Run extraction for explicitly selected GUIDs. Inventory is read-only mesh candidates; triangle and vertex indices are zero-based, valid ONLY for the exact inventory SHA256. Faces preserve triangulated extents (including openings when represented by the kernel); no convex hull across holes. IFC tessellation vertices are not automatically physical corners; choose only visible, unambiguous corresponding physical corners and review them. Normal orientation for selected triangles explicitly +1/-1 from winding. No automatic ICP or whole-model match. Curved/complex/ambiguous surfaces need extra review and are not assumed planar beyond the selected triangle.
3. Select minimum **3 noncollinear fit corners when survey scale is fixed, 4 noncoplanar fit corners for free BIM scale**, and at least **2 independent control corners**, spread over the room and height. More fit controls are recommended (at least one extra for RANSAC). For each corner specify GlobalId + vertex_index, source scan XYZ, physical feature/endpoint ID, fit/control, observation support, independent reviewer and hashed evidence. Do not reuse a fit feature for heldout verification; physically equivalent/coincident target or source points are rejected. Length-fit endpoint must not be used as IFC holdout and vice versa.
4. Optionally select individual finite triangles via GlobalId + triangle_index + normal_sign with manual evidence. Map scan surface samples to individual reviewed triangles only with observation/support evidence; unknown/occluded/unmapped/out-of-FOV remain no-data. Selecting a face does not establish that it was observed. Never set missing surfaces to zero error or not built. For large planes, select multiple triangles and map samples explicitly. Signed error follows selected normal.
5. Predeclare independent position tolerance (metres), plus optional RANSAC threshold if used. IFC-only scale is a proposal even with BIM holdout agreement. Independent field controls are still necessary for real metric status.

## Commands (examples with placeholder project paths/IDs; do not run without actual inputs)

```bash
cd repository
python -m pip install "numpy>=1.26,<2"
# No BIM: mandatory independent survey path
python -m civil_metric_bim.field_run run --root /workspace/MyCon_R4 \
  --run-id metric-bim-field-20261127-a01 \
  --geometry reviews/comparison_ready.json \
  --survey reviews/lengths.json --length-tolerance-m <ENGINEER_LIMIT_M>

# IFC only: optional dependency, extract only specified elements
python -m pip install "ifcopenshell==0.8.4"
python -m civil_metric_bim.field_run extract-ifc --root /workspace/MyCon_R4 \
  --ifc reference/room.ifc --revision <REVISION> \
  --guid <WALL_GUID> --guid <SLAB_GUID> \
  --out reviews/ifc-inventory-001.json

# Combined: survey-derived scale fixed, solve proper rotation/translation
python -m civil_metric_bim.field_run run --root /workspace/MyCon_R4 \
  --run-id metric-bim-field-20261127-b01 \
  --geometry reviews/comparison_ready.json \
  --survey reviews/lengths.json --length-tolerance-m <ENGINEER_LIMIT_M> \
  --inventory reviews/ifc-inventory-001.json \
  --selection reviews/ifc-selection.json --position-tolerance-m <ENGINEER_LIMIT_M>
```

Omit --survey and --length-tolerance-m for BIM-only provisional scale. All paths are relative to root except root itself. Compute hashes e.g. `sha256sum path/to/file` or PowerShell `Get-FileHash -Algorithm SHA256` (lowercase hexadecimal). For the selection file use exact `inventory_sha256` and `geometry_manifest_sha256`. Template files below intentionally contain **null/empty placeholders**, not actual data or valid approved records; complete and independently review each row.

- `repository/civil_metric_bim/templates/field_lengths.template.json`
- `repository/civil_metric_bim/templates/ifc_selection.template.json`

## Outputs & review

`runs/<run-id>/metric_bim/`: scale/uncertainty, transform, fit and independent holdout errors, deviation maps, Viewer handoff and hashed manifest. `field_inputs/<run-id>/`: generated reviewed_ifc_adapter.json (IFC route), field_pilot_receipt.json linking inputs and alignment manifest by SHA256, per-control error/threshold comparison. Each run ID is create-only; after interrupted/failed output use a new ID, never delete evidence. Inspect controls even if a candidate fails tolerance; investigate input errors, do not tune thresholds against heldouts. All receipt/manifest gates stay pending / false. Geometry provenance and field evidence are supplied by operator; no automated proof of camera visibility, physical surface quality, survey correctness, IFC design-as-built correspondence or calibrated uncertainty. IFC inventory file hash does not authenticate its origin; keep extraction logs, software version and reviewer record.

No measured room data was supplied in this turn. Real IFC backend is optional; fixture tests distinguish parser execution from adapter tests; CI without IfcOpenShell skips the native backend fixture. Read-only IFC geometry API reference: https://docs.ifcopenshell.org/ifcopenshell-python/geometry_processing.html and https://docs.ifcopenshell.org/ifcopenshell/geometry_settings.html (world-coords / convert-back-units).
