# MyCON B — Field-case protocol for 4D BIM / physical progress (v1)
Dated 2026-11-14. This is a **protocol**, not an executed field study. B-owned Stage09/11 consumer; A/D supply metric geometry, IFC/registration evidence; C independently evaluates held-out GT. No original construction/Recorder/IFC/checkpoint data are modified.

## Minimum real inputs (immutable, independently verifiable)
1. **As-planned IFC**: source model with 22-character actual IfcRoot GlobalIds; SHA256 of original IFC. Read-only IfcOpenShell-based index manifest `mycon.ifc_identity_index.v1`, project-relative IFC path+hash, `bim_world` frame, output units `m`, positive source IFC length unit conversion to metres, actual `global_id`, IFC class and storey, independent approval doc SHA256. This B adapter optionally uses IfcOpenShell to reopen the IFC and compare each indexed entity and native unit scale. IfcOpenShell missing or file/index mismatch -> index valid=false; linked statuses=unknown.
2. **As-built site capture**: independent source images/geometry and A's `mycon.bim_geometry_evidence.v1` sidecars; manifest SHA256, IFC/scan/registration hashes, verified metric scale, 4x4 transform, independently surveyed holdout anchors, camera intrinsics/poses, view_id/image_sha256, IFC surface sample/triangle identity and audited visibility. No 3D Gaussian centre or arbitrary reconstruction scale can certify surfaces.
3. **Approved quantities** per IFC GlobalId: planned and executed quantities, common m/m2/m3/count units, signed measurement/reviewer, approval/reference and document SHA256, UTC as-of data date; no quantity derived from coverage. A nonzero measurement supports partial installed work even if imagery is occluded; zero amount alone does not certify absence. Explicit independent zero/absence evidence with `absence_verified`, `absence_evidence_ref`, `absence_document_sha256`, `absence_approved_by`, `absence_approved_at_utc` is required for categorical `not_built`.
4. **Approved 4D schedule**: activity_id + wbs_code, dated approved planned percentage, approved baseline version and source SHA256, original start/finish, reference date; mapping CSV `activity_id,ifc_global_id,weight` with approved BOQ weights/basis/reviewer/date/hash; separate approved compatible activity/project weights. Missing or contradictory weights => physical % null, never 0.
5. **Independent expert and test truth (optional)**: source-verified element status, measured element fractions and activity percent, date-matched and site-held-out; no training/test building or site reuse. Record reviewer, capture ID, file hashes and missing predictions. Without this, categorical precision/recall/F1 and progress MAE remain null.

## Scenario matrix for operator review

| Observed construction evidence | Source quantity | Geometry/visibility gate | Reported decisions |
| --- | --- | --- | --- |
| IFC wall or pipe visible and surface matches | signed executed 8 / planned 10 m2 | valid metric and cameras | construction `in_progress`, observation `observed`, numeric 80% *element fraction*; not evidence of workmanship acceptance |
| Installed duct hidden behind ceiling | signed executed 10 / planned 10 m | trusted raycast `occluded` | construction `built`, observation `not_observable`; do not erase verified quantity |
| IFC element outside all camera frustums | missing quantity | valid `out_of_fov` | construction `unknown`, observation `not_observable`, percent `null` |
| Empty scan or unscaled reconstruction | missing quantity | metric/registration false | construction `unknown`, observation `unknown`, percent `null`; cannot assert not_built |
| Independently surveyed absent wall | verified executed 0 / planned 10 with second signed and dated absence record | imagery may be absent | construction `not_built`; numeric fraction 0 from signed measurement, not from point absence |
| IFC GlobalId absent from model index | quantity exists | unmatched IFC entity index | `ifc_linked_construction_status=unknown`, `ifc_identity_quality_gate=blocked`; cannot certify IFC identity |
| Two distinct activities share one element | measured fraction independently known | approved mappings with valid weights | separate activities can share element with disclosed many-to-many attribution; reject duplicate row/weight ambiguity |

## Versioned execution, reproducibility and output
- Run in an isolated `progress-*` directory; if directory exists exit 2, no force/overwrite. All original inputs stay read-only. Optional `--ifc-index-manifest` is a project-relative read-only approved inventory; `--geometry-manifest` requires independent expected IFC/scan/registration hashes, scale/pose QA and expected project ID.
- The five files stay exactly `element_status.csv`, `element_metrics.csv`, `activity_progress.csv`, `schedule_variance.json`, `progress_dashboard.json`. The dashboard embeds `observability_report`, `approved_quantity_report`, `wbs_boq_weight_report`, `uncertain_elements_report`, manager summary, primary scientific_progress.v2, optional independent GT/expert metrics and source hashes.
- Element fractions = [0,1] or null. Activity/project = weighted percent [0,100] or null. Schedule deviation = percentage points or null. Unknown quantity with valid BOQ produces deterministic partial-identification bounds only, not 95% CI. Legacy acceptance status and Wilson interval must not be used as physical progress.
- Reader never copies source image bytes: SHA256+view_id+surface sample IDs are auditable immutable pointers. Stage8 evidence must be registered to actual IFC `global_id`; unlabeled plane/edge/corner diagnostics are not IFC identity.

### Real input IFC index example (descriptive, NOT a real model)
```json
{
  "schema_version": "mycon.ifc_identity_index.v1",
  "ifc_file": {"path": "data/bim/reference/site_original.ifc", "sha256": "<64 real hex chars>"},
  "coordinate_frame": "bim_world", "units": "m", "ifc_length_scale_to_m": 0.001,
  "provenance": {"generator": "ifcopenshell", "reviewer": "<independent name>",
                  "approval_document_sha256": "<64 real hex chars>"},
  "elements": [{"global_id": "<real IFC 22-char compressed GUID>", "ifc_class": "IfcWall",
                 "storey_id": "<site storey id>"}]
}
```
The example hash placeholders cannot pass validation. Source-index entries only become verifiable if the actual IFC is present, its SHA matches, IfcOpenShell parses it and every GlobalId/class/source units match. A source-file hash alone does not establish independently certified as-built work.

## Acceptance (not yet completed)
CPU: `pytest -q tests/test_research_progress_monitor.py tests/test_research_progress_scientific.py`, all relevant legacy Stage09/11, CLI synthetic and manifest-failure negative paths; record exact SHA, command, exit and hardware.
Real: one independently measured IFC/project baseline/BOQ and registered site capture; sample of occluded, out-of-FOV, missing, built, partial and independently absent elements; C evaluates against site-held-out expert truth. Field-release quality gate remains `blocked_without_independent_real_survey_review` until reviewed by coordinator. No synthetic result becomes real site performance.
