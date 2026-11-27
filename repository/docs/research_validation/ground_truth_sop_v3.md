# Independent as-built Ground Truth SOP — MyCON C

**Schema:** `mycon.validation.ground_truth.v1` / **status:** `no_verified_field_gt` as of 2026-11-27. Civil engineering measurements are not inferred from arbitrary reconstruction-unit coordinates.

## Source collection and blinding

1. Independent survey crew, **not** a model tuning/data capture operator, measures dated as-built geometry using calibrated total station / TLS with instrument model, calibration certificate, environmental settings, reference frame/CRS, accuracy class/LoA, tolerances, timestamps and signed raw-file hashes. Record chain of custody and image/privacy consent. As-designed BIM/IFC alone is **not** ground truth for as-built deviations.
2. Independent custodian freezes original survey and supports (cloud/mesh/planes/edges/observed corners), instrument and reviewer files by SHA256, identifies minimum observable structural coverage, samples and excluded regions. Do not crop to best-looking regions after seeing model output.
3. Define **fit/control anchors** and **completely disjoint held-out check anchors** *before* fitting any rigid/Sim(3) transform. A stable Sim3 fit needs at least 3 non-collinear well-spread valid anchors, with held-out controls separate; recommended >=4 dispersed holdouts for spatial diagnostics. Do not estimate field accuracy from optimizer fit residuals or anchors used in alignment. Surveyor signs transform uncertainty and unit conversion.
4. Record `(site_id,building_id,session_id,source_snapshot_sha256,reference_revision,control_id,coordinate_frame,units,reviewer,review_date)` on all materials. Capture sessions of same site remain in same train/val/test outer split. Record equipment and scan angle/occlusion uncertainty.
5. Structural GT consists of finite 3D plane patches with observed polygon boundaries, normal and covariance, finite edges with support extents, `observed_supported` corners with independent landmark IDs/uncertainty, unknown/extrapolated masks, geometric class and reviewer disagreement. Do not treat infinite planes as observed geometry.
6. Scan-to-BIM optional: surveyed geometry-to-IFC GlobalId association is reviewed per element by two reviewers, explicitly retain `unassigned`, design revision and construction date; nearest AABB is only a hypothesis. Obtain signed executed/planned quantity plus compatible unit, approved BOQ weights and schedule baseline ID/date. Occluded ≠ unbuilt; unknown must be preserved.
7. Reject GT if: no independent reviewer; missing raw hash; shared anchor in fit and holdout; invalid unit/frame; unsigned engineering tolerance; survey not contemporaneous; absent IFC identity/quantity approval; pointcloud actually generated from the model being tested. Report these as `not_evaluated` or failed receipts, never zeros.

## Expected handoff

- `gt_manifest.json` (`schema_version=mycon.validation.ground_truth.v1`): independent source files each with **project-relative** path, SHA256, units, coordinate frame, reviewer signoff and measurement covariance. Separate `survey_fit_anchors.csv` and `survey_heldout_anchors.csv` with non-overlapping stable IDs.
- `asbuilt_gt_cloud.ply` or supported mesh + surveyed finite features; `annotation_masks.json` of visible/occluded/unknown regions; capture and camera receipts locked before testing.
- Optional `ifc_revision.json`, `element_ground_truth.csv` keyed GlobalId, approved BOQ/quantity and baseline. No model train/test leakage.
- Two independent reviewer logs + adjudication file and frozen engineering tolerance schedule.

## Acceptance gate

Currently no independently signed comparable field GT or holdout anchor files have been supplied to C. **Metric scale, registration, as-built geometry Accuracy/F1 and physical progress all remain `not_evaluated`**. The authentic A zip is usable for **real internal diagnostics only**, and no self-generated COLMAP sparse 3D point is accepted as an independent control survey.