# MyCON — 14021208 / Front-Side Room Metric Registration Pilot
Date: 2026-11-12 | Coordinator-owned protocol | Scope: *as-designed 3ds Max scene vs as-observed room*, eventual IFC/BIM comparison.

## Evidence grounded in actual source archives (read-only)

The user's Library contains `M2R_14021208_GitHub_TOOLKIT.zip`, `M2R_14021208_Revit_LAYERS.zip`, and an earlier raw M2R export. The toolkit's `EXPORT_REPORT_ORIGINAL.txt` reports original source `14021208.max`, system units **1 Max unit = 1 centimeter**, 3,339 scene nodes; exported OBJ, FBX and DWG, but the SAT exporter failed. The `SOURCE_ANALYSIS.json` counts 1,441 geometric sections, 2,876,798 OBJ vertices and 2,381,810 polygons. The derived `geometry_by_layer/*.obj` mesh is **Z-up but still centimeters**, after vertex transform `(x,y,z) -> (x,-z,y)`. Original OBJ source and converted OBJ must not be silently mixed.

Verified download SHA-256:
- `M2R_14021208_GitHub_TOOLKIT.zip`: `cfbbb92a68a96dc325acf8feb7baa44a157406c0e3a7ed55321ac5aa43a31780`
- `M2R_14021208_Revit_LAYERS.zip`: `51c5500c588d98e7ea3351d72409276eeb95cf51cdbc532a459ce156b5395cbb`
- Embedded `geometry_by_layer/LAYER_Main-Layer.obj`: `0c1d22edbbb799cccf8d1e1fdea9d035bfdc1234b92ccaf01daa2efc665ccc18`

**Do not treat these as independent ground truth of built geometry:** 3ds Max scene is a design mesh, not a native IFC hierarchy; extracted objects do not have verified IFC GlobalIds. Neighbors/trees/main building share a scene and full bbox cannot identify the video room. The user's hypothesis is that video depicts **one room at the front-side of this building (building right, facade-view left)**. That ROI must be visually identified and approved using matched *actual frame images and rendered model views* before any registration is accepted. Do not guess which coordinate side or use all geometry.

## Mandatory work: A — geometry and metric scaling

A1. Parse the M2R OBJ scene into per-object stable `cad_object_id` and `cad_component_version`, preserving original names, part numbers, layer, mesh checksum, cm-to-m factor and origin; exclude duplicate names as unique identifiers. Do not automatically fabricate IFC GlobalIds.

A2. Render exterior facade, plan and room candidate object boundaries. Create a **room ROI candidate list** and visual correspondence sheet (rendered CAD view vs source video frame), with explicit camera-facing left/right convention. Confirm exact room; freeze `room_roi_reviewed=true` only with audited visual evidence. Include uncertainty if multiple spaces remain candidates. No inference that a full-building bbox is room dimension.

A3. Extract only stable structural reference surfaces/landmarks in that ROI; reject decorative clutter, furnishing, vegetation and irrelevant neighbors. Surface sampling must include normals/mesh origins, not solely AABB. Use structural corners, beam/column intersections, openings and floor/ceiling intersections as provisional correspondences. Design geometry may differ from construction status: absent components are not automatically alignment outliers without review.

A4. Align reconstruction → CAD **in meters** with robust similarity transform
`p_CAD_m = s R p_reconstruction + t`, `R in SO(3), s>0`. Estimate via 3D-3D RANSAC/Umeyama or TEASER++ (when robust correspondences exist); RANSAC global FPFH / Super4PCS and point-to-plane ICP can be benchmarked as optional coarse/fine refinements. Avoid globally cropping a partially observed room against all buildings. Reject low overlap, degenerate landmarks and false room matches; record failed hypotheses. Align COLMAP camera extrinsics and gsplat geometry consistently if applicable.

A5. **Separate CAD registration from independent metric certification**. Design-derived dimensions and visual VLM guesses do *not* certify metric ground truth. Capture at least three independent, documented physical length controls (prefer mutually different width/depth/height features: structural bay, door/window opening, floor-ceiling) with units, tolerance, date, reviewer and source hash. Use fit anchors and separate held-out anchors. Compute held-out TRE/RMSE, scale drift, plan-vs-field deviation, fit inliers/outliers and transformation uncertainty; reject geometric mismatch instead of forcing ICP to hide as-built change. If survey absent, report `alignment_fit_only_metric_unverified`.

A6. Only after successful ROI identification, surface-to-room registration and coordinate certification produce additive Stage8 geometry evidence `mycon.bim_geometry_evidence.v1`; it must keep source `cad_object_id` distinct from `ifc_global_id`. If a real IFC model is supplied, include an explicit reviewed CAD-to-IFC correspondence table with mesh hashes, IFC units and LoD. Never label a CAD mesh as valid BIM semantics or use alignment fitness as progress percentage.

A7. Preserve original `.max`, archives, OBJ/FBX/DWG, recordings, Checkpoints and gsplat artifacts. Do not retrain models from zero without user approval. A should keep edits within its owned files and update PR #38.

## Mandatory work: B — physical progress and VLM evidence consumption

B1. A matched CAD source is a *design reference*; without confirmed IFC relationship, it is **not** an IFC-linked element-level physical progress source. Use explicit `cad_object_id → ifc_global_id` mapping (and provenance) or keep status `unmapped_to_ifc`.

B2. Camera visibility must come from pose-calibrated projection/raycast into the specific ROI: `visible / occluded / out_of_fov / ambiguous / unknown`. No scan points ≠ not built; VLM prediction ≠ independent quantity measurement. Preserve separately approved BOQ/executed quantity weighting, `physical_progress_estimate` vs `observability_ratio` vs calibrated `evidence_confidence`.

B3. Accept semantic observations only if containing source image/session ID, pixel bbox/mask and reviewable image crops, matched `cad_object_id` or IFC candidate, geometric compatibility check and reason code. Calibration must be measured on held-out annotated samples; language-only self-rated confidence is not calibrated.

B4. B may develop consumer adapters with synthetic fixtures now but must defer actual physical progress conclusions until certified geometry, approved quantities and schedule baseline exist. Edit only B-owned files; update PR #37.

## Mandatory work: C — scientific validation and VLM

C1. Build a **room-specific independent benchmark** with test capture/session holdout, manually annotated source-video vs CAD rendered viewpoint landmark correspondence, field-surveyed dimensions, GT room identity, and correct/incorrect/unknown object labels.
C2. Report distinct metrics:
- `room_identification_accuracy`, top-k recall and false-match count
- transformation scale ratio, independent surveyed length absolute errors, TRE/RMSE and P95 (m), ICP inlier ratio and wrong-ROI/failure rate
- structural geometry precision/recall/F-score on **observed visible surfaces** with masks and thresholds; distinguish design-vs-as-built differences
- VLM object grounding bbox/mask IoU, per-class precision/recall/F1, evidence citation validity, temporal consistency, hallucination and abstention rate
- separately measured physical quantity MAE/bounds and schedule deviation, only if independently verified GT exists
- runtimes/VRAM plus failure/timeout denominators.

C3. Compare A variants on matched inputs: (1) COLMAP-only unscaled, (2) CAD anchored Sim3, (3) robust coarse + ICP, (4) metric controls + holdout, (5) semantic/visibility feature ablations. Optional VGGT/DUSt3R/GS geometry only on actual outputs. Include site/group bootstrap only when independent sites/groups exist. If single room, call it an **exploratory case study**, not generalization across projects. The reported 2024/2025 paper benchmarks are not directly comparable until shared data/protocol exists. Edit only C-owned files, update PR #36.

## VLM policy reserved to coordinator integration ownership

Use a **tool-grounded evidence-first** pipeline: (1) registered frames and masks; (2) CAD/IFC projection + geometric/temporal checks; (3) VLM proposes structured semantic observations; (4) deterministic schema validators verify image/region IDs and geometric consistency; (5) confidence calibrated independently; (6) unresolved/occluded/mismatched predictions are UNKNOWN requiring review. Reject direct VLM dimension estimates as physical scale; use actual measured CAD/survey anchors. Do not let VLM infer progress percentages from visual appearance.

Proposed machine-readable observation: `{frame_sha256,view_id,capture_session_id,roi_id,object_source_kind,cad_object_id,ifc_global_id|null,element_class_candidate,evidence_bbox_xyxy_px|mask_ref,observed_material_or_stage,occlusion_state,geometry_crosscheck,temporal_crosscheck,decision: supported|contradicted|unknown,reason_codes,model_id,prompt_sha256}`.

## Paper/code references to implement and cite

1. Y. Zhang et al., *Global BIM-point cloud registration and association for construction progress monitoring*, **Automation in Construction** 168 (2024) 105796. DOI `10.1016/j.autcon.2024.105796`. SDF-based representation and coarse-to-fine matching of BIM/point clouds.
2. G. Z. Tsige et al., *Automated Scan-vs-BIM Registration Using Columns Segmented by Deep Learning for Construction Progress Monitoring*, ISPRS Archives 2025. DOI `10.5194/isprs-archives-XLVIII-G-2025-1455-2025`. Column correspondences + RANSAC registration.
3. `https://github.com/MIT-SPARK/TEASER-plusplus` robust outlier-tolerant scale/rotation/translation.
4. `https://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html` RANSAC-FPFH and ICP; `https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html` raycasting visibility.
5. `https://colmap.github.io/pycolmap/pycolmap.html` COLMAP reconstruction Sim3 transformation and frames.
6. `https://docs.ifcopenshell.org/ifcopenshell-python/geometry_processing.html` semantic element-level IFC surfaces.
7. `https://docs.opencv.org/4.11.0/d5/d1f/calib3d_solvePnP.html` calibrated 3D-to-2D landmark PnP and RANSAC.
8. `https://github.com/facebookresearch/sam2` video object/region masks; not engineering truth without validation.
9. *Are large pre-trained vision language models effective construction safety inspectors?* (2026), DOI `10.1017/dce.2026.10044`, reports limitations in precise VLM visual grounding; necessitates auditable image/mask checks.

## Stage gate and integration decision

**NO MERGE YET.** Use one short targeted continuation across the existing A/B/C PRs (A real room geometry first, B adapter, C benchmark), with no cross-owner edits. Coordinator holds the integration branch and merges only after full repository regression and verified room/metric provenance; if raw .max or site measurement source unavailable, finish all reusable tooling but mark the field case `blocked_missing_input`. Unit-test success does not imply real room registration or publication readiness.
