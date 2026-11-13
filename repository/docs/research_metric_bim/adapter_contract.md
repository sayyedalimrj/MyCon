# MyCON Workstream D — civil metric-scale and BIM/IFC comparison adapter

Status: **isolated implementation candidate; not field validated**. Coordinator approved D ownership under [six-stream amendment](https://github.com/sayyedalimrj/MyCon/blob/research/mycon-integration-20261112/repository/docs/integration/parallel_6_streams_approval_20261113.md), [Issue #44](https://github.com/sayyedalimrj/MyCon/issues/44). No change to A-owned Stage 8, Stage 7, B/C, main or original Drive data. Entry point: `civil_metric_bim.core`. Optional NumPy-only adapter; no IFC file is read implicitly and no physical-progress inference is performed.

## Two independent sources of scale

1. **Length anchors:** supply a JSON document conforming to `mycon.length_references.v1`. Each record must declare `source_kind` = `block`, `steel_section` or `field_measurement`; two source-frame XYZ endpoints; `physical_source_id`; `role` = `fit` or `control`; nominal length (optional); **actual measured/verified** length, actual measurement uncertainty and scan-span uncertainty; qualitative `confidence.grade`, reviewer/evidence ID, source and method/provenance. Nominal/catalogue steel/block dimension is *never* automatically substituted for measured actual size. Proposed dimensions can be reviewed and manually accepted. At least **3 approved high-reviewed independent physical source groups** are needed for length-based fitting, with 3%/3-sigma **computational** consistency guard (not a field-acceptance tolerance). An independent approved holdout is required for credible evaluation; if absent the scale remains a provisional candidate flagged as such. Duplicate physical group across fit and control fails. Approximate propagated uncertainty and between-reference dispersion are separately disclosed, not falsely called a 95% confidence interval.

2. **BIM/IFC feature alignment:** supply `mycon.bim_feature_adapter.v1`, a **manually reviewed** mapping of reconstructed supported corners and/or finite plane centres to matching reference BIM coordinates in metres. `kind=corner|plane`, explicit source and target XYZ, unique `physical_feature_id`, `observation_state`, role and approval are required. Approved fit matches produce rotation and translation via a proper (determinant +1) Umeyama similarity. If a surveyed length-based scale exists, it is held fixed; otherwise 4 or more noncoplanar reviewed BIM points yield an **unverified BIM-derived scale proposal**. Matching physical features cannot be reused as holdouts. Collinear/degenerate solutions fail closed. BIM planes are finite polygons with a reviewed normal; plane matches may additionally report signed point-to-plane and normal-angle residual. No extrapolated or unobserved corner may silently become a fit control.

### Robust registration and uncertainty

The default deterministic baseline is Umeyama on all approved BIM fit pairs. Optional robust mode uses `registration: {"method":"ransac_umeyama","inlier_threshold_m":0.01,"max_trials":200,"seed":42}` in the BIM feature adapter; this numeric threshold is **an illustrative synthetic choice only**, NOT an accepted construction tolerance. The operator must set a metre threshold according to surveyed target precision; code never tunes it using holdout controls. RANSAC samples only fit matches, rejects inconsistent correspondences, refits on inliers and reports all outlier IDs. BIM-only candidate scale uncertainty is `null` without independent field controls; for five or more fit inliers, optional leave-one-feature-out stability is reported **separately** and is not a metrological standard error.

Fit residual RMS/p95/max is recorded separately from independent control RMSE/p95/max; the latter is never fed into registration. When fewer controls are available, control-error values remain `null`. A 3D similitude is only a *proposal* until survey scale, independent validation, IFC revision and reviewed visibility/finite surfaces are all approved.

**Point-to-plane ICP is not silently applied.** Given the risk of aligning wrong/repeated walls, the standalone NumPy adapter stops at fixed-scale or free-scale reviewed Umeyama/RANSAC candidates. An optional Open3D ICP producer can be proposed separately under the authorized interface (not Stage8 edits) after controlled real BIM support and evaluation; otherwise it would add an unjustified implementation layer.

**No special room, MAX geometry or project dimensions are embedded**; the numbers in the synthetic pytest cases are artificial algorithm fixtures only. This adapter accepts reference feature JSON, not a raw IFC geometry conversion. A future owner-approved IFC producer must extract finite plane/corner coordinates and IFC GlobalIds into this schema and supply an actual revision/hash. For non-synthetic model provenance, `model_provenance.asset.path` is a project-relative path whose SHA256 is checked before any output. This JSON adapter is not a replacement for actual IFC parsing or surveying.

## Run contract

Run inside the **repository/** directory of a complete checkout. Each input must live under `--project-root`; all generated paths in manifests are project-relative.

```bash
python -m civil_metric_bim.core \
  --project-root /workspace/MyCon_R4 \
  --run-id metric-bim-pilot-001 \
  --references-json /workspace/MyCon_R4/reviewed_lengths.v1.json \
  --bim-json /workspace/MyCon_R4/reviewed_bim_features.v1.json \
  --geometry-manifest-json /workspace/MyCon_R4/runs/geom-ready-.../geometry/comparison_ready.json
```

Omit either references or BIM JSON to exercise the **other path independently**. The optional geometry manifest must be `mycon.comparison_ready_geometry.v1` and agree on source coordinate frame; all listed upstream output SHA256 checks run before output creation. An absent original capture remains untouched; a present-but-modified original source fails SHA validation. No image/PLY/checkpoint is rewritten or read for registration. Output runs use unique `metric-bim-*` IDs and create-only staging. Reusing even a partial existing run is refused.

### Outputs under `runs/<run-id>/metric_bim/`

| Artifact | Interpretation |
|---|---|
| `scale_estimate.json` | Candidate scale m/reconstruction-unit, approximate 1-sigma uncertainty, accepted/rejected reference IDs, nominal-minus-actual difference, heldout IDs and fit diagnostics |
| `alignment_transform.json` | Proper Sim(3): scale, 3×3 rotation, XYZ translation in metres, 4×4 matrix, fit/heldout feature IDs and explicit source/target frames |
| `alignment_report.csv` | Per reviewed BIM match signed plane offset when defined, unsigned Euclidean point error, angle error when normals available; fit versus control disambiguated |
| `heldout_control_errors.csv` | Independent BIM feature and measured-length control residuals; **never used to solve scale/rotation/translation** |
| `deviation_map.csv` | Per-sample signed and unsigned point-to-finite-plane difference in metres, explicit `valid_mask`, source observation state and reason |
| `deviation_heatmap.json` | Viewer-ready candidate points with transformed BIM XYZ, signed/unsigned metres, alpha mask and explicit no-data reason; positive sign follows the reviewed BIM plane normal | 
| `viewer_handoff.json` | Versioned Shape4 interchange: relative SHA256-linked transform, scale, deviation and held-out control assets; `metric_coloring_allowed=false` until independently approved survey/BIM gates |
| `metric_bim_manifest.json` | Schema version, source/file hashes, per-output hashes and project-relative paths, run ID, software, units/frames and unresolved gates |

A point receives a deviation only if it was observed, explicitly mapped to a **reviewed finite BIM plane**, falls within that plane's finite polygon, and a candidate transform exists. Missing, occluded, out-of-FOV, ambiguous or unreviewed geometry is `valid_mask=false` and its signed/unsigned distance is **null**, not zero. Even a finite convex hull may bridge unknown holes; an operator must review actual support and missing regions. The resulting candidate distance is **not** a certified field accuracy number.

## Scientific gates (fail-closed)

- All current outputs hard-code `metric_validation="pending"`, `metric_scale_verified=false`, `registration_verified=false`, `can_feed_stage8=false`, `physical_progress_state="unknown"`. This remains true for plausible synthetic registration. No downstream Stage 8/9 consumer is changed or allowed to auto-trust them.
- External gate needed: independent field/survey controls, actual BIM/IFC revision and coordinates, trusted calibration/camera/scale, holdouts disjoint by site/capture/physical control, field-tolerance acceptance and **human review**. Real-case errors are not inferred from fitted residuals or nominal block sizes.
- `confidence.grade` is **qualitative review state**, not a statistical probability; no calibrated confidence claim is made.
- The method does not infer `not_built` from no observation; it does not create, guess or assign IFC `GlobalId` for unmatched geometric hypotheses.
- The original 19 synthetic CPU checks were **executed** successfully in GitHub Actions on PR commit `b4777723c83bd46efec09bbe2c096e613cebad3d`: 19 passed / exit 0 / 0.18 s, [specialized CI log](https://github.com/sayyedalimrj/MyCon/actions/runs/43503959957). Later changes and adversarial tests require their own exact-head runs. Real-site/GPU/independent survey/IFC extraction remain **not_executed**.

## Integration proposal

An A-owned Stage 8 opt-in reader may later read `alignment_transform.json` but **must refuse it** while `can_feed_stage8=false` or `metric_validation!="approved"`. Only a separate coordinator-reviewed evidence-verification gate may set positive status in a later **versioned, signed-off artifact**. Do not edit legacy `registration_report.json`, `bim_elements.jsonl`, or Stage 9 semantics from this workstream. A/B/C continue to consume their frozen interfaces. Reviewer-facing discrepancy overlays can use `deviation_heatmap.json` without claiming accepted scale.

To run synthetic isolated tests in a full checkout:

```bash
cd repository
PYTHONPATH=. pytest -q tests/test_civil_metric_bim_synthetic.py
```

The original module's named command was executed by the specialized CI's equivalent `python -m pytest -q` invocation on the named commit. New commits must be revalidated on their exact head. Additional adversarial tests: `pytest -q tests/test_civil_metric_bim_adversarial.py`.
