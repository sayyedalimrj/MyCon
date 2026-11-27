# MyCON Work integration pilot

The consolidation branch begins at the R4 handoff with all six reviewed PR heads present as Git parents. Use the exact candidate commit or a detached checkout when repeating a result. The historical capture and checkpoints remain input-only. Keep work outputs in an access-controlled project directory outside the source checkout.

## Run the pilot

Install Python 3.11, headless-capable COLMAP, and the dependencies in `requirements-work-pilot.txt`. On a Debian or Ubuntu workstation the distribution CPU COLMAP can run sparse reconstruction. For Debian 13 without system install permission, the Work run used the distribution `colmap_3.10-2_amd64.deb` extracted with `dpkg-deb -x` alongside Ceres, glog, gflags and SPQR, with the extracted library path supplied via `LD_LIBRARY_PATH`. Run `colmap -h` first and keep the version output.

From the Git checkout's `repository/` directory:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-work-pilot.txt
QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 \
python -m reconstruction.consolidated_pilot \
  --run-dir /absolute/work/runs/recon-site01-001 \
  --review-id pilot-001 --video /absolute/input/recording.mp4 \
  --capture-id site01-visit01 --geometry-threshold 0.075
```

Use the main [Colab notebook](../../reconstruction/notebooks/MyCON_Consolidated_Pilot.ipynb) with your copy of the source recording. Set the video path in its configuration cell and run all cells. The example threshold is in arbitrary COLMAP units; preregister a data-independent threshold rule and a sensitivity study for scientific results. Optional `--test-video` and `--test-capture-id` accept a distinct view capture of the **same scene**. A second construction environment is a separate project/run. Optional `--expected-video-sha256` pins the selected train recording. Optional `--attempt-dense` tries CUDA PatchMatch via the pinned PyCOLMAP CUDA backend, with the reasons for any block in the same-run receipt.

Open `offline-<review-id>/index.html` with the whole directory together. Plotly and source images ship locally in that directory. Registered camera labels link to the verified source photographs. The pose export uses COLMAP/OpenCV world-to-camera; the viewer shows its rigid inverse without a handedness or axis flip. The viewer labels this frozen train frame as unscaled. Candidate planes, edges and hypothetical corners stay Unknown for engineering classification.

The R stage writes `frame_selection.csv`, `dataset_manifest.json`, `colmap/colmap.log`, `camera_poses.json`, `registration_quality.json`, and `coverage_report.json`. The R product stage writes evaluated sparse PLY, undistortion and train-only GS staging receipts in `same_run_products/`. A writes the sparse candidate evidence, sampled source-image projection and overlays underneath `runs/geom-civil-r-frame-.../`. The top-level `pilot-<review-id>.json` records stage status and asset identities. Tests for the candidate branch run via the [Work Integration workflow](../../../.github/workflows/mycon-work-consolidation.yml).

## Recovery and field execution

| Condition | Recovery |
| --- | --- |
| Video SHA mismatch or low overlap | Verify the supplied source version, review selection and coverage logs, and start a new run ID with documented settings. |
| COLMAP partial, missing final execution receipt | Keep the original logs and directory. Retry under a new run ID after correcting runtime/dependencies. |
| SfM completed, A/V output incomplete | Use the same run directory, omit the video selection arguments and supply a new review ID. The frozen dataset and its camera/model identities are rechecked. |
| MVS outputs incomplete or absent | Keep the sparse pilot. Use a new run directory for a full dense attempt on a compatible CUDA machine. Preserve partial products as logs. |
| IFC inventory | Run `python -m civil_metric_bim.ifc_inventory --help` and extract the exact IFC revision with IfcOpenShell. Review source units, placement and GlobalIds independently. |
| Metric trial | Run `python -m civil_metric_bim.field_run --help` with the native comparison-ready manifest, source-frame-bound survey, three independent fitting spans, two reserved check spans, and engineering tolerances. IFC requires reviewed fit and control matches for the same source geometry and exact revision. |
| Verified visibility and schedule | Run Stage 8's `bim_geometry_evidence` with independent controls, approved Sim(3), actual IFC mesh/IDs and calibrated camera evidence. Supply the verified Stage 8 manifest plus signed quantities, approved BOQ/WBS weights and baseline to `pipeline.stage_09_progress.element_status_export` for Stage 9/11. |
| A/D have only candidate receipts | Audit path, digest, units, and source identity. Preserve Unknown visibility/quantities. Approval belongs to native Stage 8 inputs, with separate source provenance and IFC revision. |

The managed branch contains the original product entry points and specialized handoffs in [docs/parallel_handoffs](../parallel_handoffs/). Its pilot does not certify a model as metric or write back to the databases of field operations.

## Capability inventory

| Stream | Preserved or connected capabilities | Evidence status |
| --- | --- | --- |
| R | Capture intake, frame QA and duplicate rejection, holdout controls, sparse pose and coverage reports, CPU COLMAP, undistortion, sparse and optional CUDA dense exports, read-only historical/checkpoint audits, train-only GS staging and held-out render scoring | CPU integration. Consult the pilot receipt for actual output and missing registrations. |
| A | Sparse/dense candidate extraction, finite plane/edge/corner review, point IDs, image-track reprojection, stability, comparison-ready diagnostics and nonmetric claims | Candidate geometry. Independent semantic and geometry GT required. |
| V | Offline camera/image inspector, frusta, cloud, Shape 3 candidates, Canvas/WebGL paths, gated Shape 4 and deviation | Diagnostic viewer. Shape 4 needs a reviewed metric/BIM result. |
| D | IFC inventory/identity/units, no-BIM span scaling, Sim(3), fitting and holdout checks, finite deviation and heatmap assets | Native parser and synthetic controls can be tested independently. Approval requires actual field evidence. |
| C | Paired view/geometry metrics, manifest checks, blinded held-out scoring, split/leakage checks, evidence export and ablation plan | Internal SfM diagnostics are available; independent paired/field evaluation pending. |
| B | Stage 9 construction vs visibility, conservative Unknown, signed quantities, BOQ/WBS, Stage 11 planned/actual variance and dashboard, read-only A/D receipt audits | Synthetic/CPU with native IFC identity failure handled. Real multidate progress still requires signed input. |

## Scientific release criteria

The original proposal sets four goals: rapid as-built geometry, construction-element semantics with limited labels, registration/deviation against BIM, and management progress against WBS and schedule. Log denominators and failure cases at each stage. Use at least two distinct construction environments for the first pilot, with independently measured data at each site and at least two dates for progress. Reserve held-out views, physical controls and labels before comparing methods. Reserve complete sites for a stronger cross-site claim. Record model versions, what was tuned, seeds, hardware, wall time and peak device memory.

| Comparison | Primary outcomes and controls |
| --- | --- |
| R/Gaussian vs COLMAP photogrammetry | Same input train views and registered held-out IDs; PSNR, SSIM and LPIPS only on matched ground-truth pixels with registered/failure denominator. Count held-out pose estimation separately. Record SIFT/mapper or render baseline, time and peak GPU memory. |
| Geometry and semantic labels | Independent measured reference, common valid support, fixed thresholds; accuracy, completeness, Chamfer L1, F1 and per-class semantic confusion with occluded/Unknown counts. Review false features and extrapolated corners. |
| Metric/IFC | Independent physical controls withheld from fit, explicit tolerances before trial, identity-bound mesh and BIM revision, holdout RMSE/P95 and deviation only where observed. BIM is a design reference; use surveyed as-built truth for accuracy. |
| Progress | Same site and exact IFC/BOQ/WBS revision across dates, signed independent quantities and schedule baseline, per-element/activity MAE and variance in percentage points. Preserve Unknown and distinguish signed absence from occlusion. |
| Predeclared ablations | Frame selection and embargo; sparse/dense support; geometry plus image evidence vs geometry-only; semantics with and without limited labels; fit-only metric vs withheld control; occlusion-aware vs naive policy. Keep failure rows and coverage in the paired comparison. |

Baseline methods are grounded in [Schönberger and Frahm, 2016, DOI 10.1109/CVPR.2016.445](https://doi.org/10.1109/CVPR.2016.445), [Kerbl et al., 2023, DOI 10.1145/3592433](https://doi.org/10.1145/3592433), and [Bosché et al., 2015, DOI 10.1016/j.autcon.2014.05.014](https://doi.org/10.1016/j.autcon.2014.05.014). The per-stream reviews retain method/code/license and executable example provenance. Record each case in the validation stream before exposing its held-out checks. No paired accuracy/promised performance threshold applies to the exploratory room-video pilot.
