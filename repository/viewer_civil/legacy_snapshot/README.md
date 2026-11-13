# MyCON Civil Viewer — independent review package

**Status:** Locally tested standalone prototype, **not** integrated into MyCON, not GPU/real-capture validated. This package deliberately does not modify any Reconstruction/Geometry/Metric/Notebook paths. Synthetic geometry and images are illustrations, not scientific or construction evidence.

## Quick start

From the package root, Python 3.10+:

```bash
python -m pip install -e '.[test]'
python -m mycon_viewer.cli demo ./sample_run --run-id viewer-synthetic-local1
python -m mycon_viewer.cli render ./sample_run ./viewer_outputs/local1
# Open viewer_outputs/local1/index.html in any modern WebGL-capable browser
python -m pytest tests -q
# Optional (requires Chromium and Playwright):
python -m mycon_viewer.cli capture ./viewer_outputs/local1/index.html ./viewer_outputs/local1/screenshot.png
python -m mycon_viewer.cli video ./viewer_outputs/local1/index.html ./viewer_outputs/local1/demo.mp4 --seconds 6
```

The `demo`, `render` and `capture` operations **refuse to overwrite an existing output path**. All reads of source data are read-only. Video capture uses ffmpeg and headless Chromium; no GPU training.

## What it displays

Offline 3D Plotly/WebGL room viewer (rotatable, zoomable), mesh candidate surfaces, finite planes, confidence-classified structural edges/corners, missing/unknown features, camera pose glyphs and pinhole frusta, camera path, clickable camera gallery and linked frame, individual layer toggles, independent confidence/coverage/neutral display modes, optional Shape 4 BIM overlay. It contains no point-cloud reconstruction or model training logic. `plotly.min.js` is copied locally into each preview for offline operation; **Viser/gsplat is optional and not installed nor exercised**. A future Viser backend must use the same validated `load_bundle()` result and never rewrite producer files.

## Canonical input API

```python
from mycon_viewer import load_bundle, build_preview
bundle = load_bundle('/path/to/read_only_bundle')
print(bundle['assets']['camera_poses']['cameras'])
print(bundle['assets']['frame_manifest']['frames'])
print(bundle['assets']['planes']['planes'])
print(bundle['assets']['edges']['edges'])
print(bundle['assets']['corners']['corners'])
print(bundle['assets']['mesh']['vertices'])
print(bundle['assets']['alignment_result'])
print(bundle['deviation_enabled'], bundle['deviation_reason'])
build_preview('/path/to/read_only_bundle','/path/to/new_output_folder')
```

The root manifest is `viewer_manifest.json` with `schema_version: mycon.viewer_bundle.v1`. Every input must have a role receipt: `path` (safe relative path), `sha256`, `schema_version`, `units`, `coordinate_frame`, `quality` and `provenance`. All input documents repeat schema, units, coordinate frame, quality and provenance. Camera extrinsics are explicit `camera_to_world` 4×4 row-major (+Z forward; +Y image-down), calibrated intrinsics `fx/fy/cx/cy/width/height` and stable `camera_id/frame_id`. All frames are separately hashed and strictly linked. Camera poses can be imported via a future producer-side adapter, not guessed from video.

Unknown confidence is `null`, not 0, and visual coverage is **never** reported as physical construction completion. Units may be `arbitrary` until an independent scale control is verified. Geometry surface observations and physically executed quantity are independent.

For real inputs, adapters should translate A's `mycon.comparison_ready_geometry.v1` output into this viewer schema **outside** A's owned pipeline: use `comparison_ready.json` receipts to map canonical clean geometry and planes/edges/corners; expose imagery/poses only when available and calibrated. Missing cameras/images produce explicit conversion failure, not synthetic fallback. Stable IDs and provenance must be preserved.

### Shape 4 / BIM and deviation gate

A Shape 4 overlay may be displayed when explicitly supplied as `mesh.bim_overlay` but is labeled **unregistered** until verified. Deviation heatmap requires all of:

- `synthetic=false`, manifest and alignment both `metric_scale_verified=true` and `registration_verified=true`;
- `alignment_result.independent_control_verified=true` and matching coordinate frame;
- coordinates and distances in **meters**, not reconstruction units;
- independently verified `mesh.deviations.per_face_m` (one per mesh facet) with provenance.

Otherwise the heatmap stays disabled, shows reason and **never generates or labels numerical deviation**. Showing unregistered BIM alongside a scene is for orientation only, not accuracy, progress, or as-built comparison. A physically meaningful tolerance legend must be user-approved and data-specific; example display bands in the source are *visual hints*, not contractual acceptance limits.

## Standard thesis outputs

`index.html` + local `plotly.min.js` and assets are the reproducible, interactable preview; `viewer_build.json` includes the source manifest SHA256 and metric gate reason. Chromium screenshots are PNG (1440×900); optional narrated-free MP4 export is a real render of the rotating viewer at 8 fps, 1440×900 and x264 yuv420p. These captures are **screen captures of synthetic data** in the provided example and need explicit “SYNTHETIC” captions in the thesis. Provenance is the root manifest, data hashes, build receipt, and screenshot/video hashes computed at handoff. PNG output has no embedded scale bar because no validated metric geometry exists.

## Colab / Drive integration (lead-owned notebook cells)

**No notebook or `repository/colab/**` files are modified**. In a fresh Colab session after mounting Drive: `pip install -e './MyCON_Viewer_Standalone'`, then `build_preview('/content/drive/MyDrive/MyCon/<run_id>/viewer_bundle','/content/drive/MyDrive/MyCon/viewer/<viewer_run_id>')`. Open/download `index.html` together with its sibling JS and `frames/`. Drive subdirectories and `viewer_run_id` must be unique per workstream. Colab's ephemeral runtime may disconnect/reset at any time; viewer preview can survive if output was already written to Drive. Persist small receipts and previews first, and do not put 3DGS checkpoints or raw videos into static preview. If a rendering job is interrupted, create a **new** output dir and rerun without deleting the old one. The Integration Lead—not this workstream—adds final Notebook UI cells.

## Validation scope and outstanding items

- Included: SHA256/content checks, unsafe-relative-path rejection, unit/frame checks, source-frame selection and basic layer toggle browser checks, synthetic local image tests.
- Not validated: project MyCon production files, Google Drive real recordings/checkpoints, GPU gsplat model rendering, physical scale, surveyed BIM registration, calibrated confidence and publication-ready error bars.
- Governance: existing Integration contract assigns A/B/C only. **Owner-approved Viewer amendment and exact directory/branch are required before committing**. No PR, merge or ownership-crossing edits should be performed until approved.
