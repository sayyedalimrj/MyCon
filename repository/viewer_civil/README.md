# MyCON Civil Viewer — independent reader and 3D evidence explorer

**Scope:** Viewer only. Neither pipeline reconstruction nor survey/metric estimation is performed here. The original R4 videos, 6999 checkpoints, prior COLMAP and geometry workstreams remain immutable. Source and Preview outputs always use **different folders**, no deletion/force operations.

**Status:** Implemented with a deterministic synthetic fixture. Prior standalone [sample ZIP in the related ChatGPT delivery] remains preserved. No real-site field validation; no GPU gsplat rendering claimed. See [research evidence](../docs/research_viewer/evidence_review.md), [handoff](../docs/parallel_handoffs/viewer.md).

## Quick start (from MyCON's \`repository/\` directory)

\`\`\`bash
python -m pip install -r requirements-research-viewer.txt
python -m viewer_civil demo ./runs/viewer-synthetic-20261114/input --run-id viewer-synthetic-20261114
python -m viewer_civil render ./runs/viewer-synthetic-20261114/input ./runs/viewer-synthetic-20261114/preview
# Open runs/viewer-synthetic-20261114/preview/index.html in Chromium/Firefox (offline)
python -m pytest -q tests/test_research_viewer_bundle.py
# Optional browser/image/video capture requires playwright + browser + ffmpeg:
python -m pip install playwright
python -m playwright install chromium
python -m viewer_civil capture ./runs/viewer-synthetic-20261114/preview/index.html ./runs/viewer-synthetic-20261114/figure.png
python -m viewer_civil video ./runs/viewer-synthetic-20261114/preview/index.html ./runs/viewer-synthetic-20261114/orbit.mp4 --seconds 4
\`\`\`

All target directories must be new. **No \`--force\`, no overwrite and no training**. The preview writes a local Plotly JS bundle, local CSS/JS, SHA256-verified PNG/JPEG frame copies and \`viewer_build.json\`. Use a local browser, not an external CDN. PNG and MP4 are actual rendered browser captures when the optional capture step is executed. The synthetic input frames are deliberately synthetic even though the browser screenshots are genuine; label them **SYNTHETIC / NOT FIELD EVIDENCE** in the thesis.

## Functionality

- Native 3D Plotly/WebGL scene: rotate, zoom, camera frusta oriented by 4×4 camera-to-world pose, paths, scene mesh, optional point samples, finite plane patches, candidate edges/corners, Unknown groups.
- Canvas projected 3D fallback when WebGL unavailable or Plotly render fails; pointer rotation, shift-drag pan, wheel zoom, and clickable camera marker. Fallback is not GPU-accelerated or photorealistic GS rendering.
- Select camera in list or 3D marker; display the exact independently hashed source frame, image dimensions and camera origin.
- Mode selector: **confidence / coverage / neutral**, distinct layers and Unknown. **Coverage is never construction completion**.
- Shape 4 BIM reference wireframe only when input explicitly provides it; mark unregistered when unverified.
- Metric deviation disabled until independent scale, registration and independent surveyed controls all validate; for non-metric geometry never show a numeric difference.

## Strict reader contract

\`viewer_manifest.json\` must use \`schema_version="mycon.viewer_bundle.v1"\`, safe \`run_id\`, \`coordinate_frame\`, \`units\`, explicit source \`provenance\`, \`quality\`, and SHA256 file receipts. Required read-only JSON roles: \`camera_poses\`, \`frame_manifest\`, \`planes\`, \`edges\`, \`corners\`, \`mesh\`, \`alignment_result\`. Optional validated role: \`point_cloud\` as JSON \`points: [[x,y,z],...]\`; no automatic conversion from a raw GS point-center PLY to surveyed geometry.

Each role's receipt includes relative \`path\`, \`sha256\`, \`schema_version\`, \`coordinate_frame\`, \`units\`, \`quality\` and \`provenance\`. Role documents repeat these fields. Frame manifest contains \`frame_id\`, safe relative raster \`path\`, SHA256, quality and provenance. Camera poses use unique \`camera_id\`, linked \`frame_id\`, rigid \`camera_to_world\` (row-major 4×4), \`extrinsic_convention: camera_to_world\`, **+Z camera optical forward, +Y image down**, \`fx/fy/cx/cy/width/height\`, quality and provenance. Degenerate/non-rigid poses and source frame/hash mismatch reject the entire bundle; no guessing coordinate axes. **Separate adapters** must explicitly convert COLMAP coordinate conventions and splat/geometry units before use.

For the \`mycon.comparison_ready_geometry.v1\` producer from team A, integration is a **future optional producer-independent conversion**. Consumer must not modify A's artifacts. A missing camera/pose must fail or be explicitly marked absent, never synthesized as actual photographed evidence.

Use:
\`\`\`python
from viewer_civil import load_bundle, build_preview
b = load_bundle("runs/<run_id>/input")
print(b["assets"]["camera_poses"]["cameras"])
print(b["assets"]["frame_manifest"]["frames"])
print(b["deviation_enabled"], b["deviation_reason"])
build_preview("runs/<run_id>/input", "runs/<new_viewer_run_id>/preview")
\`\`\`

**Numerical deviation gate:** \`synthetic != true\`; BOTH manifest and alignment assert \`metric_scale_verified=true\` and \`registration_verified=true\`; alignment asserts \`independent_control_verified=true\`; frame and units in metres; \`mesh.deviations\` contains \`units=m\`, independent provenance, an actual per-face list matching all faces, and \`independently_verified=true\`. Otherwise disabled. Independent controls and evidence must be actually reviewed; a tampered JSON with falsely asserted booleans is **not** a trusted survey; cryptographic authenticity/reviewer signatures are outside v1. Never publish metric claims merely because a viewer gate flag is true.

## Colab / Google Drive (coordinator owns final notebook)

After the Coordinator mounts Drive and installs viewer deps, read the source Drive bundle **without editing it**, and output to a fresh \`MyDrive/MyCon/viewer/<unique-viewer-run-id>/\` folder. Preview assets and receipts persist if copied to Drive. A free Colab Runtime may disconnect or lose local browser/GPU state at any point: rerun the viewer command to a NEW output folder if incomplete, do not overwrite or restart reconstruction/training. Notebook UI cell edits are exclusively the coordinator's ownership. Static offline preview needs no long-lived Colab server; Viser/nerfview remains opt-in and is **not** a tested runtime dependency.

## Integrity and limitations

- WebGL GPU rendering capability is browser-dependent; Canvas fallback is intended for review only.
- Browser screenshots are not independent measurement evidence.
- Preview JSON is bounded to 25 MB per role; very large PLY / .splat assets need a future streamed visualization adapter with versioned provenance; no unsupported Gaussian-Ply-as-built interpretation.
- Source file checks are SHA256/content integrity, not signatures/authentication. \`frame_manifest\` SHA receipts and source data should be stored under controlled immutable permissions.
- No reconstruction quality benchmarks, real dimensions, live source captures, certified IFC identity or physical progress are claimed in this module.
