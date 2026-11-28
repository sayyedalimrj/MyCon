# Recorder full-capture reconstruction and visual review

This work addresses the original Recorder capture `20261013_064010_793_MYCON.zip`.
The 10.2 s short section and the long result use the same physical starting area. The previous 34.9 s pilot is outside this comparison.

## Inputs and reproducible commands

Use a clean output directory. The original ZIP and old MyCon_R4 assets are read-only.
Extract the Recorder ZIP with an archive reader, after confirming its member list and digest. The original video is `arcore_recording.mp4`.
This capture uses the verified 90° clockwise pixel orientation recorded in the historical orientation and upright receipts.

Run from `repository/`. The Work environment used Python 3.12, PyCOLMAP 4.2.1 CPU, FFmpeg/FFprobe, NumPy, SciPy, OpenCV headless and Trimesh.
Use the original v4.8 cache inputs and keep `prediction.npz` (and `complete.json` if available) in their numbered window directories.

```bash
python -m pip install 'numpy==1.26.4' 'pycolmap==4.2.1' \
  'opencv-python-headless==4.11.0.86' scipy pillow trimesh

# Omit --historical-roster for a new Recorder video.
# Add --select-only to inspect the adaptive frame/overlap plan before mapping.
python -m reconstruction.long_capture --recorder /private/recorder --rotation 90 \
  --video-sha256 b6f8e01a258e0f1dcaa98b536fa835099e1b43f872a5fd2acd1018693f078518 \
  --historical-roster /private/long/input_manifest.json \
  --out /private/new/full_sfm --feature-size 1000

python -m reconstruction.da3_replay --input /private/short/input_manifest.json \
  --alignment /private/short/alignment.json --poses /private/short/poses.json \
  --windows /private/short/windows --out /private/new/short_replay

python -m reconstruction.da3_replay --input /private/long/input_manifest.json \
  --alignment /private/long/alignment.json --poses /private/long/poses.json \
  --windows /private/long/windows --short-input /private/short/input_manifest.json \
  --short-alignment /private/short/alignment.json --short-poses /private/short/poses.json \
  --short-windows /private/short/windows --out /private/new/long_replay --grid 48

python -m reconstruction.da3_pose_fusion --input /private/long/input_manifest.json \
  --alignment /private/long/alignment.json --poses /private/long/poses.json \
  --windows /private/long/windows --sfm /private/new/full_sfm \
  --short-binding /private/new/long_replay/short_binding.json \
  --short-ply /private/new/short_replay/legacy_replayed.ply --out /private/new/global_fusion
```

The short and full archived DA3 models replay from their intended stored predictions.
New videos use motion-adaptive sampling on native presentation timestamps. Raw recorder calibration is rotated with the pixels. Phone pose proposals are optional and must pass IMU-vs-pose rate checks. SIFT geometry verification works independently of the phone pose.
Nearby frames preserve local overlap and visually similar nonlocal pairs enter the global bundle adjustment after geometric verification. Planar and panoramic pairs alone cannot measure depth.
Each output directory is create-only. A bad or empty geometric match graph blocks mapping. Resume on a new path or explicitly recover a checked database outside the original run.

The replay stage retains a fixed per-frame spatial grid at any duration. The old global cap is retained as a separate reproducibility output. The old short PLY remains the protected reference. Registration uses held-out visually matched views before rendering with shared coordinates. The pose-fusion stage uses verified SfM camera/image identity and held-out camera positions to gate a global candidate. A failed control or seam gate keeps that candidate out of the viewer. Sparse and learned clouds stay in the systems recorded in their receipts. Neither ARCore session units nor DA3/COLMAP scale is surveyed.

For three-way visual review, run:

```bash
python -m reconstruction.pointcloud_compare \
 --model "Short reference=/private/new/short_replay/legacy_replayed.ply" \
 --model "Old long=/private/new/long_replay/legacy_common_frame.ply" \
 --model "Fixed long=/private/new/long_replay/reference_locked_extension.ply" \
 --reference-poses /private/short/poses.json --out /private/new/visual
```

Open `Viewer.html` directly. Drag for synchronized free orbit, scroll to zoom, and use **Overview** to inspect scene extent. The PNGs and MP4 use shared cameras and image dimensions. The exported point clouds are colored candidates; no watertight surface or survey accuracy is inferred from fill or point count.

## Method scope

- COLMAP provides an independently tracked, globally optimized sparse camera trajectory and a verified visual pair graph. View count and graph edges do not prove correct surfaces.
- The historical DA3 worker cached 336-pixel inference with 12 views per window and 8 shared views. The old fixed total point budget lowered the spatial grid from 48×48 to 25×25. Replaying 48×48 enriches the existing predictions. It cannot restore detail the model never predicted or correct ghosting by itself.
- Official DA3 Streaming supports overlapping chunks and Sim(3) loop optimization. Its published default is a separate profile; it is not represented by this older R4 cache. MapAnything can accept image inputs and optional camera priors, which permits a matched image-only vs validated-pose comparison. Each would require a new inference run and separate receipts before its geometry is merged. [DA3 Streaming configuration](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/configs/base_config.yaml) · [MapAnything inference](https://github.com/facebookresearch/map-anything/blob/main/README.md) · [COLMAP tutorial](https://colmap.github.io/tutorial)
- GS6999 was checked via its existing trajectory render and canonical model receipt. It is a separate trained result, not a substitute for same-frame geometry review. No new GS training was started.

## Actual run on the original Recorder video

- Original ZIP SHA-256 `811c5a1323e45e2a4ef635eeda9f4335be06ef01ccb74ed3b431328400f862b6`. The encoded MP4 SHA-256 matched the historical upright receipt. The 2,875 native PTS ran from 0.148467 to 96.073611 s. The Recorder trace also covered a 95.93 s interval. Its raw gyro-vs-ARCore angular-rate correlation was 0.082, below the 0.45 prior gate. The calibrated camera intrinsics and 90° pixel rotation were used; phone position/orientation did not constrain bundle adjustment.
- The 24-window short model replayed byte-for-byte at 48×48 and 152,353 points. The 94-window historical long PLY also replayed byte-for-byte, at 25×25 and 152,779 points. The same 94 predictions produced 570,803 points at 48×48; that isolated experiment changed sampling, with the original camera/window transforms preserved.
- The fresh full-video COLMAP run used the 384 historical indices decoded from the verified original, plus a new independently computed match database. It registered 383 frames in one sparse model, with coverage in every 10-second bin. It verified 2,910 visual pairs, including 353 nonlocal pairs, with mixed geometric configurations. The exported model contained 97,391 sparse points. A receipt-serialization issue occurred after geometry export; the receipt was recovered from the exported model and the integer conversion was corrected. A separate full-video adaptive selection run selected 791 frames. Its matching/mapping was outside this exploratory comparison.
- The visual short/long binding used 30 fitting and 10 reserved view pairs. The global camera-constrained DA3 solve used 321 fitting cameras, 62 check cameras, and 185 shared-depth window edges. Its check-camera median residual fell from 0.134 to 0.0061 in short-model units. The cross-window P95 rose from 0.0116 to 0.0221. This is a tradeoff between consistency to the global camera path and the historical local window joins. The candidate includes 152,353 protected short vertices, whose positions and RGB exactly match the original reference when interpreted as float32/RGB8, and 432,382 new vertices.
- Identical free views show a cleaner entrance and floor than the historical long model. Near 72 s the late wall and wall junction are more continuous. Near 92 s the model recovers a wall missing from the old rendered view; dotted bands and poor texture fidelity remain noticeable. These observations do not establish uniform short-reference surface detail throughout the entire capture. The offline browser ran WebGL2 software rendering and a real pointer drag changed the shared orbit and the rendered image without page errors.
