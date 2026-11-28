# MyCON: original long-capture reconstruction quality (priority reset 2026-11-28)

**Primary acceptance is visual/geometric quality on the original full recording, not unit-test counts.** Do not substitute the unrelated 34.92 s pilot.

## Historical inputs and evidence to recover privately
- Original Recorder archive `MyDrive/MyCon_R4/captures/20261013_064010_793_MYCON.zip`, original ~95.93 s video and sidecars. Resolve exact member names, PTS, orientation, ARCore/IMU/intrinsics and SHA from original archive. Historical `full_video_upright.mp4` ~96 s; establish identity vs source via receipt. Preserve both originals.
- Successful ~10.2 s / 102 selected views short reference: historical DA3 PLY 152353 points; COLMAP 102/102 reported. Recover exact short clip/frame roster, original notebook, model/PLY, camera poses, rendering settings, visual comparison. Do not assume the clean output covers the whole scene or is surveyed.
- Historical full DA3 run `worksite_v480_934414c04ed4ef8a`: 384 frames, 94 twelve-frame windows overlap 8; 82 accepted; 5 scale jumps, 4 low baseline, 3 trajectory inconsistencies. Full candidate PLY 152779; nonmetric. Original per-image grid fell 48x48→25x25 with fixed budget; this is a visualization sampling issue, not proof of learned-model degradation. Historical COLMAP resume incorrectly accepted empty geometric match table; repair without overwriting old DB. Cached windows and SIFT are read-only.
- Historical distinct GS6999 run: preserve original model and free-view / trajectory renders. Do not compare metrics across unmatched input sets.
- Historical architecture ADR: Recorder timestamp/orientation/K/pose QA before geometry. MapAnything independent image-only vs reviewed pose-guided, then robust common-frame / pose graph; COLMAP baseline and DA3-Streaming fallback. No forced join of inconsistent segments. Treat pure rotation as coverage, not depth certainty.

## New implementation (preparation only, not evidence of improved reconstruction)
The current consolidation branch now has opt-in `--long-video` (8 fps target, 1200-frame cap per source, full decode scan and disk-backed lossless accepted candidates). Defaults for ordinary short pilot unchanged. Actual sampling follows the integer decoder stride, not exact native PTS. Candidate spool can consume significant disk; use local scratch with sufficient free space, outputs are new and private. Temporal coverage report groups selected/registered views into 10 s windows and logs gaps including ends. A single-capture test block still intentionally removes training views and can break continuity; use an independent same-scene test capture for full-span training. No evidence of a full reconstructed long recording yet. The new notebook pins implementation 59788186d5926c967d58d9a40f69878c0717f5d0. Existing renderer, DA3 grid, Recorder ingest and dense fusion are NOT changed by this preparation.

From repository/ after installing the documented environment, using verified local copies:
```bash
python -m reconstruction.consolidated_pilot --long-video \
  --run-dir /private/scratch/recon-full96-001 --review-id full96-001 \
  --video /private/input/full_video_upright.mp4 --capture-id recorder-full96 \
  --test-video /private/input/independent_same_scene.mp4 --test-capture-id independent-visit \
  --sample-hz 8 --max-frames 1200 --geometry-threshold 0.075
```
If independent capture is unavailable, omit BOTH test arguments for exploratory temporal holdout ONLY, not an accepted full-span result. The pilot has no heavy trainer. Do not start new heavy GPU training without approval.

## Work next action: reconstruct the right capture, not more generic tests
1. Retrieve private short and full outputs and show actual side-by-side crops at the same physical viewpoints and output resolution. Include orbit/free-view away from original path; distinguish color/render vs actual point cloud/surface detail.
2. Reproduce short reference with exact original pipeline/config first. Next process entire original 96 s capture with verified Recorder timeline, K, orientation and poses. Diagnose camera motion/overlap/rotation-only/loop closure, matched graph per segment, source frame identity. Preserve ceiling/floor/corners; don't discard bridge frames via aggressive duplicate rejection. Compare 4/8/10 fps and adaptive keyframes on TRAIN/development data; freeze independent test separately. Don't rely on arbitrary global frame cap.
3. Reuse historical depth window and SIFT cache on copies only if exact source and identity match. Compare original DA3 export at fixed per-view detail vs old global cap; render same local area with unchanged camera/size to isolate display decimation from reconstruction defects. Use full-resolution assets with spatial tiled/LOD streaming for viewer, don't globally decimate geometry as duration increases.
4. Diagnose long-range drift with overlapping 10–20 s segments plus geometrically verified loops/pose graph; no independent local coordinate systems forcibly merged. Evaluate independent MapAnything image-only vs pose-guided with validated Recorder prior; compare existing COLMAP and DA3-Streaming. Verify actual dependencies and run resources before heavy inference. Preserve geometry under camera rotation, unknown where parallax absent.
5. Connect the real dense output to A/V after same-run coordinate verification. Current pilot is sparse-first; do not claim dense integrated. Defer semantic/BIM/progress feature expansion until the original video's output is visually/geometrically acceptable. Preserve all existing capabilities.
6. Acceptance package: original source identity and span, coverage for every 10 s, actual short-vs-long equal-resolution images and free-view clips, full PLY/mesh (not proxy only), camera trajectory and visible gaps, time/memory, exact code/settings and raw logs. Compare heldout only when same valid reference; visual review before proposing user-approved heavy training. No target success asserted based on tests or point count. User reviews actual output quality.

Existing private outputs were not recovered as files in this coordinator turn; the historical details above derive from the saved November 2/3 reports and prior conversations. The long profile was committed but not run on the actual recording or GPU; Work must verify in the runtime. No existing PR or main changes.
