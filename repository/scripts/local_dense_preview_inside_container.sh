#!/usr/bin/env bash
set -euo pipefail

MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-800}"
NUM_SRC_IMAGES="${NUM_SRC_IMAGES:-8}"
GPU_INDEX="${GPU_INDEX:-0}"

PROJECT="/workspace"
IMAGES="$PROJECT/data/sfm/site01/images"
SPARSE="$PROJECT/data/sparse_refined/site01/0"
OUT="$PROJECT/data/dense_preview/site01_cuda_local"
FUSED="$OUT/fused.ply"

echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true

echo "=== COLMAP ==="
command -v colmap
colmap help | head -10

echo "=== Input checks ==="
test -d "$IMAGES" || { echo "MISSING images dir: $IMAGES"; exit 10; }
test -d "$SPARSE" || { echo "MISSING sparse model dir: $SPARSE"; exit 11; }
test -s "$SPARSE/cameras.bin" || { echo "MISSING cameras.bin"; exit 12; }
test -s "$SPARSE/images.bin" || { echo "MISSING images.bin"; exit 13; }
test -s "$SPARSE/points3D.bin" || { echo "MISSING points3D.bin"; exit 14; }

echo "=== Cleaning preview workspace ==="
rm -rf "$OUT"
mkdir -p "$OUT"

UNDISTORTER_EXTRA=()
if colmap image_undistorter -h 2>&1 | grep -q -- "--num_patch_match_src_images"; then
  UNDISTORTER_EXTRA+=(--num_patch_match_src_images "$NUM_SRC_IMAGES")
fi

echo "=== image_undistorter ==="
colmap image_undistorter \
  --image_path "$IMAGES" \
  --input_path "$SPARSE" \
  --output_path "$OUT" \
  --output_type COLMAP \
  --max_image_size "$MAX_IMAGE_SIZE" \
  "${UNDISTORTER_EXTRA[@]}"

echo "=== patch_match_stereo ==="
colmap patch_match_stereo \
  --workspace_path "$OUT" \
  --workspace_format COLMAP \
  --PatchMatchStereo.gpu_index "$GPU_INDEX" \
  --PatchMatchStereo.max_image_size "$MAX_IMAGE_SIZE" \
  --PatchMatchStereo.depth_min 0.000000001 \
  --PatchMatchStereo.depth_max 100.0 \
  --PatchMatchStereo.window_radius 5 \
  --PatchMatchStereo.window_step 1 \
  --PatchMatchStereo.num_samples 8 \
  --PatchMatchStereo.num_iterations 3 \
  --PatchMatchStereo.geom_consistency 0 \
  --PatchMatchStereo.filter 1 \
  --PatchMatchStereo.filter_min_ncc 0.05 \
  --PatchMatchStereo.filter_min_triangulation_angle 2.0 \
  --PatchMatchStereo.filter_min_num_consistent 2 \
  --PatchMatchStereo.cache_size 8

echo "=== stereo_fusion ==="
colmap stereo_fusion \
  --workspace_path "$OUT" \
  --workspace_format COLMAP \
  --input_type photometric \
  --output_path "$FUSED" \
  --StereoFusion.max_image_size "$MAX_IMAGE_SIZE" \
  --StereoFusion.min_num_pixels 3 \
  --StereoFusion.max_reproj_error 3.0 \
  --StereoFusion.max_depth_error 0.05

echo "=== Output ==="
test -s "$FUSED" || { echo "FUSED_PLY_NOT_CREATED: $FUSED"; exit 20; }
grep -m1 "^element vertex" "$FUSED" || true
ls -lh "$FUSED"
echo "LOCAL_DENSE_PREVIEW_OK fused=$FUSED"
