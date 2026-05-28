$ErrorActionPreference = "Stop"

$ProjectRoot = "\\wsl.localhost\Ubuntu-22.04\home\ali\projects\construction-progress-ai-bim"
$LinuxProjectRoot = "/home/ali/projects/construction-progress-ai-bim"
$Image = "colmap/colmap:latest"

# Conservative preview settings for RTX 3060 Laptop 6GB.
# برای خروجی سریع‌تر و کم‌ریسک‌تر. اگر خواستی بهترش کنیم، بعداً MaxImageSize را 1200 می‌کنیم.
$MaxImageSize = 1000
$NumSrcImages = 10
$GpuIndex = "0"

Write-Host "Pulling official CUDA-enabled COLMAP image..."
docker pull $Image

Write-Host "Cleaning old preview output..."
wsl -d Ubuntu-22.04 -u root -- bash -lc "rm -rf '$LinuxProjectRoot/data/dense_preview/site01_cuda_local' && mkdir -p '$LinuxProjectRoot/data/dense_preview/site01_cuda_local' && chown -R ali:ali '$LinuxProjectRoot/data/dense_preview'"

$Bash = @"
set -euo pipefail

echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo "=== COLMAP ==="
colmap help | head -8

echo "=== Paths ==="
test -d /workspace/data/sfm/site01/images
test -d /workspace/data/sparse_refined/site01/0

echo "=== image_undistorter ==="
colmap image_undistorter \
  --image_path /workspace/data/sfm/site01/images \
  --input_path /workspace/data/sparse_refined/site01/0 \
  --output_path /workspace/data/dense_preview/site01_cuda_local \
  --output_type COLMAP \
  --max_image_size $MaxImageSize \
  --num_patch_match_src_images $NumSrcImages

echo "=== patch_match_stereo ==="
colmap patch_match_stereo \
  --workspace_path /workspace/data/dense_preview/site01_cuda_local \
  --workspace_format COLMAP \
  --PatchMatchStereo.gpu_index $GpuIndex \
  --PatchMatchStereo.max_image_size $MaxImageSize \
  --PatchMatchStereo.window_radius 5 \
  --PatchMatchStereo.window_step 1 \
  --PatchMatchStereo.num_samples 10 \
  --PatchMatchStereo.num_iterations 3 \
  --PatchMatchStereo.geom_consistency 1 \
  --PatchMatchStereo.geom_consistency_regularizer 0.3 \
  --PatchMatchStereo.geom_consistency_max_cost 5.0 \
  --PatchMatchStereo.filter 1 \
  --PatchMatchStereo.filter_min_ncc 0.05 \
  --PatchMatchStereo.filter_min_triangulation_angle 3.0 \
  --PatchMatchStereo.filter_min_num_consistent 2 \
  --PatchMatchStereo.filter_geom_consistency_max_cost 2.0 \
  --PatchMatchStereo.cache_size 8

echo "=== stereo_fusion ==="
colmap stereo_fusion \
  --workspace_path /workspace/data/dense_preview/site01_cuda_local \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path /workspace/data/dense_preview/site01_cuda_local/fused.ply \
  --StereoFusion.max_image_size $MaxImageSize \
  --StereoFusion.min_num_pixels 3 \
  --StereoFusion.max_reproj_error 2.0 \
  --StereoFusion.max_depth_error 0.02

echo "=== Output ==="
test -s /workspace/data/dense_preview/site01_cuda_local/fused.ply
grep -m1 '^element vertex' /workspace/data/dense_preview/site01_cuda_local/fused.ply || true
ls -lh /workspace/data/dense_preview/site01_cuda_local/fused.ply
echo "LOCAL_DENSE_PREVIEW_OK fused=/workspace/data/dense_preview/site01_cuda_local/fused.ply"
"@

docker run --rm --gpus all `
  --mount type=bind,source="$ProjectRoot",target=/workspace `
  -w /workspace `
  $Image `
  bash -lc $Bash

Write-Host "Fixing WSL ownership..."
wsl -d Ubuntu-22.04 -u root -- bash -lc "chown -R ali:ali '$LinuxProjectRoot/data/dense_preview'"

Write-Host "Done. Output:"
Write-Host "$ProjectRoot\data\dense_preview\site01_cuda_local\fused.ply"
