
# Stage 8 Visual Anchor Workflow

This helper converts image-based anchor observations into scan-space 3D anchor coordinates.

## Why it exists

In real site video, exact scan coordinates are usually not known in advance. The practical workflow is:

1. Define benchmark/anchor points in BIM coordinates.
2. Observe the same anchors in two or more posed video frames.
3. Use COLMAP camera poses to triangulate each anchor into scan/SfM space.
4. Merge the triangulated scan coordinates into the metric anchor CSV.
5. Run Stage 8 metric alignment and BIM registration.

## Marker path

If ArUco, AprilTag, QR, barcode, or a printed benchmark target exists, use its detected center/corners as high-confidence visual observations.

## No-marker path

If no marker exists, use stable structural anchors:

- column edges
- wall corners
- slab/wall intersections
- opening corners
- MEP sleeve or embedded object corners if stable

`scripts/detect_structural_edge_candidates.py` creates line candidates from images. A human or future VLM/edge-matcher can assign `anchor_id` to the relevant candidate.

## Required observation CSV

`data/bim/design/visual_anchor_observations.csv`

```csv
anchor_id,image_name,u_px,v_px,confidence,method,notes
A,img001.jpg,523.2,412.7,0.9,manual_column_edge,corner of column A
A,img018.jpg,499.1,388.5,0.9,manual_column_edge,corner of column A
Commands
Generate edge candidates:

python3 scripts/detect_structural_edge_candidates.py
Triangulate and merge:

python3 scripts/merge_visual_anchor_observations.py \
  --cameras-txt data/sparse_refined/site01/0/cameras.txt \
  --images-txt data/sparse_refined/site01/0/images.txt \
  --observations-csv data/bim/design/visual_anchor_observations.csv \
  --metric-anchors-template data/bim/design/metric_anchors.csv \
  --picked-output data/bim/design/picked_scan_anchors.csv \
  --merged-output data/bim/design/metric_anchors_working.csv
Then configure metric alignment to use metric_anchors_working.csv.

Boundary
This is not visual progress estimation. It only creates better scan-to-BIM initialization anchors.


## Visual anchor quality gate

Visual anchor triangulation must be treated as measurement evidence, not truth by default.

Each triangulated anchor should be checked for:

- minimum number of observations
- minimum ray angle / baseline
- reprojection error
- finite 3D point
- accepted/rejected quality decision

The helper module is:

```text
pipeline/stage_08_bim_eval/visual_anchor_quality.py
```

Rejected anchors must not be used as high-confidence metric alignment anchors.
