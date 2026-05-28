"""Stage 6: conditional DA3 depth assistance.

This stage is intentionally optional and conditional. It does not download model
weights or run a built-in DA3 model by default. Instead it assesses Stage 5 dense
coverage, prepares a reproducible depth-assistance contract, aligns external or
precomputed DA3 depth maps to the COLMAP reconstruction frame, and optionally
fuses aligned depths into an assisted point cloud.
"""
