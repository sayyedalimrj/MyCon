"""Stage 3: COLMAP sparse SfM using ALIKED + LightGlue.

This package consumes Stage 2 keyframes/manifest and produces a COLMAP database
plus a sparse reconstruction under the repository file contract.
"""

__all__ = ["run_sparse"]
