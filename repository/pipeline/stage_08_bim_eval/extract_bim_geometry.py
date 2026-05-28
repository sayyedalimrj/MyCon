"""Public wrapper for Stage 8 BIM geometry extraction.

The implementation lives in ifc_to_mesh.py so the file name remains explicit for
IfcOpenShell-specific conversion while this module provides the contract named in
the project prompt pack.
"""
from __future__ import annotations

from .ifc_to_mesh import BimExtractionResult, extract_ifc_geometry

__all__ = ["BimExtractionResult", "extract_ifc_geometry"]
