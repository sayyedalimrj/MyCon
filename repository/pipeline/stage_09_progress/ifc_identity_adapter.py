"""Read-only IFC GlobalId inventory adapter for audited 4D BIM work packages.

This is a checksum-bound index with optional authoritative IFC parser validation.
When IfcOpenShell is missing or the IFC cannot be parsed, identities fail closed.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "mycon.ifc_identity_index.v1"
IFC_GUID = re.compile(r"^[0-9A-Za-z_$]{22}$")
SHA = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class IFCIndex:
    valid: bool
    reason_codes: tuple[str, ...]
    ifc_sha256: str | None
    index_sha256: str | None
    elements: Mapping[str, dict[str, Any]]


def read_ifc_identity_index(manifest_path: Path | str | None, *,
                            project_root: Path | str,
                            expected_ifc_sha256: str | None = None) -> IFCIndex:
    if manifest_path is None:
        return IFCIndex(False, ("ifc_index_not_supplied",), None, None, {})
    errors: list[str] = []
    index_hash = None
    ifc_hash = None
    try:
        root = Path(project_root).resolve()
        manifest = Path(manifest_path).resolve()
        if not manifest.is_relative_to(root) or not manifest.is_file() or manifest.is_symlink():
            raise ValueError("index_outside_project_or_missing")
        raw = manifest.read_bytes()
        index_hash = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA:
            raise ValueError("invalid_index_schema")
        descriptor = payload.get("ifc_file")
        if not isinstance(descriptor, dict):
            raise ValueError("missing_ifc_source_descriptor")
        rel = descriptor.get("path")
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or "\\" in rel:
            raise ValueError("unsafe_ifc_relative_path")
        original = (root / rel).resolve()
        if not original.is_relative_to(root) or not original.is_file() or original.is_symlink():
            raise ValueError("missing_or_outside_ifc_source")
        digest = hashlib.sha256()
        with original.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        ifc_hash = digest.hexdigest()
        if not SHA.fullmatch(str(descriptor.get("sha256") or "")):
            errors.append("invalid_recorded_ifc_hash")
        if ifc_hash != descriptor.get("sha256"):
            errors.append("ifc_file_hash_mismatch")
        if expected_ifc_sha256 is not None and ifc_hash != expected_ifc_sha256:
            errors.append("independent_ifc_hash_mismatch")
        if payload.get("coordinate_frame") != "bim_world":
            errors.append("unsupported_coordinate_frame")
        if payload.get("units") != "m":
            errors.append("unverified_ifc_units")
        prov = payload.get("provenance")
        if (not isinstance(prov, dict) or not prov.get("generator") or
                not prov.get("reviewer") or not SHA.fullmatch(str(prov.get("approval_document_sha256") or ""))):
            errors.append("ifc_index_provenance_unapproved")
        rows = payload.get("elements")
        if not isinstance(rows, list) or not rows:
            errors.append("missing_ifc_elements")
            rows = []
        ids: dict[str, dict[str, Any]] = {}
        for rec in rows:
            if not isinstance(rec, dict):
                errors.append("invalid_ifc_element_record"); continue
            gid = rec.get("global_id")
            if not isinstance(gid, str) or not IFC_GUID.fullmatch(gid) or gid in ids:
                errors.append("bad_or_duplicate_ifc_globalid"); continue
            ids[gid] = {k: rec.get(k) for k in ("global_id", "ifc_class", "storey_id")}
        # The external inventory is NOT itself proof that IFC GlobalIds exist.
        # Parse the immutable model with IfcOpenShell when available and verify
        # identity/type, including the author's model-unit conversion to metres.
        try:
            import ifcopenshell
            import ifcopenshell.util.unit
        except ImportError:
            errors.append("ifcopenshell_parser_unavailable")
        else:
            try:
                model = ifcopenshell.open(str(original))
                real = {e.GlobalId: e for e in model.by_type("IfcRoot")
                        if getattr(e, "GlobalId", None)}
                for gid, rec in ids.items():
                    element = real.get(gid)
                    if element is None:
                        errors.append("global_id_not_found_in_ifc")
                    elif rec.get("ifc_class") and element.is_a() != rec["ifc_class"]:
                        errors.append("ifc_element_class_mismatch")
                scale = float(ifcopenshell.util.unit.calculate_unit_scale(model))
                recorded_scale = float(payload.get("ifc_length_scale_to_m"))
                if (not math.isfinite(scale) or scale <= 0 or
                        not math.isfinite(recorded_scale) or recorded_scale <= 0 or
                        not math.isclose(scale, recorded_scale, rel_tol=1e-9, abs_tol=1e-12)):
                    errors.append("ifc_unit_conversion_mismatch")
            except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
                errors.append("ifc_parser_validation_failed:" + type(exc).__name__)
        if errors:
            return IFCIndex(False, tuple(sorted(set(errors))), ifc_hash, index_hash, {})
        return IFCIndex(True, (), ifc_hash, index_hash, ids)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return IFCIndex(False, (f"ifc_identity_load_failure:{type(exc).__name__}:{exc}",),
                        ifc_hash, index_hash, {})
