"""Sample-level provenance and leakage guards for held-out building evaluation."""
from __future__ import annotations

import hashlib
from pathlib import Path


ALLOWED_SPLITS = {"train", "validation", "test"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != "mycon.validation.v1":
        raise ValueError("Expected schema_version=mycon.validation.v1")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest must list one or more samples")
    ids, groups, sites, captures, hashes = set(), {}, {}, {}, {}
    for s in samples:
        for field in ("id", "site_id", "building_id", "capture_session_id", "split", "source_sha256"):
            if not isinstance(s.get(field), str) or not s[field].strip():
                raise ValueError("Sample missing required provenance: " + field)
        if s["split"] not in ALLOWED_SPLITS:
            raise ValueError("Unrecognized split")
        if s["id"] in ids:
            raise ValueError("Duplicate sample identifier")
        ids.add(s["id"])
        # Hold out entire site/building; capture sessions within a building cannot cross splits.
        group = (s["site_id"], s["building_id"])
        if group in groups and groups[group] != s["split"]:
            raise ValueError("Building/site leakage across splits")
        groups[group] = s["split"]
        if s["site_id"] in sites and sites[s["site_id"]] != s["split"]:
            raise ValueError("Site leakage across splits")
        sites[s["site_id"]] = s["split"]
        capture = s["capture_session_id"]
        if capture in captures and captures[capture] != s["split"]:
            raise ValueError("Capture-session leakage across splits")
        captures[capture] = s["split"]
        digest = s["source_sha256"].lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("source_sha256 must be a real 64-character SHA256 digest")
        if digest in hashes and hashes[digest] != s["split"]:
            raise ValueError("Duplicate underlying capture across splits")
        hashes[digest] = s["split"]
        if not isinstance(s.get("methods"), dict) or not s["methods"]:
            raise ValueError("Each sample needs named method manifests")
        if s.get("ground_truth") and not isinstance(s["ground_truth"], dict):
            raise ValueError("ground_truth must be a map")


def resolve_input(root: Path, name: str) -> Path:
    """Read-only input resolver: resolve without permitting traversal out of dataset root."""
    candidate = (root / name).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Path traversal outside input root")
    return candidate
