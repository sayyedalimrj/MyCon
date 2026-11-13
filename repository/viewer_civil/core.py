"""MyCON Civil Viewer: read-only, fail-closed input adapter and standalone preview."""
from __future__ import annotations
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

SCHEMA = "mycon.viewer_bundle.v1"
ROLES = ("camera_poses", "frame_manifest", "planes", "edges", "corners", "mesh", "alignment_result")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
HASH = re.compile(r"^[0-9a-f]{64}$")


class ViewerInputError(ValueError):
    """A source bundle is untrustworthy or not compatible with this viewer."""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ViewerInputError("unsafe relative path")
    if Path(relative).is_absolute() or any(p in ("", ".", "..") for p in relative.split("/")):
        raise ViewerInputError("unsafe relative path")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ViewerInputError("path missing or escapes bundle: " + relative)
    return path


def json_file(path):
    if path.stat().st_size > 25_000_000:
        raise ViewerInputError("JSON input too large for preview")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise ViewerInputError("invalid JSON") from e
    if not isinstance(value, dict):
        raise ViewerInputError("expected JSON object")
    return value


def _vector(v, n=3):
    return isinstance(v, list) and len(v) == n and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v
    )


def _matrix(m):
    if not (isinstance(m, list) and len(m) == 4 and all(_vector(r, 4) for r in m)):
        return False
    if any(abs(m[3][i] - [0, 0, 0, 1][i]) > 1e-4 for i in range(4)):
        return False
    # Do not silently accept scaled, degenerate or reflected poses as rigid calibrated extrinsics.
    axes = [[m[r][c] for r in range(3)] for c in range(3)]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    if any(abs(dot(axes[i], axes[j]) - (1 if i == j else 0)) > .025
           for i in range(3) for j in range(3)):
        return False
    a, b, c = axes
    determinant = dot(a, [b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0]])
    return determinant > .95


def _deviation_gate(m, assets):
    a = assets["alignment_result"]
    if m.get("synthetic") is True:
        return False, "synthetic_dataset"
    if not (m.get("metric_scale_verified") is True and a.get("metric_scale_verified") is True):
        return False, "metric_scale_not_verified"
    if not (m.get("registration_verified") is True and a.get("registration_verified") is True):
        return False, "registration_not_verified"
    if a.get("independent_control_verified") is not True:
        return False, "independent_control_not_verified"
    if m["units"] != "m" or a["units"] != "m" or a["coordinate_frame"] != m["coordinate_frame"]:
        return False, "metric_coordinate_mismatch"
    d = assets["mesh"].get("deviations")
    faces = assets["mesh"].get("faces", [])
    if not isinstance(d, dict) or not d.get("provenance") or d.get("independently_verified") is not True:
        return False, "no_independent_deviation_evidence"
    vals = d.get("per_face_m")
    if d.get("units") != "m" or not isinstance(vals, list) or len(vals) != len(faces):
        return False, "deviation_schema_mismatch"
    if any(x is not None and (not isinstance(x, (int, float)) or not math.isfinite(x)) for x in vals):
        return False, "deviation_nonfinite"
    return True, "independently_verified"


def load_bundle(folder):
    """Read source files without modifying originals. Fail before displaying any untrusted input."""
    root = Path(folder).resolve()
    manifest = json_file(read_path(root, "viewer_manifest.json"))
    if manifest.get("schema_version") != SCHEMA:
        raise ViewerInputError("schema_version mismatch")
    if not RUN_ID.fullmatch(str(manifest.get("run_id", ""))):
        raise ViewerInputError("invalid run_id")
    if not isinstance(manifest.get("coordinate_frame"), str) or not manifest["coordinate_frame"]:
        raise ViewerInputError("missing coordinate frame")
    if manifest.get("units") not in ("m", "arbitrary", "reconstruction_units"):
        raise ViewerInputError("unknown units")
    if not isinstance(manifest.get("provenance"), dict) or not manifest["provenance"]:
        raise ViewerInputError("missing provenance")
    files = manifest.get("files")
    if not isinstance(files, dict) or not set(ROLES).issubset(files):
        raise ViewerInputError("missing required input roles")
    assets = {}
    for role, receipt in files.items():
        if not isinstance(receipt, dict):
            raise ViewerInputError("invalid receipt " + role)
        for key in ("path", "sha256", "schema_version", "units", "coordinate_frame", "quality", "provenance"):
            if key not in receipt:
                raise ViewerInputError(role + " missing " + key)
        if receipt["units"] != manifest["units"] or receipt["coordinate_frame"] != manifest["coordinate_frame"]:
            raise ViewerInputError(role + " frame/units mismatch")
        path = read_path(root, receipt["path"])
        if not HASH.fullmatch(str(receipt["sha256"])) or sha256(path) != receipt["sha256"]:
            raise ViewerInputError(role + " SHA256 mismatch")
        body = json_file(path)
        for key in ("schema_version", "units", "coordinate_frame", "quality", "provenance"):
            if body.get(key) != receipt.get(key) and key in ("schema_version", "units", "coordinate_frame"):
                raise ViewerInputError(role + " content/schema mismatch")
            if key not in body:
                raise ViewerInputError(role + " missing " + key + " in payload")
        if not isinstance(body["provenance"], dict) or not body["provenance"]:
            raise ViewerInputError(role + " missing source provenance")
        assets[role] = body
    frames = assets["frame_manifest"].get("frames")
    cameras = assets["camera_poses"].get("cameras")
    if not isinstance(frames, list) or not isinstance(cameras, list):
        raise ViewerInputError("invalid frames or cameras list")
    ids = set()
    for item in frames:
        fid = item.get("frame_id")
        if not isinstance(fid, str) or not RUN_ID.fullmatch(fid) or fid in ids:
            raise ViewerInputError("duplicate/invalid frame_id")
        ids.add(fid)
        image = read_path(root, item.get("path"))
        if image.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            raise ViewerInputError("only safe raster frame previews are supported")
        if not HASH.fullmatch(str(item.get("sha256", ""))) or sha256(image) != item["sha256"]:
            raise ViewerInputError("source image SHA256 mismatch")
    camera_ids = set()
    for c in cameras:
        cid = c.get("camera_id")
        if (not isinstance(cid, str) or not RUN_ID.fullmatch(cid) or cid in camera_ids
                or c.get("frame_id") not in ids):
            raise ViewerInputError("camera/frame identifier invalid")
        camera_ids.add(cid)
        if c.get("extrinsic_convention") != "camera_to_world" or not _matrix(c.get("camera_to_world")):
            raise ViewerInputError("camera_to_world must be rigid 4x4, +Z forward")
        for k in ("fx", "fy", "cx", "cy", "width", "height"):
            if not isinstance(c.get(k), (int, float)) or not math.isfinite(c[k]):
                raise ViewerInputError("bad camera intrinsic " + k)
        if min(c["fx"], c["fy"], c["width"], c["height"]) <= 0:
            raise ViewerInputError("non-positive focal length/dimensions")
    mesh = assets["mesh"]
    vertices, faces = mesh.get("vertices", []), mesh.get("faces", [])
    if not isinstance(vertices, list) or not all(_vector(v) for v in vertices):
        raise ViewerInputError("mesh vertices invalid")
    if not isinstance(faces, list) or not all(
        isinstance(f, list) and len(f) == 3 and all(
            isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(vertices) for i in f
        ) for f in faces
    ):
        raise ViewerInputError("mesh faces invalid")
    for p in assets["planes"].get("planes", []):
        if not isinstance(p.get("id"), str) or not all(_vector(v) for v in p.get("vertices", [])):
            raise ViewerInputError("plane invalid")
    for e in assets["edges"].get("edges", []):
        if not _vector(e.get("start")) or not _vector(e.get("end")):
            raise ViewerInputError("edge invalid")
    for c in assets["corners"].get("corners", []):
        if not _vector(c.get("xyz")):
            raise ViewerInputError("corner invalid")
    allowed, reason = _deviation_gate(manifest, assets)
    return {"manifest": manifest, "assets": assets, "deviation_enabled": allowed,
            "deviation_reason": reason, "root": root}


def build_preview(source, destination):
    """Create a new, standalone, offline HTML preview without editing source files."""
    import plotly
    root = Path(source).resolve()
    b = load_bundle(root)
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite
    (out / "frames").mkdir()
    for frame in b["assets"]["frame_manifest"]["frames"]:
        src = read_path(root, frame["path"])
        target = out / "frames" / (frame["frame_id"] + src.suffix.lower())
        shutil.copyfile(src, target)
        frame["preview_path"] = "frames/" + target.name
    payload = {key: value for key, value in b.items() if key != "root"}
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    static = Path(__file__).parent / "static"
    index = (static / "index.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(index.replace("__MYCON_PAYLOAD__", raw), encoding="utf-8")
    for name in ("viewer.js", "viewer.css"):
        shutil.copyfile(static / name, out / name)
    shutil.copyfile(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", out / "plotly.min.js")
    receipt = {"schema_version": "mycon.viewer_preview.v1", "run_id": b["manifest"]["run_id"],
               "source_manifest_sha256": sha256(root / "viewer_manifest.json"),
               "metric_deviation_enabled": b["deviation_enabled"],
               "metric_gate_reason": b["deviation_reason"], "synthetic": b["manifest"].get("synthetic")}
    (out / "viewer_build.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return out / "index.html"
