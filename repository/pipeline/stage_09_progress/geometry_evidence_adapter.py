"""Fail-closed consumer of the optional Stage-8 A->B geometry evidence v1.

Never infer verified IFC element identity from AABBs or geometric proximity.
All paths are project-relative and SHA-256 checked. An invalid global bundle
invalidates *all* derived visibility claims; independently approved quantities
are unaffected. JSONL is the canonical B contract-fixture representation.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA = 'mycon.bim_geometry_evidence.v1'
VISIBILITY_STATES = {'visible', 'occluded', 'out_of_fov', 'ambiguous', 'unknown'}
IFC_UNKNOWN = {'', 'unknown', 'unassigned', 'null'}


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value.lower())


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _matrix4(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 4
            and all(isinstance(row, list) and len(row) == 4 and all(_number(v) for v in row) for row in value)
            and all(abs(a-b) < 1e-8 for a,b in zip(value[3], [0,0,0,1])))


def _sandbox_file(root: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or '\\' in rel:
        raise ValueError('unsafe_relative_path')
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()) or path.is_symlink() or not path.is_file():
        raise ValueError('missing_or_unsafe_file')
    return path


def _load_sidecar(root: Path, descriptor: Any) -> list[dict[str, Any]]:
    if not isinstance(descriptor, dict) or not _hash(descriptor.get('sha256')):
        raise ValueError('sidecar_descriptor_invalid')
    path = _sandbox_file(root, descriptor.get('path'))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != descriptor['sha256'].lower():
        raise ValueError('sidecar_hash_mismatch')
    if path.suffix != '.jsonl':
        raise ValueError('unsupported_sidecar_format_use_jsonl')
    output = []
    for line in raw.decode('utf-8').splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError('jsonl_record_not_object')
            output.append(value)
    return output


@dataclass(frozen=True)
class GeometryEvidenceBundle:
    valid: bool
    reason_codes: tuple[str, ...]
    manifest_sha256: str | None
    rows: Mapping[str, dict[str, Any]]

    def for_element(self, global_id: str, physical_fraction: float | None = None) -> dict[str, Any]:
        """Per-ID classification. Physical status needs independent signed quantity.

        Values are additive Stage9 fields. No view -> unknown, not unbuilt.
        """
        base = {'geometry_adapter_schema': SCHEMA,
                'geometry_adapter_valid': self.valid,
                'geometry_manifest_sha256': self.manifest_sha256,
                'geometry_evidence_state': 'unknown_manifest' if not self.valid else 'unknown_no_element_evidence',
                'observability_ratio': None, 'observability_source': '',
                'evidence_confidence': None, 'evidence_confidence_source': '',
                'geometry_evidence_reason_codes': ';'.join(self.reason_codes),
                'geometry_conformance_state': 'unknown',
                'geometry_n_views': 0, 'geometry_quality_gate': 'blocked',
                'image_evidence_refs_json': '[]', 'geometry_support_sample_ids_json': '[]'}
        if not self.valid:
            return base
        record = self.rows.get(global_id)
        if not record:
            return base
        base.update({k:record[k] for k in ('geometry_evidence_state','observability_ratio',
                    'observability_source','geometry_evidence_reason_codes','geometry_conformance_state',
                    'geometry_n_views','geometry_quality_gate')})
        base['image_evidence_refs_json'] = json.dumps(record.get('image_evidence_refs', []), sort_keys=True)
        base['geometry_support_sample_ids_json'] = json.dumps(record.get('geometry_support_sample_ids', []))
        if record['geometry_evidence_state'] == 'observed_geometry_consistent' and physical_fraction is not None:
            if 0 < physical_fraction < 1:
                base['geometry_evidence_state'] = 'partial_built_verified'
            elif physical_fraction > 0:
                base['geometry_evidence_state'] = 'built_observed_verified'
        return base


def load_geometry_evidence_manifest(
    manifest_path: Path | str, *, project_root: Path | str,
    expected_global_ids: set[str] | None = None,
    expected_ifc_sha256: str | None = None,
    expected_scan_sha256: str | None = None,
    expected_registration_sha256: str | None = None,
    expected_project_id: str | None = None,
) -> GeometryEvidenceBundle:
    """Read A-produced manifest with strict common-frame / visibility gates.

    Never trust optional camera and sample ownership unless all referenced files,
    sample IDs, camera IDs, hashes, registration and scale references validate.
    Invalid bundle => all geometry visibility unknown, never a false 0%.
    """
    errors: list[str] = []
    try:
        root = Path(project_root).resolve()
        path = Path(manifest_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('manifest_not_in_project_root')
        raw = path.read_bytes()
        manifest_sha = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        if manifest.get('schema_version') != SCHEMA: errors.append('schema_version_mismatch')
        if not manifest.get('run_id') or not manifest.get('project_id') or not manifest.get('created_at_utc'):
            errors.append('missing_run_project_timestamp')
        if expected_project_id is not None and manifest.get('project_id') != expected_project_id:
            errors.append('project_id_mismatch')
        if manifest.get('coordinate_frame') != 'bim_world' or manifest.get('units') != 'm':
            errors.append('frame_or_units_invalid')
        if not _matrix4(manifest.get('scan_to_bim_4x4')):
            errors.append('registration_transform_invalid')
        for name, expected in [('ifc_sha256', expected_ifc_sha256),
                               ('scan_sha256', expected_scan_sha256),
                               ('registration_report_sha256', expected_registration_sha256)]:
            value = manifest.get(name)
            if not _hash(value): errors.append(name + '_invalid')
            if expected is not None and value != expected: errors.append(name + '_mismatch')
        if manifest.get('metric_scale_verified') is not True or not manifest.get('independent_control_evidence_ref'):
            errors.append('metric_scale_not_independently_verified')
        gate = manifest.get('registration_quality_gate')
        if not isinstance(gate, dict) or gate.get('status') not in ('pass', 'accepted'):
            errors.append('registration_gate_not_passed')
        prov = manifest.get('provenance')
        if not isinstance(prov, dict) or not all(k in prov and prov[k] is not None and prov[k] != '' for k in
                ('code_commit','config_sha256','seed','dataset_id','capture_session_id','software_versions')):
            errors.append('insufficient_provenance')
        samples = _load_sidecar(root, manifest.get('surface_samples'))
        cameras = _load_sidecar(root, manifest.get('camera_poses'))
        visibility = _load_sidecar(root, manifest.get('element_visibility'))
        # Duplicate sample/camera references make the entire output ambiguous.
        sample_ids: dict[str, str] = {}
        for rec in samples:
            sid, gid = rec.get('sample_id'), rec.get('global_id')
            xyz = rec.get('xyz_m')
            if not sid or sid in sample_ids or not isinstance(gid, str) or not isinstance(xyz, list) or len(xyz)!=3 or not all(_number(v) for v in xyz) or not rec.get('geometry_version') or not rec.get('sampling_policy') or not (rec.get('triangle_id') or rec.get('surface_id')):
                errors.append('invalid_or_duplicate_surface_sample')
            if sid: sample_ids[sid] = gid
            if expected_global_ids is not None and gid not in expected_global_ids and gid not in IFC_UNKNOWN:
                errors.append('unrecognized_global_id')
        camera_ids: set[str] = set()
        cameras_by_view: dict[str, dict[str, Any]] = {}
        for rec in cameras:
            vid = rec.get('view_id')
            if not vid or vid in camera_ids or not rec.get('capture_session_id') or not _hash(rec.get('image_sha256')) or not isinstance(rec.get('image_width'), int) or rec['image_width'] <=0 or not isinstance(rec.get('image_height'),int) or rec['image_height']<=0 or not _matrix4(rec.get('world_to_camera_4x4')):
                errors.append('invalid_camera_pose')
            intr = rec.get('intrinsics')
            if not isinstance(intr, dict) or not all(_number(intr.get(k)) and intr[k]>0 for k in ('fx','fy')) or not all(_number(intr.get(k)) for k in ('cx','cy')) or not rec.get('pose_quality_ref'):
                errors.append('invalid_camera_calibration')
            if vid:
                camera_ids.add(vid)
                cameras_by_view[vid] = rec
        by_id: dict[str, list[dict[str, Any]]] = {}
        visible_samples: dict[str,set[str]] = {}
        eligible_samples: dict[str,set[str]] = {}
        explicit_states: dict[str,set[str]] = {}
        for rec in visibility:
            gid, state, view = rec.get('global_id'),rec.get('state'),rec.get('view_id')
            if not isinstance(gid,str) or state not in VISIBILITY_STATES or view not in camera_ids:
                errors.append('invalid_visibility_record'); continue
            if expected_global_ids is not None and gid not in expected_global_ids:
                errors.append('unrecognized_visibility_global_id')
            if rec.get('method') not in ('raycast_zbuffer','audited_manual') or not rec.get('qa_provenance_ref'):
                errors.append('visibility_method_or_qa_missing')
            eligible = rec.get('eligible_sample_ids')
            visible = rec.get('visible_sample_ids')
            if not isinstance(eligible,list) or not isinstance(visible,list) or len(set(eligible))!=len(eligible) or len(set(visible))!=len(visible) or not set(visible).issubset(set(eligible)):
                errors.append('visibility_sample_sets_invalid');continue
            if any(sample_ids.get(sid)!=gid for sid in eligible):
                errors.append('visibility_cross_element_sample_id');continue
            if state in {'out_of_fov','occluded'} and visible:
                errors.append('inconsistent_occlusion_state')
            if state == 'visible' and not visible:
                errors.append('visible_state_without_visible_samples')
            if rec.get('segmentation_status') in {'failed','invalid'}:
                errors.append('segmentation_failure_detected')
            eligible_samples.setdefault(gid,set()).update(eligible if state!='out_of_fov' else [])
            visible_samples.setdefault(gid,set()).update(visible)
            explicit_states.setdefault(gid,set()).add(state)
            by_id.setdefault(gid,[]).append(rec)
        if errors:
            return GeometryEvidenceBundle(False,tuple(sorted(set(errors))), manifest_sha, {})
        result: dict[str,dict[str,Any]] = {}
        for gid, records in by_id.items():
            states = explicit_states[gid]
            eligible = eligible_samples.get(gid,set())
            visible = visible_samples.get(gid,set())
            reasons: list[str] = []
            if 'unknown' in states or 'ambiguous' in states:
                state = 'unknown_visibility'; ratio=None
            elif not eligible:
                state = 'out_of_fov' if states=={'out_of_fov'} else 'unknown_observability'
                ratio = None
            else:
                ratio = len(visible)/len(eligible)
                state = ('observed_geometry_unassessed' if visible else 'occluded')
            conformance = 'unknown'
            # Geometry nonconformance needs independently QA'd deviation,
            # not a failed nearest-neighbor match or missing point.
            deviations = [r.get('geometry_assessment') for r in records if r.get('geometry_assessment')]
            if deviations and ratio is not None and ratio > 0:
                if all(isinstance(d,dict) and _number(d.get('max_deviation_m')) and _number(d.get('tolerance_m')) and d['tolerance_m']>0 and d.get('quality_gate')=='pass' for d in deviations):
                    conformance = ('nonconforming' if any(d['max_deviation_m']>d['tolerance_m'] for d in deviations) else 'conforming')
                    state = ('geometric_nonconformance' if conformance == 'nonconforming'
                             else 'observed_geometry_consistent')
                else:
                    reasons.append('unverified_geometric_deviation')
            n_views = len({r['view_id'] for r in records})
            quality_gate = ('pass' if n_views >= 2 and conformance in {'conforming','nonconforming'}
                            else 'limited_view_support_or_unassessed_geometry')
            result[gid] = {
                'geometry_n_views':n_views,
                'geometry_quality_gate':quality_gate,
                'geometry_evidence_state':state,
                'observability_ratio':ratio,
                'observability_source':'raycast_zbuffer' if ratio is not None else '',
                'geometry_evidence_reason_codes':';'.join(reasons),
                'geometry_conformance_state':conformance,
                'geometry_support_sample_ids': sorted(eligible | visible),
                'image_evidence_refs': [
                    {'view_id': v, 'capture_session_id': cameras_by_view[v]['capture_session_id'],
                     'image_sha256': cameras_by_view[v]['image_sha256'],
                     'qa_provenance_ref': next((item.get('qa_provenance_ref') for item in records
                        if item['view_id'] == v), None)}
                    for v in sorted({item['view_id'] for item in records})
                    if v in cameras_by_view
                ],
            }
        return GeometryEvidenceBundle(True,(),manifest_sha,result)
    except (OSError,ValueError,KeyError,TypeError,UnicodeError,json.JSONDecodeError) as exc:
        return GeometryEvidenceBundle(False,(f'manifest_load_failure:{type(exc).__name__}:{exc}',),None,{})
