"""Deterministic CPU tests: synthetic assertions only, not measured site accuracy."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from pipeline.stage_07_cleanup.comparison_ready import (
    SCHEMA, clean_points, extract_planes, extract_edges, extract_corners,
    project_corner, run_ready, reuse_stage7_planes, _ply_write, _ply_read,
)


def room(n=16, noise=0., seed=11):
    grid = np.linspace(0, 1.7, n)
    u,v = np.meshgrid(grid, grid)
    p = np.vstack([
        np.column_stack([np.zeros(u.size), u.ravel(), v.ravel()]),
        np.column_stack([u.ravel(), np.zeros(u.size), v.ravel()]),
        np.column_stack([u.ravel(), v.ravel(), np.zeros(u.size)])
    ])
    if noise:
        p = p + np.random.default_rng(seed).normal(0, noise, p.shape)
    return p


def features(p, *, tol=.018, n=60):
    pl=extract_planes(p, threshold=tol, min_points=n, max_planes=4, trials=350)
    ed=extract_edges(p, pl, radius=.20, min_support=2)
    co=extract_corners(p, pl, ed, radius=.28, min_support=2)
    return pl, ed, co


def test_observed_room_edges_and_corner():
    p = room()
    pl, ed, co = features(p)
    assert len(pl) == 3
    assert len(ed) >= 2
    assert len(co) >= 1
    assert np.linalg.norm(co[0]['xyz']) < .1
    assert co[0]['state'] == 'observed_supported'
    assert all(x['infinite_plane_confirmed'] is False for x in pl)


def test_outlier_cleaning_keeps_real_architectural_edges_and_rgb():
    p = np.vstack([room(n=12), [[200,200,200], [-100,22,52]]])
    colors = np.arange(len(p)*3).reshape(-1,3)%256
    cleaned, rgb, qa = clean_points(p, colors, outlier_mad=5, voxel=.03)
    assert qa['removed_outlier'] >= 1
    assert len(cleaned) <= len(p)
    assert len(rgb) == len(cleaned)
    assert np.max(np.linalg.norm(cleaned,axis=1)) < 10


def test_rotated_non_manhattan_room():
    p=room(noise=.001)
    a=.49
    rot=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    pl,ed,co=features(p@rot.T)
    assert len(pl)==3
    assert len(co)>=1
    assert np.linalg.norm(co[0]['xyz']) < .15


def test_disjoint_planes_cannot_confirm_far_intersection():
    p = room(n=14)
    p[p[:,0]==0] += [0,8,0]  # move one wall away from both other patches
    pl,ed,co=features(p)
    assert not any(c['state']=='observed_supported' for c in co)


def test_parallel_planes_reject_degenerate_trihedral():
    grid=np.linspace(0,1,16)
    xx,yy=np.meshgrid(grid,grid)
    p=np.vstack([np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)],
                 np.c_[xx.ravel(),yy.ravel(),np.ones(xx.size)]])
    pl,ed,co=features(p,n=50)
    assert not co


def test_camera_reports_fov_only_not_occlusion():
    c={'frame_convention':'colmap_opencv_world_to_camera','calibration_verified':True,
       'world_to_camera_4x4':np.eye(4).tolist(),'fx':100,'fy':100,
       'cx':50,'cy':50,'width':100,'height':100,'view_id':'a'}
    assert project_corner(c,np.array([0,0,2]))['status']=='in_fov_occlusion_unverified'
    assert project_corner(c,np.array([0,0,-1]))['status']=='outside_fov'
    assert project_corner({**c,'calibration_verified':False},np.array([0,0,2]))['status']=='unverified_camera'


def test_run_manifest_and_create_only_and_original_colors(tmp_path:Path):
    raw=tmp_path/'raw.ply'
    p=room(n=13)
    rgb=np.tile(np.array([[15,55,255]],dtype=np.uint8),(len(p),1))
    _ply_write(raw,p,rgb)
    manifest=run_ready(raw,tmp_path,'geom-ready-test-0001', threshold=.018,min_plane_points=55, trials=300)
    assert manifest['schema_version']==SCHEMA
    assert manifest['metric_scale_verified'] is False
    assert manifest['registration_verified'] is False
    assert manifest['plane_count']>=3
    target=tmp_path/'runs'/'geom-ready-test-0001'/'geometry'
    assert (target/'comparison_ready.json').is_file()
    assert (target/'xy_geometry_preview.png').is_file()
    saved, saved_colors=_ply_read(target/'clean_cloud.ply')
    assert np.allclose(saved,p,atol=1e-5)
    assert np.array_equal(saved_colors,rgb)
    assert len(np.load(target/'clean_cloud.npy')) == len(saved)
    with pytest.raises(FileExistsError):
        run_ready(raw,tmp_path,'geom-ready-test-0001')
    assert raw.is_file()


def test_no_metric_claim_from_unknown_units(tmp_path:Path):
    p=room(n=10)
    raw=tmp_path/'small.ply'
    _ply_write(raw,p,None)
    result=run_ready(raw,tmp_path,'geom-ready-arbitrary-1',threshold=.02,min_plane_points=30,preview=False)
    assert result['units']=='unknown_reconstruction_units'
    assert result['supported_corners'] <= result['corner_count']
    assert result['source_camera_sha256'] is None



def test_deterministic_planes_edges_corners_across_repeats():
    source=room(n=15,noise=.002)
    first=features(source,tol=.02,n=55)
    second=features(source,tol=.02,n=55)
    assert [p["coefficients"] for p in first[0]] == [p["coefficients"] for p in second[0]]
    assert [e["endpoints_xyz"] for e in first[1]] == [e["endpoints_xyz"] for e in second[1]]
    assert [c["xyz"] for c in first[2]] == [c["xyz"] for c in second[2]]


def test_density_reduction_still_recovers_observed_planes():
    p=room(n=25,noise=.0003)
    hi=features(p,tol=.015,n=60)[0]
    lo=features(p[::4],tol=.015,n=35)[0]
    assert len(hi)==3
    assert len(lo)==3


def test_occluded_corner_stays_unverified():
    p=room(n=19)
    p=p[np.linalg.norm(p,axis=1)>.48]
    pl,ed,co=features(p,tol=.02,n=60)
    assert len(pl)>=2
    assert not any(c["state"] == "observed_supported" for c in co)


def test_non_orthogonal_planes_kept_without_false_orthogonality_claim():
    p=room(n=18)
    # shear one of the walls: plane no longer perpendicular to the floor
    p[:18*18,0] = .25*p[:18*18,2]
    pl,ed,co=features(p,tol=.02,n=60)
    assert len(pl)==3
    assert all(p["fit_state"]=="candidate_observed_patch" for p in pl)
    assert all(c["state"] in {"observed_supported","candidate_extrapolated","unknown"} for c in co)

def test_reuse_existing_stage7_guided_planes_without_duplicate_ownership(tmp_path: Path):
    xyz = room(n=15)
    inp = tmp_path / "stage7_cleaned.ply"
    _ply_write(inp, xyz, None)
    planes = []
    for index in range(3):
        source = tmp_path / f"plane_{index+1:03d}.ply"
        _ply_write(source, xyz[index*225:(index+1)*225], None)
        planes.append({"plane_id": f"plane_{index+1:03d}", "label": "unknown",
                       "cloud_path": source.as_posix()})
    stage7 = tmp_path / "stage7_planes.json"
    stage7.write_text(json.dumps({"planes": planes}), encoding="utf-8")
    patches = reuse_stage7_planes(xyz, stage7, threshold=.03, min_points=55)
    assert len(patches) == 3
    assert len(set().union(*(set(p["support_indices"]) for p in patches))) == sum(p["point_count"] for p in patches)
    run = run_ready(inp, tmp_path, "geom-ready-stage7-reuse-001", threshold=.03,
                    min_plane_points=55, stage7_planes_json=stage7, preview=False)
    assert run["plane_source"] == "reused_stage7_plane_clouds"
    assert run["source_stage7_planes_sha256"] is not None
    assert run["plane_count"] == 3


pytestmark=pytest.mark.lightweight
