"""Resumable SIFT/verified match graph and independently exported components."""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
import sqlite3

import cv2
import numpy as np
import pycolmap as pc


def extract(intake, work, manifest, config, say):
    groups=defaultdict(list)
    for row in manifest["frames"]:
        groups[row["group"]].append(row)
    database=work/"native_features.db"
    for group,rows in groups.items():
        options=pc.FeatureExtractionOptions(max_image_size=config["feature_size"],use_gpu=False,
                                           num_threads=config["threads"])
        options.sift.max_num_features=config["features"]
        known=next((v.get("initial_camera_k") for v in manifest["videos"] if v["group"]==group),None)
        reader = pc.ImageReaderOptions(camera_model="OPENCV" if known else "SIMPLE_RADIAL")
        if known:
            reader.camera_params=",".join(map(str,[*known,0,0,0,0]))
        # Keep intrinsics separate across recordings. For still photos let
        # COLMAP inspect EXIF and use its default automatic camera grouping.
        stills=rows[0]["pts_s"] is None
        mode=pc.CameraMode.AUTO if stills else pc.CameraMode.PER_FOLDER
        pc.extract_features(database,intake/"frames",image_names=[r["name"] for r in rows],
                            camera_mode=mode,reader_options=reader,extraction_options=options,device=pc.Device.cpu)
        say(f"sift · camera group {group+1} · {len(rows)} images")
    with closing(sqlite3.connect(database)) as c:
        n=c.execute("SELECT count(*) FROM keypoints WHERE rows >= 8").fetchone()[0]
    if n<3:
        raise ValueError("Fewer than three images have sufficient visual features")
    seal_db(database,work/"features.db")
    return dict(feature_images=n,engine=f"PyCOLMAP {pc.__version__} CPU SIFT")


def retrieval_descriptor(path):
    grey=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if grey is None:
        raise ValueError(f"Unreadable feature image {path.name}")
    mini=cv2.resize(grey,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    d=cv2.dct(mini)[:9,:9].ravel()[1:]
    return d/max(np.linalg.norm(d),1e-8)


def propose(intake, rows, limit, targeted=None):
    features=np.array([retrieval_descriptor(intake/"frames"/r["name"]) for r in rows])
    n=len(rows);all_pairs=set();nonlocal_pairs=set()
    groups=np.array([r["group"] for r in rows]);segments=np.array([r.get("segment",0) for r in rows])
    seconds=np.array([r["pts_s"] if r["pts_s"] is not None else i*.35 for i,r in enumerate(rows)])
    inds=np.arange(n)
    for i in range(n):
        if targeted is not None and rows[i]["name"] not in targeted:
            continue
        if n<=45 and targeted is None:
            all_pairs.update((i,j) for j in range(i+1,n))
            continue
        same=(groups==groups[i])&(segments==segments[i])
        distance=abs(seconds-seconds[i]);desc=features @ features[i]
        near=inds[same&(abs(inds-i)<=limit)&(i!=inds)]
        adjacent=inds[same&(distance>.45)&(distance<8.)&(abs(inds-i)>limit)]
        distant=inds[(~same)|(distance>=8.)]
        for j in near:
            all_pairs.add(tuple(sorted((i,int(j)))))
        for candidate,k in [(adjacent,5),(distant,5 if targeted is None else 20)]:
            if not len(candidate):
                continue
            k=min(k,len(candidate))
            top=candidate[np.argpartition(desc[candidate],-k)[-k:]]
            for j in top:
                pair=tuple(sorted((i,int(j))))
                if desc[j]>.37:
                    all_pairs.add(pair)
                    if not same[j] or distance[j]>=8.:
                        nonlocal_pairs.add(pair)
    return all_pairs,nonlocal_pairs


def copy_db(source,target):
    uri=source.resolve().as_uri()+"?mode=ro"
    with closing(sqlite3.connect(uri,uri=True)) as original,closing(sqlite3.connect(target)) as backup:
        original.backup(backup)
        backup.execute("PRAGMA journal_mode=DELETE")


def remove_native_db(path):
    for suffix in ("","-wal","-shm"):
        path.with_name(path.name+suffix).unlink(missing_ok=True)


def seal_db(source,target):
    copy_db(source,target)
    remove_native_db(source)


def apply_matches(db, path, candidates, rows, config):
    names=[r["name"] for r in rows]
    path.write_text("".join(f"{names[i]} {names[j]}\n" for i,j in sorted(candidates)))
    if not candidates:
        return
    pc.match_image_pairs(db,matching_options=pc.FeatureMatchingOptions(
        num_threads=config["threads"],use_gpu=False,max_num_matches=config["features"]),
        pairing_options=pc.ImportedPairingOptions(match_list_path=path),
        verification_options=pc.TwoViewGeometryOptions(min_num_inliers=22),device=pc.Device.cpu)


def match(intake, features, work, manifest, config, say):
    db=work/"native_matching.db"
    copy_db(features/"features.db",db)
    rows=manifest["frames"]
    pr,far=propose(intake,rows,config["neighbors"])
    apply_matches(db,work/"pairs.txt",pr,rows,config)
    with closing(sqlite3.connect(db)) as c:
        count=c.execute("SELECT count(*) FROM two_view_geometries WHERE rows >= 22").fetchone()[0]
        names=dict(c.execute("SELECT image_id,name FROM images"))
        found=c.execute("SELECT pair_id,rows,config FROM two_view_geometries WHERE rows>=22").fetchall()
    if count==0:
        raise ValueError("Geometric verification found no supported pairs")
    index={r["name"]:i for i,r in enumerate(rows)}
    nonlocal_count=0
    counts=defaultdict(int)
    for pid,_,kind in found:
        a,b=pc.pair_id_to_image_pair(pid)
        key=tuple(sorted((index[names[a]],index[names[b]])))
        if key in far:
            nonlocal_count+=1
        counts[str(pc.TwoViewGeometryConfiguration(kind))]+=1
    seal_db(db,work/"matching.db")
    say(f"matching · {count}/{len(pr)} geometrically verified pairs")
    return dict(candidate_pairs=len(pr),verified_pairs=count,nonlocal_verified_pairs=nonlocal_count,
                verified_pair_types=dict(counts))


def solve(db,image_dir,out,config):
    out.mkdir()
    opt=pc.IncrementalPipelineOptions(num_threads=config["threads"],random_seed=42,
              ba_refine_principal_point=False,min_model_size=4)
    opt.mapper.init_min_num_inliers=55
    opt.mapper.init_min_tri_angle=6.
    opt.ba_global_max_num_iterations=35
    opt.ba_global_frames_ratio=1.6
    opt.ba_global_points_ratio=1.6
    recs=pc.incremental_mapping(db,image_dir,out,opt)
    return list(recs.values())


def rank(rec):
    return (rec.num_reg_images(),-max(0.,rec.compute_mean_reprojection_error()),rec.num_points3D())


def map_scene(intake,matches,work,manifest,config,say):
    db=work/"map_working.db"
    copy_db(matches/"matching.db",db)
    image_dir=intake/"frames"
    rows=manifest["frames"]
    recs=solve(db,image_dir,work/"initial_models",config)
    valid=[r for r in recs if r.num_reg_images()>=4 and r.num_points3D()>=25]
    repaired=False
    if valid:
        top=max(valid,key=rank)
        known={top.images[i].name for i in top.reg_image_ids()}
    else:
        known=set()
    missing={r["name"] for r in rows}-known
    if len(known)/len(rows)<.82 and config["retry_matching"]:
        # Retain the verified original DB. Add wider candidate retrieval only
        # for lost views and rerank whole models, never fuse disconnected worlds.
        retry=work/"targeted.db"
        copy_db(db,retry)
        more,_=propose(intake,rows,max(config["neighbors"]*2,12),targeted=missing)
        original={tuple(x.split()) for x in (matches/"pairs.txt").read_text().splitlines()}
        more={(i,j) for i,j in more if (rows[i]["name"],rows[j]["name"]) not in original}
        if more:
            say(f"mapping · targeted recovery for {len(missing)} unmatched or failed views")
            apply_matches(retry,work/"targeted_pairs.txt",more,rows,config)
            second=solve(retry,image_dir,work/"recovery_models",config)
            replacement=[r for r in second if r.num_reg_images()>=4 and r.num_points3D()>=25]
            if replacement and (not valid or rank(max(replacement,key=rank))>rank(max(valid,key=rank))):
                valid=replacement
                repaired=True
        remove_native_db(retry)
    if not valid:
        raise ValueError("Visual overlaps produced no recoverable, nondegenerate structure")
    remove_native_db(db)
    ordered=sorted(valid,key=rank,reverse=True)
    best=ordered[0]
    model=work/"model";model.mkdir();best.write(model)
    best.export_PLY(work/"Sparse.ply")
    for i,rec in enumerate(ordered[1:],2):
        rec.export_PLY(work/f"Disconnected_{i}.ply")
    registered={best.images[i].name for i in best.reg_image_ids()}
    coverage=defaultdict(lambda:[0,0])
    for r in rows:
        stamp=int(r["pts_s"]//10)*10 if r["pts_s"] is not None else None
        key=(r["group"],stamp)
        coverage[key][0]+=1;coverage[key][1]+=r["name"] in registered
    say(f"global bundle adjustment · {len(registered)}/{len(rows)} views · {len(ordered)} components")
    return dict(registered=len(registered),selected=len(rows),components=[r.num_reg_images() for r in ordered],
                sparse_points=best.num_points3D(),mean_internal_reprojection_px=best.compute_mean_reprojection_error(),
                targeted_recovery_selected=repaired,nonmetric=True,
                missing_views=sorted({r["name"] for r in rows}-registered),
                coverage=[dict(group=g,ten_second_bin=stamp,selected=k[0],registered=k[1])
                          for (g,stamp),k in coverage.items()])
