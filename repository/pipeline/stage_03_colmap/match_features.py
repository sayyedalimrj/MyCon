"""Build and run COLMAP feature matching commands."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .colmap_cli import ColmapRunner
from .config_access import bool_to_colmap, cfg_bool, cfg_get, cfg_int, resolve_project_path


def build_sequential_matcher_args(
    cfg: Any,
    database_path: Path,
    matcher_type: str,
    model_options: list[str] | None = None,
) -> list[str]:
    loop_detection = cfg_bool(cfg, "colmap.sequential_loop_detection", False)
    args = [
        "sequential_matcher",
        "--database_path",
        str(database_path),
        "--FeatureMatching.type",
        matcher_type.upper(),
        "--SequentialMatching.overlap",
        str(cfg_int(cfg, "colmap.sequential_overlap", 15)),
        "--SequentialMatching.quadratic_overlap",
        bool_to_colmap(cfg_bool(cfg, "colmap.sequential_quadratic_overlap", True)),
        "--SequentialMatching.loop_detection",
        bool_to_colmap(loop_detection),
    ]
    vocab_tree = cfg_get(cfg, "colmap.sequential_vocab_tree_path", None)
    if loop_detection and vocab_tree:
        # COLMAP loop detection is useful for return paths but should remain
        # explicit because vocab-tree availability varies across deployments.
        args.extend(["--SequentialMatching.vocab_tree_path", str(resolve_project_path(cfg, "colmap.sequential_vocab_tree_path"))])
    if model_options:
        args.extend(model_options)
    return args


def match_features(
    runner: ColmapRunner,
    cfg: Any,
    database_path: Path,
    matcher_type: str,
    model_options: list[str] | None = None,
) -> None:
    strategy = str(cfg_get(cfg, "colmap.matching_strategy", "sequential")).lower()
    if strategy != "sequential":
        raise ValueError("Stage 3 currently supports only colmap.matching_strategy=sequential")
    args = build_sequential_matcher_args(cfg, database_path, matcher_type, model_options)
    runner.run(args, name=f"sequential_matcher:{matcher_type}")
