#!/usr/bin/env python3
"""Preflight a configured local/offline Stage 10 VLM endpoint."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.common.config import load_config  # noqa: E402
from pipeline.stage_10_copilot.evidence_builder import EvidencePackage  # noqa: E402
from pipeline.stage_10_copilot.vlm_answer import answer_with_vlm  # noqa: E402


def _dummy_package(root: Path) -> EvidencePackage:
    evidence = root / "local_vlm_preflight_evidence.json"
    image = root / "preflight.png"
    # Minimal valid PNG: 1x1 transparent pixel.
    image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c6360000002000100ffff03000006000557bfab0000000049454e44ae426082"
        )
    )
    package = EvidencePackage(
        question="Reply exactly LOCAL_VLM_OK and mention no metric values.",
        route={"category": "general_explanation"},
        selected_context={},
        image_paths={"preflight": image.as_posix()},
        metrics={"preflight": {"status": "ok", "data": {}}},
        schedule_context={},
        limitations=[],
        confidence_flags=["preflight"],
        evidence_path=evidence.as_posix(),
        selected_element_id=None,
        selected_activity_id=None,
    )
    evidence.write_text("{}", encoding="utf-8")
    return package


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight local/offline VLM used by Stage 10.")
    parser.add_argument("--config", default="configs/site01.yaml")
    parser.add_argument("--require-real", action="store_true", help="Fail if provider falls back to mock.")
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    with tempfile.TemporaryDirectory(prefix="stage10_vlm_preflight_") as tmp:
        answer = answer_with_vlm(cfg, _dummy_package(Path(tmp)))
    if args.require_real and "mock" in answer.provider:
        raise SystemExit(f"STAGE_10_LOCAL_VLM_FAILED provider={answer.provider} answer={answer.answer[:300]}")
    print(f"STAGE_10_LOCAL_VLM_OK provider={answer.provider} confidence={answer.confidence}")
    print(answer.answer[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
