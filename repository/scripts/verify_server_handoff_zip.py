from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.export_server_handoff_zip import FORBIDDEN_PREFIXES, REQUIRED_HANDOFF_FILES


@dataclass(frozen=True)
class HandoffZipVerification:
    status: str
    passed: bool
    zip_path: str
    file_count: int
    size_bytes: int
    sha256: str
    missing_required: list[str]
    forbidden_entries: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def verify_server_handoff_zip(path: Path) -> HandoffZipVerification:
    warnings: list[str] = []

    if not path.exists():
        return HandoffZipVerification(
            status="missing_zip",
            passed=False,
            zip_path=str(path),
            file_count=0,
            size_bytes=0,
            sha256="",
            missing_required=list(REQUIRED_HANDOFF_FILES),
            forbidden_entries=[],
            warnings=["zip_file_not_found"],
        )

    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            bad_zip_member = zf.testzip()
    except zipfile.BadZipFile:
        return HandoffZipVerification(
            status="bad_zip",
            passed=False,
            zip_path=str(path),
            file_count=0,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
            missing_required=list(REQUIRED_HANDOFF_FILES),
            forbidden_entries=[],
            warnings=["not_a_valid_zip_file"],
        )

    if bad_zip_member:
        warnings.append(f"zip_integrity_warning:first_bad_member={bad_zip_member}")

    missing_required = [
        item
        for item in REQUIRED_HANDOFF_FILES
        if item not in names
    ]

    forbidden_entries = [
        item
        for item in sorted(names)
        if item.startswith(FORBIDDEN_PREFIXES)
        or "/__pycache__/" in item
        or item.endswith((".pyc", ".pyo"))
    ]

    status = (
        "ok"
        if not missing_required and not forbidden_entries and not bad_zip_member
        else "failed"
    )

    return HandoffZipVerification(
        status=status,
        passed=status == "ok",
        zip_path=str(path),
        file_count=len(names),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        missing_required=missing_required,
        forbidden_entries=forbidden_entries,
        warnings=warnings,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify official server handoff ZIP completeness."
    )
    parser.add_argument("zip_path")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-json", default="")

    args = parser.parse_args()

    result = verify_server_handoff_zip(Path(args.zip_path))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(
            "SERVER_HANDOFF_ZIP_VERIFY "
            f"status={result.status} "
            f"passed={str(result.passed).lower()} "
            f"files={result.file_count} "
            f"size={result.size_bytes} "
            f"sha256={result.sha256}"
        )

        if result.missing_required:
            print("missing_required=" + ",".join(result.missing_required))

        if result.forbidden_entries:
            print("forbidden_entries=" + ",".join(result.forbidden_entries[:20]))

        if result.warnings:
            print("warnings=" + ",".join(result.warnings))

    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
