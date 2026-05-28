#!/usr/bin/env python3
"""Render a method-comparison JSON to a thesis-grade LaTeX booktabs table.

Reads a ``method_comparison.v1`` JSON (the output of
:func:`pipeline.common.method_comparison.to_dict`) and writes a
``\\begin{table} ... \\end{table}`` snippet ready to paste into a Q1
paper draft.

Usage
-----

    python3 scripts/render_method_comparison_latex.py \\
        --input  runs/example_walkthrough/method_comparison.json \\
        --output paper/figures/method_comparison.tex \\
        [--label tab:method_comparison] \\
        [--decimals 1] [--ci-decimals 1] \\
        [--also-ascii  paper/figures/method_comparison.txt]

If ``--output -`` (or omitted) the LaTeX is written to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.common.method_comparison import (  # noqa: E402
    METHOD_COMPARISON_SCHEMA_VERSION,
    from_dict,
    to_ascii,
    to_latex,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="method_comparison.v1 JSON")
    p.add_argument(
        "--output",
        type=str,
        default="-",
        help="Output .tex path; '-' (default) writes to stdout.",
    )
    p.add_argument("--label", type=str, default=None, help=r"Optional \\label{...} value")
    p.add_argument("--decimals", type=int, default=1)
    p.add_argument("--ci-decimals", type=int, default=1)
    p.add_argument(
        "--also-ascii",
        type=Path,
        default=None,
        help="Optional path for a parallel ASCII rendering of the same table.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    in_path: Path = args.input
    if not in_path.exists():
        print(f"RENDER_FAILED: input does not exist: {in_path}", file=sys.stderr)
        return 1
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    schema = payload.get("schema_version")
    if schema != METHOD_COMPARISON_SCHEMA_VERSION:
        print(
            f"RENDER_FAILED: unexpected schema_version {schema!r}; expected "
            f"{METHOD_COMPARISON_SCHEMA_VERSION!r}",
            file=sys.stderr,
        )
        return 2
    table = from_dict(payload)
    latex = to_latex(
        table,
        decimals=args.decimals,
        ci_decimals=args.ci_decimals,
        label=args.label,
    )
    if args.output == "-" or args.output == "":
        sys.stdout.write(latex)
        sys.stdout.write("\n")
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(latex + "\n", encoding="utf-8")
    if args.also_ascii is not None:
        args.also_ascii.parent.mkdir(parents=True, exist_ok=True)
        args.also_ascii.write_text(
            to_ascii(table, decimals=args.decimals, ci_decimals=args.ci_decimals) + "\n",
            encoding="utf-8",
        )
    print(
        "RENDER_OK "
        f"input={in_path} "
        f"methods={len(table.methods)} "
        f"metrics={len(table.metrics)} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
