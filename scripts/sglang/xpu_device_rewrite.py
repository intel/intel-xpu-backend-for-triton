#!/usr/bin/env python3
"""
Rewrite hardcoded CUDA device references to XPU in SGLang's kernel test tree.

Only device *selection* is rewritten; occurrences of "cuda" in kernel names,
comments and skip reasons are left alone so that failures stay readable. The
rewrite is idempotent.

Usage:
  xpu_device_rewrite.py [--check] [--quiet] PATH [PATH ...]

Arguments:
  PATH: files or directories to rewrite (directories are walked for ``*.py``).

Options:
  --check: report what would change, do not write.
  --quiet: only print the totals.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (name, pattern, replacement). `device-const` matches identifiers *ending* in
# device/dev, so `device_type="cuda"` is left alone: tests construct CUDA
# platform descriptors to assert that a capability matcher rejects them.
_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("device-kwarg", re.compile(r"""(\bdevice\s*=\s*)(['"])cuda(:\d+)?\2"""), r'\1"xpu\3"'),
    ("device-const", re.compile(r"""(\b[A-Za-z_]*(?:DEVICE|device|DEV|dev)\s*=\s*)(['"])cuda(:\d+)?\2"""),
     r'\1"xpu\3"'),
    ("torch-device", re.compile(r"""(torch\.device\(\s*)(['"])cuda(:\d+)?\2"""), r'\1"xpu\3"'),
    ("torch-device-fstr", re.compile(r"""(torch\.device\(\s*f)(['"])cuda:"""), r"\1\g<2>xpu:"),
    ("to-str", re.compile(r"""(\.to\(\s*)(['"])cuda(:\d+)?\2"""), r'\1"xpu\3"'),
    ("dot-cuda", re.compile(r"""\.cuda\(\s*\)"""), '.to("xpu")'),
    ("torch-cuda-api", re.compile(r"""\btorch\.cuda\.(?=[A-Za-z_])"""), "torch.xpu."),
]


def rewrite_text(text: str) -> tuple[str, dict[str, int]]:
    """Apply every rule, returning the new text and a per-rule substitution count."""
    counts: dict[str, int] = {}
    for name, pattern, repl in _RULES:
        text, n = pattern.subn(repl, text)
        if n:
            counts[name] = counts.get(name, 0) + n
    return text, counts


def iter_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"WARNING: no such path, skipping: {p}", file=sys.stderr)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", metavar="PATH")
    parser.add_argument("--check", action="store_true", help="report only, do not write")
    parser.add_argument("--quiet", action="store_true", help="only print the totals")
    args = parser.parse_args()

    totals: dict[str, int] = {}
    changed = 0
    files = iter_files(args.paths)
    for path in files:
        original = path.read_text(encoding="utf-8", errors="surrogateescape")
        rewritten, counts = rewrite_text(original)
        if not counts:
            continue
        changed += 1
        for name, n in counts.items():
            totals[name] = totals.get(name, 0) + n
        if not args.quiet:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            verb = "would rewrite" if args.check else "rewrote"
            print(f"{verb} {path}: {detail}")
        if not args.check:
            path.write_text(rewritten, encoding="utf-8", errors="surrogateescape")

    summary = ", ".join(f"{k}={v}" for k, v in sorted(totals.items())) or "no changes"
    print(f"**** xpu-device-rewrite: {changed}/{len(files)} files, {summary} ****")
    return 0


if __name__ == "__main__":
    sys.exit(main())
