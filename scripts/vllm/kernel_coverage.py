#!/usr/bin/env python3
"""Report which vLLM tests launch the given kernels, from kernel-test-map-per-file.sh output."""

import argparse
import json
import pathlib
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("maps_dir", type=pathlib.Path, help="directory containing the recorded *.json files")
parser.add_argument("kernels", nargs="*", help="kernel names (default: read whitespace-separated from stdin)")
parser.add_argument("--tests", action="store_true", help="list full test ids instead of test files")
args = parser.parse_args()

names = args.kernels or sys.stdin.read().split()

tests_by_kernel = {}
for path in sorted(args.maps_dir.rglob("*.json")):
    for kernel, tests in json.loads(path.read_text(encoding="utf-8")).items():
        tests_by_kernel.setdefault(kernel, set()).update(tests)

for name in names:
    tests = tests_by_kernel.get(name)
    if not tests:
        print(f"[ ] {name}")
        continue
    print(f"[x] {name}")
    for item in sorted(tests if args.tests else {t.split("::", 1)[0] for t in tests}):
        print(f"      {item}")
