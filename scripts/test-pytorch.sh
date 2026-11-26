#!/usr/bin/env bash

set -euo pipefail

# Provides the `pip` wrapper (pip or `uv pip`).
source "$(dirname "$0")/pip-utils.sh"

cd pytorch
pip install -r .ci/docker/requirements-ci.txt

export

test_cmd="python test/run_test.py --keep-going --include "
if [[ "${1:-all}" = "all" ]]; then
    # Use run_test.py's own test discovery: listing test/inductor also picks up
    # non-test entries and blocklisted files, which --include rejects.
    for test in $(python -c 'from tools.testing.discover_tests import TESTS; print(" ".join(t for t in TESTS if t.startswith("inductor/")))'); do
        test_cmd="${test_cmd} $test"
    done
else
    for test in "$@"; do
        test_cmd="${test_cmd} $test"
    done
fi

eval $test_cmd
