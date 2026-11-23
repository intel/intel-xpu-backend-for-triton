#!/usr/bin/env bash

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 TEST_FILE..." >&2
    exit 1
fi

# Provides the `pip` wrapper (pip or `uv pip`).
source "$(dirname "$0")/pip-utils.sh"

cd pytorch
pip install -r .ci/docker/requirements-ci.txt

export

test_cmd="python test/run_test.py --keep-going --include "
for test in "$@"; do
    test_cmd="${test_cmd} $test"
done

eval $test_cmd
