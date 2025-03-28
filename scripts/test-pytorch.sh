#!/usr/bin/env bash

set -euo pipefail

cd pytorch
pip install -r .ci/docker/requirements-ci.txt

export

test_cmd="python test/run_test.py --keep-going --include "
if [[ "${1:-all}" = "all" ]]; then
    for test in $(ls test/inductor | grep test); do
        test_cmd="${test_cmd} inductor/$test"
    done
else
    for test in "$@"; do
        test_cmd="${test_cmd} $test"
    done
fi

eval $test_cmd
