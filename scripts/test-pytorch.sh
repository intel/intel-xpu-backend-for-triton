#!/usr/bin/env bash

set -euo pipefail

cd pytorch
pip install -r .ci/docker/requirements-ci.txt

export

test_cmd="python test/run_test.py --keep-going --include inductor/test_select_algorithm.py -k test_convolution1"

eval $test_cmd
