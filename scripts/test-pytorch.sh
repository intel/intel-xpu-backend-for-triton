#!/usr/bin/env bash

set -euo pipefail

cd pytorch
pip install -r .ci/docker/requirements-ci.txt

export

# warmup cache
pytest test/inductor/test_flex_attention.py -n 16 || true
echo PYTEST_FINISHED

python test/run_test.py --keep-going --include inductor/test_flex_attention.py
