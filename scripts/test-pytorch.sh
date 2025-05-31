#!/usr/bin/env bash

set -euo pipefail

cd pytorch
pip install -r .ci/docker/requirements-ci.txt

export

pytest test/inductor/test_flex_attention.py -n 8 --reruns 2
