#!/usr/bin/env bash

saved_args=("$@")

source "run_triton_tests.sh" --install-only
rc=$?
if (( rc != 0 )); then
    echo "Failed to install Triton xpu test env" >&2
    exit $rc
fi

cd $TRITON_PROJ

test -d pytorch || (
    git clone https://github.com/pytorch/pytorch
    rev=$(cat .github/pins/pytorch.txt)
    cd pytorch
    git checkout $rev
  )

cd pytorch
pip install -r .ci/docker/requirements-ci.txt

export PYTORCH_TESTING_DEVICE_ONLY_FOR="xpu"
export TRITON_LESS_FLEX_ATTN_BWD_CONFIGS="1"

echo "Forwarding arguments: ${saved_args[@]}"
python $TRITON_PROJ/scripts/test_pytorch.py "${saved_args[@]}"

exit_code=$?

conda deactivate

set PYTHONPATH=$OLD_PATH
echo "Revert python path to old value $PYTHONPATH"

exit $exit_code
