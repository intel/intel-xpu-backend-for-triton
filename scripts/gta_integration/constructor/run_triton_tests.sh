#!/usr/bin/env bash

# Save all arguments into one string
ALL_ARGS="$*"

# Clear all positional arguments to prevent them from being passed as an argument to the conda activation script
set --

echo "Pythonpath: $PYTHONPATH"
echo "Replace python path"
export OLD_PATH=$PYTHONPATH
echo "Env old_path: $OLD_PATH"
export PYTHONPATH=/home/gta/setup_triton/lib/python3.12/site-packages
echo "New pythonpath: $PYTHONPATH"

if [ -z "$HOME" ]; then
    echo "HOME variable is not set or empty, setting it to /root"
    # export HOME="/root"
    export HOME="/gta"
fi

if [ -d "$HOME/triton-xpu" ]; then
  echo "Triton xpu test env is already created."
else
  echo "Installing Triton xpu test env"
  bash TritonXPU-GTA.sh
fi

export TRITON_PROJ="$HOME/triton-xpu/snapshot"
# Fix the git error fatal: detected dubious ownership in repository at '/root/triton-xpu/intel-xpu-backend-for-triton'
git config --global --add safe.directory "$TRITON_PROJ"

source "$HOME/triton-xpu/bin/activate"
source /opt/intel/oneapi/setvars.sh --force

cd $TRITON_PROJ

# This env variable might be no longer needed
export TRITON_RELAX_PROFILING_CHECK=1
echo "$ALL_ARGS"
python3 scripts/test_triton.py --home-dir $HOME $ALL_ARGS
exit_code=$?

conda deactivate

set PYTHONPATH=$OLD_PATH
echo "Revert python path to old value $PYTHONPATH"

exit $exit_code
