#!/bin/bash

install_conda() {
    conda create -n triton --override-channels -c conda-forge python=$python_version.*
    conda env update -f scripts/triton.yml
    find /opt/intel/oneapi/ \( -name '*.so' -or -name '*.so.*' \) -exec cp -n {} $HOME/miniforge3/envs/triton/lib \;
    ln -snf /usr/include/level_zero $HOME/miniforge3/envs/triton/bin/../x86_64-conda-linux-gnu/sysroot/usr/include/level_zero
    find /usr -name libze_\* -exec cp -n {} $HOME/miniforge3/envs/triton/lib \;
    conda info
    conda list -n triton
}

source /opt/intel/oneapi/setvars.sh >/dev/null

set -vx

script_dir=$(dirname "$0")
source "$script_dir/run-util.sh"

export PATH="$HOME/miniforge3/bin:$PATH"
test -d "$HOME/miniforge3/envs/triton" || install_conda
print_conda_info

conda run --no-capture-output -n triton bash "$script_name"
