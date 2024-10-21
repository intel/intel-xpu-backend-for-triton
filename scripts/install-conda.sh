#!/bin/bash

install_env() {
    export PATH="$HOME/miniforge3/bin:$PATH"
    conda create -n triton --override-channels -c conda-forge python=$python_version.*
    conda env update -f scripts/triton.yml
    find /opt/intel/oneapi/ \( -name '*.so' -or -name '*.so.*' \) -exec cp -n {} $HOME/miniforge3/envs/triton/lib \;
    ln -snf /usr/include/level_zero $HOME/miniforge3/envs/triton/bin/../x86_64-conda-linux-gnu/sysroot/usr/include/level_zero
    find /usr -name libze_\* -exec cp -n {} $HOME/miniforge3/envs/triton/lib \;
    ln -snf /opt/intel/oneapi/compiler/2024.1/include/sycl $HOME/miniforge3/envs/triton/lib/python3.9/site-packages/triton/backends/intel/include
}

print_env_info() {
    conda info
    conda list -n triton
}

script_dir=$(dirname "$0")
source "$script_dir/env-util.sh"

install_env
print_env_info
