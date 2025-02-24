#!/bin/bash

link_sycl() {
    mkdir -p $HOME/miniforge3/envs/triton/$1
    ln -snf /opt/intel/oneapi/compiler/latest/include/sycl $HOME/miniforge3/envs/triton/$1/
    ln -snf /opt/intel/oneapi/compiler/latest/include/sycl/CL $HOME/miniforge3/envs/triton/$1/
}

install_env() {
    export PATH="$HOME/miniforge3/bin:$PATH"
    conda create -n triton --override-channels -c conda-forge python=$python_version.*
    conda env update -f scripts/triton.yml
    find /opt/intel/oneapi/ \( -name '*.so' -or -name '*.so.*' \) -exec cp -n {} $HOME/miniforge3/envs/triton/lib \;
    ln -snf /usr/include/level_zero $HOME/miniforge3/envs/triton/bin/../x86_64-conda-linux-gnu/sysroot/usr/include/level_zero
    find /usr -name libze_\* -exec cp -n {} $HOME/miniforge3/envs/triton/lib \;
    link_sycl lib/python$python_version/site-packages/triton/backends/intel/include
    link_sycl x86_64-conda-linux-gnu/sysroot/usr/include
}

print_env_info() {
    conda info
    conda list -n triton
}

script_dir=$(dirname "$0")
source "$script_dir/env-util.sh"

install_env
print_env_info
