#!/bin/bash

set -vxe

link_sycl() {
    mkdir -p $HOME/miniforge3/envs/triton/$1
    ln -snf /opt/intel/oneapi/compiler/2024.1/include/sycl $HOME/miniforge3/envs/triton/$1/
    ln -snf /opt/intel/oneapi/compiler/2024.1/include/sycl/CL $HOME/miniforge3/envs/triton/$1/
}

install_env() {
    export PATH="$HOME/miniforge3/bin:$PATH"
    conda create -n triton --override-channels -c conda-forge python=$python_version.*
    conda env update -f scripts/triton.yml

    ln -snf /usr/include/level_zero $HOME/miniforge3/envs/triton/x86_64-conda-linux-gnu/sysroot/usr/include/level_zero
    find /usr -name libze_\* -exec ln -sf {} $HOME/miniforge3/envs/triton/lib/ \;

    find /opt/intel/oneapi/mkl/2025.0/lib/ \( -name '*.so' -or -name '*.so.*' \) -exec cp -n {} $HOME/miniforge3/envs/triton/lib \;
    find /opt/intel/oneapi/compiler/2024.1/lib/ \( -name '*.so' -or -name '*.so.*' \) -exec cp -n {} $HOME/miniforge3/envs/triton/lib \;

    python -m venv ./.venv; source ./.venv/bin/activate
    pip3 install intel-sycl-rt

#    ln -snf /opt/intel/oneapi/compiler/2024.1/include/sycl $HOME/miniforge3/envs/triton/include/sycl

#    find /opt/intel/oneapi/ \( -name '*.so' -or -name '*.so.*' \) -exec cp -n {} $HOME/miniforge3/envs/triton/lib \;
#    link_sycl lib/python$python_version/site-packages/triton/backends/intel/include
#    link_sycl x86_64-conda-linux-gnu/sysroot/usr/include
    find / -name libpti_view.so.0.9
    exit 1
}

print_env_info() {
    conda info
    conda list -n triton
}

script_dir=$(dirname "$0")
source "$script_dir/env-util.sh"

install_env
print_env_info