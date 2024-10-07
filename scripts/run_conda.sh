#!/bin/bash

set -vx

install_conda() {
    conda create -n triton --override-channels -c conda-forge python=${{ matrix.python }}.*
    conda env update -f scripts/triton.yml
    find /opt/intel/oneapi/ \( -name '*.so' -or -name '*.so.*' \) -exec cp -n {} $HOME/miniforge3/envs/triton/lib \;
    ln -snf /usr/include/level_zero $HOME/miniforge3/envs/triton/bin/../x86_64-conda-linux-gnu/sysroot/usr/include/level_zero
    find /usr -name libze_\* -exec cp -n {} $HOME/miniforge3/envs/triton/lib \;
    conda info
    conda list -n triton
}


test -d $HOME/miniforge3/envs/triton || install_conda

source /opt/intel/oneapi/setvars.sh >/dev/null
conda run --no-capture-output -n triton bash $1
