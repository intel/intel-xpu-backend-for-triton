#!/usr/bin/env bash

tar -xf intel-xpu-backend-for-triton.tar.gz
tar -xf wheels.tar.gz

# cp run.sh ~

source $PREFIX/etc/profile.d/conda.sh

conda activate base
pip install wheels/*

mkdir -p ~/.local/bin
tar Jxf vulkan-sdk.tar.xz -C $HOME/.local/bin --strip-components 3 --no-anchored spirv-dis

/bin/sh ./intel-deep-learning-essentials_offline.sh -a --silent --eula accept
