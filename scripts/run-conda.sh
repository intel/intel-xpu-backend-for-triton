#!/bin/bash

source /opt/intel/oneapi/setvars.sh >/dev/null

script_dir=$(dirname "$0")
source "$script_dir/env-util.sh"

export PATH="$HOME/miniforge3/bin:$PATH"

conda run --no-capture-output -n triton bash "$script_name"
