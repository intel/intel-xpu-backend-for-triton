#!/bin/bash

source /opt/intel/oneapi/setvars.sh >/dev/null

script_dir=$(dirname "$0")
source "$script_dir/env-util.sh"
source "$script_name"
