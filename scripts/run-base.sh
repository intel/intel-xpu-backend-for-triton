#!/bin/bash

install_env() {
  :
}

source /opt/intel/oneapi/setvars.sh >/dev/null

script_dir=$(dirname "$0")
source "$script_dir/run-util.sh"
source "$script_name"
