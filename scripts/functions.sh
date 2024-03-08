#!/usr/bin/env bash

set -ueo pipefail

set_env() {
  TRITON_PROJ=$(cd "$SCRIPTS_DIR/.." && pwd -P)
  BASE=$(dirname "$TRITON_PROJ")
  PACKAGES_DIR="$BASE/packages"
  SPIRV_TOOLS="$PACKAGES_DIR/spirv-tools"
  LLVM_PROJ="$BASE/llvm"
  TRITON_PROJ_BUILD="$TRITON_PROJ/python/build"
}

print_failed() {
  local code
  local lineno
  local msg

  code="$1"
  lineno="$2"
  msg="$3"
  echo "FAILED ($code): $lineno: $msg"
}
trap 'print_failed $? $LINENO "$BASH_COMMAND"' ERR

