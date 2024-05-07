#!/bin/bash

QUIET=false
for arg in "$@"; do
  case $arg in
    -q|--quiet)
      QUIET=true
      shift
      ;;
    --help)
      echo "Example usage: ./capture-hw-detauls.sh [-q | --quiet]"
      exit 1
      ;;
    *)
      ARGS+="${arg} "
      shift
      ;;
  esac
done

export LIBIGC1_VERSION=$(dpkg-query -W --showformat='${version}\n' libigc1 | grep -oP '.+(?=~)')
export LEVEL_ZERO_VERSION=$(dpkg-query -W --showformat='${version}\n' intel-level-zero-gpu | grep -oP '.+(?=~)')
export GPU_DEVICE=$(clinfo --json | jq -r '.devices[].online[].CL_DEVICE_NAME')
export AGAMA_VERSION=$(dpkg-query -W --showformat='${version}\n' libigc1 | sed 's/.*-\(.*\)~.*/\1/')

if [ "$QUIET" = false ]; then
    echo "LIBIGC1_VERSION=$LIBIGC1_VERSION"
    echo "LEVEL_ZERO_VERSION=$LEVEL_ZERO_VERSION"
    echo "GPU_DEVICE=$GPU_DEVICE"
    echo "AGAMA_VERSION=$AGAMA_VERSION"
fi
