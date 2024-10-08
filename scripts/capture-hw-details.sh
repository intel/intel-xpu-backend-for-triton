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

if dpkg-query --show libigc1 &> /dev/null; then
    export LIBIGC1_VERSION=$(dpkg-query --show --showformat='${version}\n' libigc1 | grep -oP '.+(?=~)')
else
    export LIBIGC1_VERSION="Not Installed"
fi

if dpkg-query --show libze1 &> /dev/null; then
    export LEVEL_ZERO_VERSION=$(dpkg-query --show --showformat='${version}\n' libze1 | grep -oP '.+(?=~)')
elif dpkg-query --show intel-level-zero-gpu &> /dev/null; then
    export LEVEL_ZERO_VERSION=$(dpkg-query --show --showformat='${version}\n' intel-level-zero-gpu | grep -oP '.+(?=~)')
else
    export LEVEL_ZERO_VERSION="Not Installed"
fi

if dpkg-query --show libigc1 &> /dev/null; then
    export AGAMA_VERSION=$(dpkg-query --show --showformat='${version}\n' libigc1 | sed 's/.*-\(.*\)~.*/\1/')
else
    export AGAMA_VERSION="Not Installed"
fi

if command -v clinfo &> /dev/null; then
    export GPU_DEVICE=$(clinfo --json | jq -r '[.devices[].online[] | select(.CL_DEVICE_TYPE.raw == 4)][0].CL_DEVICE_NAME')
elif command -v nvidia-smi &> /dev/null; then
    export GPU_DEVICE=$(nvidia-smi -L | sed -e 's,\(.*\) (UUID.*),\1,')
else
    export GPU_DEVICE="Not Installed"
fi

if python -c "import torch" &> /dev/null; then
    export TORCH_VERSION=$(python -c "import torch; from packaging.version import Version; print(Version(torch.__version__).base_version)")
else
    export TORCH_VERSION="Not installed"
fi

if icpx --version &> /dev/null; then
    export COMPILER_VERSION=$(icpx --version | grep "DPC++/C++ Compiler" | sed 's/.*(\(.*\))/\1/' | cut -d '.' -f 1-3)
else
    export COMPILER_VERSION="Not installed"
fi

if [ "$QUIET" = false ]; then
    echo "LIBIGC1_VERSION=$LIBIGC1_VERSION"
    echo "LEVEL_ZERO_VERSION=$LEVEL_ZERO_VERSION"
    echo "AGAMA_VERSION=$AGAMA_VERSION"
    echo "GPU_DEVICE=$GPU_DEVICE"
    echo "TORCH_VERSION=$TORCH_VERSION"
    echo "COMPILER_VERSION=$COMPILER_VERSION"
    if [[ ! "${BENCHMARKING_METHOD:-}" = "" ]]; then
        echo "BENCHMARKING_METHOD=$BENCHMARKING_METHOD"
    fi
fi
