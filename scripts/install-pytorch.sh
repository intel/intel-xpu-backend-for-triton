#!/usr/bin/env bash

set -euo pipefail

# Select what to install.
BUILD_PYTORCH=false
BUILD_LATEST=false
FORCE_REINSTALL=false
CHECK_WHEEL=false
PYTORCH_CURRENT_COMMIT=""
VENV=false
CLEAN=true
for arg in "$@"; do
  case $arg in
    --source)
      BUILD_PYTORCH=true
      ;;
    --latest)
      # Build from the latest pytorch commit in the main branch.
      BUILD_PYTORCH=true
      BUILD_LATEST=true
      ;;
    --force-reinstall)
      FORCE_REINSTALL=true
      ;;
    --check-wheel)
      # Check if PyTorch wheel exists
      CHECK_WHEEL=true
      ;;
    --venv)
      VENV=true
      ;;
    -nc|--no-clean)
      CLEAN=false
      ;;
    --help)
      echo "Example usage: ./install-pytorch.sh [--source | --latest | --force-reinstall | --check-wheel | --venv]"
      exit 1
      ;;
    *)
      echo "Unknown argument: $arg."
      exit 1
      ;;
  esac
done

if [ "$VENV" = true ]; then
  echo "**** Activate virtual environment *****"
  if [[ $OSTYPE = msys ]]; then
    source .venv/Scripts/activate
  else
    source .venv/bin/activate
  fi
fi

# intel-xpu-backend-for-triton project root
ROOT=$(cd "$(dirname "$0")/.." && pwd)

############################################################################
# Check installed torch

if [ "$BUILD_LATEST" = false ]; then
  PYTORCH_PINNED_COMMIT="$(<$ROOT/.github/pins/pytorch.txt)"
  echo "***** Using pinned PyTorch commit $PYTORCH_PINNED_COMMIT by default. *****"
fi

if pip show torch &>/dev/null; then
  PYTORCH_CURRENT_COMMIT="$(python -c 'import torch;print(torch.__version__)')"
  PYTORCH_CURRENT_COMMIT=${PYTORCH_CURRENT_COMMIT#*"git"}
  echo "**** PyTorch is already installed. Current commit is $PYTORCH_CURRENT_COMMIT ****"
  if [ "$BUILD_LATEST" = false ]; then
    if [[ "$PYTORCH_PINNED_COMMIT" = "$PYTORCH_CURRENT_COMMIT"* ]]; then
      echo "**** PyTorch is already installed and its current commit is equal to the pinned commit: $PYTORCH_PINNED_COMMIT. ****"
      echo "**** There is no need to build anything, just exit. ****"
      exit 0
    else
      echo "**** Current PyTorch commit $PYTORCH_CURRENT_COMMIT ****"
      echo "**** Pinned PyTorch commit $PYTORCH_PINNED_COMMIT ****"
    fi
  fi
  if [ "$FORCE_REINSTALL" = false ]; then
    echo "**** Exiting without action. ****"
    echo "**** INFO: Run the install-pytorch.sh script with the --force-reinstall flag to force reinstallation of PyTorch,
      or uninstall the current version of PyTorch manually. ****"
    exit 1
  fi
  pip uninstall -y torch
fi

############################################################################
# Check installed torch pinned dependencies

PINNED_TORCH_DEPENDENCIES_REGEX="^torchaudio==|^torchvision=="
INSTALLED_PINNED_TORCH_DEPENDENCIES=$(pip list --format=freeze | grep -iE "$PINNED_TORCH_DEPENDENCIES_REGEX" || true)

if [ -n "$INSTALLED_PINNED_TORCH_DEPENDENCIES" ]; then
  echo "**** The following pinned torch dependencies are installed: ****"
  echo
  echo "$INSTALLED_PINNED_TORCH_DEPENDENCIES"
  echo
  if [ "$FORCE_REINSTALL" = false ]; then
    echo "**** Exiting without action. ****"
    echo "**** INFO: Add --force-reinstall flag to force the PyTorch pinned dependencies re-installation. ****"
    echo "**** INFO: PyTorch pinned dependencies build from source mode is not supported. ****"
    exit 1
  fi
  pip uninstall -y torchaudio torchvision
fi

############################################################################
# Install PyTorch and pinned dependencies from nightly builds.

if [ "$BUILD_PYTORCH" = false ]; then
  if `! which gh &> /dev/null` || `! which jq &> /dev/null`; then
    echo "****** ERROR: gh or jq is missing. Please install gh and jq first. ******"
    exit 1
  fi
  echo "**** Download nightly builds. ****"
  PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  RUN_ID=$(gh run list --workflow nightly-wheels.yml --branch main -R intel/intel-xpu-backend-for-triton --json databaseId,conclusion | jq -r '[.[] | select(.conclusion=="success")][0].databaseId')
  TEMP_DIR=$(mktemp -d)
  WHEEL_PATTERN="wheels-pytorch-py${PYTHON_VERSION}*"
  gh run download $RUN_ID \
    --repo intel/intel-xpu-backend-for-triton \
    --pattern "$WHEEL_PATTERN" \
    --dir $TEMP_DIR
  cd $TEMP_DIR/$WHEEL_PATTERN
  echo "**** Install PyTorch and pinned dependencies from nightly builds. ****"
  pip install torch*
  rm -rf $TEMP_DIR
  exit 0
fi

############################################################################
# Configure, build and install PyTorch from source.

SCRIPTS_DIR=$ROOT/scripts
PYTORCH_PROJ=${PYTORCH_PROJ:-$ROOT/.scripts_cache/pytorch}
BASE=$(dirname "$PYTORCH_PROJ")

echo "**** BASE is set to $BASE ****"
echo "**** PYTORCH_PROJ is set to $PYTORCH_PROJ ****"
mkdir -p $BASE

function pytorch_wheel_exists {
  if [[ ! -d $PYTORCH_PROJ/dist ]]; then
    return 1
  fi
  PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
  PYTORCH_VERSION=$(<$PYTORCH_PROJ/version.txt)
  PYTORCH_COMMIT=${PYTORCH_PINNED_COMMIT:-main}
  if [[ $OSTYPE = msys ]]; then
    PYTORCH_OS=win
    PYTORCH_ARCH="amd64"
  else
    PYTORCH_OS=linux
    PYTORCH_ARCH="x86_64"
  fi
  PYTORCH_WHEEL_NAME="torch-${PYTORCH_VERSION}+git${PYTORCH_COMMIT:0:7}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-${PYTORCH_OS}_${PYTORCH_ARCH}.whl"
  if [[ -f $PYTORCH_PROJ/dist/$PYTORCH_WHEEL_NAME ]]; then
    echo "**** $PYTORCH_WHEEL_NAME exists ****"
    return 0
  else
    echo "**** $PYTORCH_WHEEL_NAME does not exist ****"
    return 1
  fi
}

function build_pytorch {
  if [ "$CLEAN" = true ]; then
    if [ -d "$PYTORCH_PROJ" ] && cd "$PYTORCH_PROJ" && \
      git fetch --recurse-submodules && \
      git reset --hard ${PYTORCH_PINNED_COMMIT:-main} && \
      git submodule update --init --recursive && \
      git clean -xffd; then
      echo "**** Cleaning $PYTORCH_PROJ before build ****"
    else
      cd $BASE
      rm -rf "$PYTORCH_PROJ"
      echo "**** Cloning PyTorch into $PYTORCH_PROJ ****"
      git clone --single-branch -b main --recurse-submodules https://github.com/pytorch/pytorch.git
      cd "$PYTORCH_PROJ"

      if [ "$BUILD_LATEST" = false ]; then
        git checkout $PYTORCH_PINNED_COMMIT
        git submodule update --init --recursive
        git clean -xffd
      fi
    fi

    # Apply Triton specific patches to PyTorch.
    $SCRIPTS_DIR/patch-pytorch.sh
  fi

  echo "****** Building $PYTORCH_PROJ ******"
  cd "$PYTORCH_PROJ"
  # FIXME: Compatibility with versions of CMake older than 3.5 has been removed, this breaks compilation of third_party/protobuf:
  # CMake Error at third_party/protobuf/cmake/CMakeLists.txt:2 (cmake_minimum_required)
  pip install 'cmake<4.0.0'
  pip install -r requirements.txt
  pip install cmake ninja
  USE_XCCL=1 USE_STATIC_MKL=1 python setup.py bdist_wheel
}

function install_pytorch {
  echo "****** Installing PyTorch ******"
  cd "$PYTORCH_PROJ"
  pip install dist/*.whl
}

if [ "$CHECK_WHEEL" = false ] || ! pytorch_wheel_exists; then
  build_pytorch
fi
install_pytorch

# Change working directory to avoid the following error:
#
# ImportError: Failed to load PyTorch C extensions:
#     ...
#     This error can generally be solved using the `develop` workflow
#         $ python setup.py develop && python -c "import torch"  # This should succeed
#     or by running Python from a different directory.
cd $BASE
python -c "import torch;print(torch.__version__)"
