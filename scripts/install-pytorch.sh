#!/usr/bin/env bash

set -euo pipefail

# Select what to install.
BUILD_PYTORCH=false
BUILD_LATEST=false
FORCE_REINSTALL=false
PYTORCH_CURRENT_COMMIT=""
VENV=false
for arg in "$@"; do
  case $arg in
    --source)
      BUILD_PYTORCH=true
      shift
      ;;
    --latest)
      # Build from the latest pytorch commit in the main branch.
      BUILD_PYTORCH=true
      BUILD_LATEST=true
      shift
      ;;
    --force-reinstall)
      FORCE_REINSTALL=true
      shift
      ;;
    --venv)
      VENV=true
      shift
      ;;
    --help)
      echo "Example usage: ./install-pytorch.sh [--source | --latest | --force-reinstall | --venv]"
      exit 1
      ;;
    *)
      ARGS+="${arg} "
      shift
      ;;
  esac
done

set +o xtrace

if [ "$VENV" = true ]; then
  echo "**** Activate virtual environment *****"
  source .venv/bin/activate
fi

# intel-xpu-backend-for-triton project root
ROOT=$(cd $(dirname "$0")/.. && pwd)

############################################################################
# Check installed torch

if [ "$BUILD_LATEST" = false ]; then
  PYTORCH_PINNED_COMMIT="$(<$ROOT/.github/pins/pytorch-upstream.txt)"
  echo -e "***** Using pinned PyTorch commit $PYTORCH_PINNED_COMMIT by default. *****\n"
fi

if pip show torch &>/dev/null; then
  PYTORCH_CURRENT_COMMIT=`python -c "import torch;print(torch.__version__)"`
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
    echo "**** INFO: Add --force-reinstall flag to force the PyTorch re-installation. ****"
    exit 0
  fi
  pip uninstall -y torch
fi

############################################################################
# Check installed torch pinned dependencies

INSTALLED_PINNED_TORCH_DEPENDENCIES=$(pip list --format=freeze | grep -iE "$PINNED_TORCH_DEPENDENCIES_REGEX" || true)

if [ -n "$INSTALLED_PINNED_TORCH_DEPENDENCIES" ]; then
  echo -e "**** The following pinned torch dependencies are installed: ****\n"
  echo -e "$INSTALLED_PINNED_TORCH_DEPENDENCIES\n"
  if [ "$FORCE_REINSTALL" = false ]; then
    echo "**** Exiting without action. ****"
    echo "**** INFO: Add --force-reinstall flag to force the PyTorch pinned dependencies re-installation. ****"
    echo "**** INFO: PyTorch pinned dependencies build from source mode is not supported. ****"
    exit 0
  fi
  pip uninstall -y torchtext torchaudio torchvision
fi

############################################################################
# Install PyTorch and pinned dependencies from nightly builds.

if [ "$BUILD_PYTORCH" = false ]; then
  if `! which gh &> /dev/null` || `! which jq &> /dev/null`; then
    echo "****** ERROR: gh or jq is missing. Please install gh and jq first. ******"
    exit 1
  fi
  echo "**** Download nightly builds. ****"
  PYTHON_VERSION=$( python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" )
  RUN_ID=$(gh run list -w "Triton wheels" -R intel/intel-xpu-backend-for-triton --json databaseId,conclusion | jq -r '[.[] | select(.conclusion=="success")][0].databaseId')
  TEMP_DIR=$(mktemp -d)
  WHEEL_PATTERN="wheels-pytorch-py${PYTHON_VERSION}*"
  gh run download $RUN_ID \
    --repo intel/intel-xpu-backend-for-triton \
    --pattern "$WHEEL_PATTERN" \
    --dir $TEMP_DIR
  cd $TEMP_DIR/$WHEEL_PATTERN
  echo "**** Install PyTorch and pinned dependencies from nightly builds. ****"
  pip install torch*
  rm -r $TEMP_DIR
  exit 0
fi

############################################################################
# Configure, build and install PyTorch from source.

if [ ! -v BASE ]; then
  echo "**** BASE is not given *****"
  BASE=$ROOT/.scripts_cache
  if [ ! -d "$BASE" ]; then
    mkdir $BASE
  fi
  echo -e "**** Default BASE is set to $BASE ****\n"
fi

PYTORCH_PROJ=$BASE/pytorch
SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "**** Cleaning $PYTORCH_PROJ before build ****"
if rm -rf $PYTORCH_PROJ &>/dev/null; then
  echo "**** $PYTORCH_PROJ have been cleaned ****"
fi

echo "**** Cloning $PYTORCH_PROJ ****"
cd $BASE
git clone --single-branch -b main --recurse-submodules https://github.com/pytorch/pytorch.git

cd $PYTORCH_PROJ

if [ "$BUILD_LATEST" = false ]; then
  git fetch --all
  git checkout $PYTORCH_PINNED_COMMIT
  git submodule update --recursive
fi

# Apply Triton specific patches to PyTorch.
$SCRIPTS_DIR/patch-pytorch.sh

echo "****** Building $PYTORCH_PROJ ******"
pip install -r requirements.txt
pip install cmake ninja mkl-static mkl-include
python setup.py bdist_wheel

echo "****** Installing PyTorch ******"
pip install dist/*.whl
# Change working directory to avoid the following error:
#
# ImportError: Failed to load PyTorch C extensions:
#     ...
#     This error can generally be solved using the `develop` workflow
#         $ python setup.py develop && python -c "import torch"  # This should succeed
#     or by running Python from a different directory.
cd $BASE
python -c "import torch;print(torch.__version__)"
