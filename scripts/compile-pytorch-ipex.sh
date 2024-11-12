#!/usr/bin/env bash

set -euo pipefail

# Select what to install.
BUILD_PYTORCH=true
BUILD_IPEX=true
FORCE_REINSTALL=false
VENV=false
for arg in "$@"; do
  case $arg in
    --force-reinstall)
      FORCE_REINSTALL=true
      shift
      ;;
    --venv)
      VENV=true
      shift
      ;;
    --help)
      echo "Example usage: ./compile-pytorch-ipex.sh [ --force-reinstall | --venv]"
      exit 1
      ;;
    *)
      echo "Unknown argument $arg."
      exit 1
      ;;
  esac
done

if [ "$VENV" = true ]; then
  echo "**** Activate virtual environment *****"
  source .venv/bin/activate
fi

############################################################################
# Setup directories

# intel-xpu-backend-for-triton project root
ROOT=$(cd $(dirname "$0")/.. && pwd)
SCRIPTS_DIR=$ROOT/scripts
BASE=$ROOT/.scripts_cache

echo "**** BASE is set to $BASE ****"
if [ ! -d "$BASE" ]; then
  mkdir $BASE
fi

############################################################################
# Check installed torch

PYTORCH_PINNED_COMMIT="$(<$ROOT/.github/pins/pytorch.txt)"
IPEX_PINNED_COMMIT="$(<$ROOT/.github/pins/ipex.txt)"
echo "***** Using pinned PyTorch commit $PYTORCH_PINNED_COMMIT by default. *****"
echo "***** Using pinned IPEX commit $IPEX_PINNED_COMMIT by default. *****"

if pip show torch &>/dev/null; then
  PYTORCH_CURRENT_COMMIT="$(python -c 'import torch;print(torch.__version__)')"
  PYTORCH_CURRENT_COMMIT=${PYTORCH_CURRENT_COMMIT#*"git"}
  echo "**** PyTorch is already installed. Current commit is $PYTORCH_CURRENT_COMMIT ****"
  if [[ "$PYTORCH_PINNED_COMMIT" = "$PYTORCH_CURRENT_COMMIT"* ]]; then
    echo "**** PyTorch is already installed and its current commit is equal to the pinned commit: $PYTORCH_PINNED_COMMIT. ****"
    BUILD_PYTORCH=false
  else
    echo "**** Current PyTorch commit $PYTORCH_CURRENT_COMMIT ****"
    echo "**** Pinned PyTorch commit $PYTORCH_PINNED_COMMIT ****"
    if [ "$FORCE_REINSTALL" = false ]; then
      echo "**** Exiting without action. ****"
      echo "**** INFO: Run the install-pytorch script with the --force-reinstall flag to force reinstallation of PyTorch,
        or uninstall the current version of PyTorch manually. ****"
      exit 1
    fi
    pip uninstall -y torch
  fi
fi

############################################################################
# Check installed torch pinned dependencies

if [[ $BUILD_PYTORCH = true ]]; then
  PINNED_TORCH_DEPENDENCIES_REGEX="^torchtext==|^torchaudio==|^torchvision=="
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
    pip uninstall -y torchtext torchaudio torchvision
  fi
fi

############################################################################
# Configure, build and install PyTorch from source.

if [[ $BUILD_PYTORCH = true ]]; then
  PYTORCH_PROJ=$BASE/pytorch-stonepia

  echo "**** Cleaning $PYTORCH_PROJ before build ****"
  rm -rf $PYTORCH_PROJ

  echo "**** Cloning $PYTORCH_PROJ ****"
  cd $BASE
  git clone --single-branch -b dev/triton-test-3.0 --recurse-submodules --jobs 8 https://github.com/Stonepia/pytorch.git

  cd $PYTORCH_PROJ

  git fetch --all
  git checkout $PYTORCH_PINNED_COMMIT
  git submodule update --recursive

  echo "****** Building $PYTORCH_PROJ ******"
  pip install -r requirements.txt
  pip install cmake ninja "numpy<2.0"
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
fi

# IPEX cannot be checked and installed before torch

############################################################################
# Check installed IPEX

if pip show intel_extension_for_pytorch &>/dev/null; then
  IPEX_CURRENT_COMMIT="$(python -c 'import intel_extension_for_pytorch as ipex;print(ipex.__version__)')"
  IPEX_CURRENT_COMMIT=${IPEX_CURRENT_COMMIT#*"git"}
  echo "**** IPEX is already installed. Current commit is $IPEX_CURRENT_COMMIT ****"
  if [[ "$IPEX_PINNED_COMMIT" = "$IPEX_CURRENT_COMMIT"* ]]; then
    echo "**** IPEX is already installed and its current commit is equal to the pinned commit: $IPEX_PINNED_COMMIT. ****"
    BUILD_IPEX=false
  else
    echo "**** Current IPEX commit $IPEX_CURRENT_COMMIT ****"
    echo "**** Pinned IPEX commit $IPEX_PINNED_COMMIT ****"
    if [ "$FORCE_REINSTALL" = false ]; then
      echo "**** Exiting without action. ****"
      echo "**** INFO: Run the install-pytorch script with the --force-reinstall flag to force reinstallation of IPEX,
        or uninstall the current version of IPEX manually. ****"
      exit 1
    fi
    pip uninstall -y intel_extension_for_pytorch
  fi
fi

############################################################################
# Configure, build and install IPEX from source.

if [[  $BUILD_IPEX = true ]]; then
  IPEX_PROJ=$BASE/intel-extension-for-pytorch

  echo "**** Cleaning $IPEX_PROJ before build ****"
  rm -rf $IPEX_PROJ

  echo "**** Cloning $IPEX_PROJ ****"
  cd $BASE
  git clone --single-branch -b dev/triton-test-3.0 --recurse-submodules --jobs 8 https://github.com/intel/intel-extension-for-pytorch.git

  cd $IPEX_PROJ

  git fetch origin $IPEX_PINNED_COMMIT
  git checkout $IPEX_PINNED_COMMIT
  git submodule sync
  git submodule update --init --recursive

  echo "****** Building $IPEX_PROJ ******"
  pip install -r requirements.txt
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
  python -c "import intel_extension_for_pytorch as ipex;print(ipex.__version__)"
fi

if [[ $BUILD_PYTORCH = false && $BUILD_IPEX = false ]]; then
  echo "**** There is no need to build anything, just exit. ****"
fi
