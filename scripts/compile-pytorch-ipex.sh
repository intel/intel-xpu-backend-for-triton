#!/usr/bin/env bash

set -euo pipefail

# intel-xpu-backend-for-triton project root
ROOT=$(cd $(dirname "$0")/.. && pwd)
SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Select what to build.
BUILD_PYTORCH=false
UPSTREAM_PYTORCH=false
BUILD_IPEX=false
NO_OP_IPEX=false
BUILD_PINNED=false
BUILD_FROM_SOURCE=false
CLEAN=false
VENV=false
for arg in "$@"; do
  case $arg in
    --pytorch)
      BUILD_PYTORCH=true
      shift
      ;;
    --upstream-pytorch)
      UPSTREAM_PYTORCH=true
      shift
      ;;
    --ipex)
      BUILD_IPEX=true
      shift
      ;;
    --no-op-ipex)
      NO_OP_IPEX=true
      shift
      ;;
    --pinned)
      # Build the pinned commit (from source if --source is used, otherwise download the wheel from github)
      BUILD_PINNED=true
      shift
      ;;
    --source)
      BUILD_FROM_SOURCE=true
      shift
      ;;
    --clean)
      CLEAN=true
      shift
      ;;
    --venv)
      VENV=true
      shift
      ;;
    --help)
      echo "Example usage: ./compile-pytorch-ipex.sh [--pytorch | --upstream-pytorch | --ipex | --no-op-ipex | --pinned | --source | --clean | --venv]"
      exit 1
      ;;
    *)
      ARGS+="${arg} "
      shift
      ;;
  esac
done

set +o xtrace

if [ "$BUILD_PYTORCH" = true ] && [ "$UPSTREAM_PYTORCH" = true ]; then
  echo "***** Use '--pytorch' or '--upstream-pytorch' *****"
  exit 1
fi

if [ "$BUILD_PYTORCH" = false ] && [ "$UPSTREAM_PYTORCH" = false ]; then
  echo "***** Use upstream pytorch by the default *****"
  UPSTREAM_PYTORCH=true
fi

if [ "$BUILD_PINNED" = false ] && [ "$BUILD_FROM_SOURCE" = false ]; then
  echo "***** Use pinned pytorch by the default *****"
  BUILD_PINNED=true
fi

if [ "$BUILD_IPEX" = true ] && [ "$NO_OP_IPEX" = true ]; then
  echo "***** Use '--ipex' or '--no-op-ipex' *****"
  exit 1
fi

if [ "$BUILD_PYTORCH" = true ] && [ "$NO_OP_IPEX" = true ]; then
  echo "***** Use of '--no-op-ipex' isn't allowed with '--pytorch' *****"
  exit 1
fi

if [ "$UPSTREAM_PYTORCH" = true ] && [ "$BUILD_IPEX" = true ]; then
  echo "***** Use of '--ipex' isn't allowed with '--upstream-pytorch' *****"
  exit 1
fi

if [ ! -v BASE ]; then
  echo "**** BASE is not given *****"
  BASE=$ROOT/.scripts_cache
  if [ ! -d "$BASE" ]; then
    mkdir $BASE
  fi
  echo "**** Default BASE is set to $BASE ****"
fi

if [ "$VENV" = true ]; then
  echo "**** Activate virtual environment *****"
  source .venv/bin/activate
fi

export PYTORCH_PROJ=$BASE/pytorch
export PYTORCH_PROXY_REPO=pytorch-stonepia
if [ "$UPSTREAM_PYTORCH" = false ]; then
  export PYTORCH_PROJ=$BASE/$PYTORCH_PROXY_REPO
fi

export IPEX_PROJ=$BASE/intel-extension-for-pytorch

if [ "$CLEAN" = true ]; then
  echo "**** Cleaning $PYTORCH_PROJ and $IPEX_PROJ before build ****"
  if rm -rf $PYTORCH_PROJ $IPEX_PROJ &>/dev/null; then
    pip uninstall -y torch intel_extension_for_pytorch
  fi
fi

if [ "$UPSTREAM_PYTORCH" = true ]; then
  # This is a simplification that allows not to use two flags at the same time.
  # It's convenient because `UPSTREAM_PYTORCH` flag only specifies the repository
  # and has no meaning without `BUILD_PYTORCH` flag.
  BUILD_PYTORCH=true
  # Only no-op IPEX works with PyTorch upstream.
  NO_OP_IPEX=true
fi

if [ "$NO_OP_IPEX" = true ]; then
  # This is a simplification that allows not to use two flags at the same time.
  # It's convenient because `NO_OP_IPEX` has no meaning without `BUILD_IPEX` flag.
  BUILD_IPEX=true
fi

if [ "$BUILD_PINNED" = true ]; then
  echo "**** Determine if the installed PyTorch version is the same as the pinned version. ****"
  if [ "$UPSTREAM_PYTORCH" = true ]; then
    PYTORCH_PINNED_COMMIT="$(<$ROOT/.github/pins/pytorch-upstream.txt)"
  else
    PYTORCH_PINNED_COMMIT="$(<$ROOT/.github/pins/pytorch.txt)"
  fi

  BUILD_PYTORCH=true
  if pip show torch &>/dev/null; then
    PYTORCH_CURRENT_COMMIT=`python -c "import torch;print(torch.__version__)"`
    PYTORCH_CURRENT_COMMIT=${PYTORCH_CURRENT_COMMIT#*"git"}
    if [[ "$PYTORCH_PINNED_COMMIT" = "$PYTORCH_CURRENT_COMMIT"* ]]; then
      echo "**** PyTorch is already installed and its current commit is equal to the pinned commit: $PYTORCH_PINNED_COMMIT. ****"
      BUILD_PYTORCH=false
    else
      echo "**** Current PyTorch commit $PYTORCH_CURRENT_COMMIT ****"
      echo "**** Pinned PyTorch commit $PYTORCH_PINNED_COMMIT ****"
    fi
  fi

  echo "**** Determine if the installed IPEX version is the same as the pinned version. ****"

  if [ "$NO_OP_IPEX" = true ]; then
    # should be in sync with `create-noop-ipex.sh`
    IPEX_PINNED_COMMIT="2.4.0+noop"
  else
    IPEX_PINNED_COMMIT="$(<$ROOT/.github/pins/ipex.txt)"
  fi

  BUILD_IPEX=true
  if pip show intel_extension_for_pytorch &>/dev/null; then
    IPEX_CURRENT_COMMIT=`python -c "import intel_extension_for_pytorch as ipex;print(ipex.__version__)"`
    IPEX_CURRENT_COMMIT=${IPEX_CURRENT_COMMIT#*"git"}
    if [[ "$IPEX_PINNED_COMMIT" = "$IPEX_CURRENT_COMMIT"* ]]; then
      echo "**** IPEX is already installed and its current commit is equal to the pinned commit: $IPEX_PINNED_COMMIT. ****"
      BUILD_IPEX=false
    else
      IPEX_CURRENT_COMMIT=${IPEX_CURRENT_COMMIT#*"git"}
      if [[ $IPEX_PINNED_COMMIT = $IPEX_CURRENT_COMMIT* ]]; then
        echo "**** IPEX is already installed and its current commit is equal to the pinned commit: $IPEX_PINNED_COMMIT. ****"
        BUILD_IPEX=false
      else
        echo "**** Current IPEX commit $IPEX_CURRENT_COMMIT ****"
        echo "**** Pinned IPEX commit $IPEX_PINNED_COMMIT ****"
      fi
    fi
  fi

  if [[ "$BUILD_PYTORCH" = false && "$BUILD_IPEX" = false ]]; then
    echo "**** There is no need to build anything, just exit. ****"
    exit 0
  fi
fi

if [ "$BUILD_FROM_SOURCE" = false ]; then
  if `! which gh &> /dev/null` || `! which jq &> /dev/null`; then
    echo "****** WARNING: gh or jq is missing ******"
    BUILD_FROM_SOURCE=true
  fi
fi

if [ "$BUILD_FROM_SOURCE" = false ]; then
  echo "**** Download nightly builds. ****"
  PYTHON_VERSION=$( python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" )
  RUN_ID=$(gh run list -w "Triton wheels" -R intel/intel-xpu-backend-for-triton --json databaseId,conclusion | jq -r '[.[] | select(.conclusion=="success")][0].databaseId')
  TEMP_DIR=$(mktemp -d)
  WHEEL_PATTERN="wheels-py${PYTHON_VERSION}*"
  if [ "$UPSTREAM_PYTORCH" = true ]; then
    WHEEL_PATTERN="wheels-pytorch-py${PYTHON_VERSION}*"
  fi
  gh run download $RUN_ID \
    --repo intel/intel-xpu-backend-for-triton \
    --pattern "$WHEEL_PATTERN" \
    --dir $TEMP_DIR
  cd $TEMP_DIR/$WHEEL_PATTERN
  echo "**** Install PyTorch from nightly builds. ****"
  pip install torch-*
  if [ "$NO_OP_IPEX" = true ]; then
    echo "**** Setup no-op IPEX ****"
    python $SCRIPTS_DIR/create-noop-ipex.py
  else
    echo "**** Install IPEX from nightly builds. ****"
    pip install intel_extension_for_pytorch-*
  fi
  rm -r $TEMP_DIR
  exit 0
fi


if [ "$BUILD_PYTORCH" = false ] && [ "$BUILD_IPEX" = false ]; then
  # Avoid overriding the user's existing PyTorch by default.
  if ! pip show torch &>/dev/null; then
    echo "**** pip didn't find PyTorch, so it will be built from sources. ****"
    BUILD_PYTORCH=true
  fi
  # Avoid overriding the user's existing IPEX by default.
  if ! pip show intel_extension_for_pytorch &>/dev/null; then
    echo "**** pip didn't find IPEX, so it will be built from sources. ****"
    BUILD_IPEX=true
  fi
  if [ "$BUILD_PINNED" = true ]; then
    echo "**** Even if pip sees PyTorch and IPEX in the environment they will be build from sources since '--pinned' option is used. ****"
    BUILD_PYTORCH=true
    BUILD_IPEX=true
  fi
fi

############################################################################
# Configure and build the pytorch project.

build_pytorch() {
  if [ ! -d "$PYTORCH_PROJ" ]; then
    echo "**** Cloning $PYTORCH_PROJ ****"
    cd $BASE
    if [ "$UPSTREAM_PYTORCH" = true ]; then
      git clone --single-branch -b main --recurse-submodules https://github.com/pytorch/pytorch.git
      pushd $PYTORCH_PROJ
      $SCRIPTS_DIR/patch-pytorch.sh
      popd
    else
      git clone --single-branch -b dev/triton-test-3.0 --recurse-submodules --jobs 8 https://github.com/Stonepia/pytorch.git $PYTORCH_PROXY_REPO
    fi
  fi
  echo "****** Building $PYTORCH_PROJ ******"
  cd $PYTORCH_PROJ
  if [ ! -d "$PYTORCH_PROJ/dist" ]; then
    if [ "$BUILD_PINNED" = true ]; then
      git fetch --all
      git checkout $PYTORCH_PINNED_COMMIT
      git submodule update --recursive
    fi
    pip install cmake ninja
    pip install mkl-static mkl-include
    pip install -r requirements.txt
    python setup.py bdist_wheel
  fi
  pip install dist/*.whl
  cd $BASE
  python -c "import torch;print(torch.__version__)"
}

############################################################################
# Configure and build the ipex project.

build_ipex() {
  if [ "$NO_OP_IPEX" = true ]; then
    echo "**** Setup no-op IPEX ****"
    python $SCRIPTS_DIR/create-noop-ipex.py
  else
    if [ ! -d "$IPEX_PROJ" ]; then
      cd $BASE
      echo "**** Cloning $IPEX_PROJ ****"
      git clone --single-branch -b dev/triton-test-3.0 --recurse-submodules --jobs 8 https://github.com/intel/intel-extension-for-pytorch.git
    fi
    echo "****** Building $IPEX_PROJ ******"
    cd $IPEX_PROJ
    if [ ! -d "$IPEX_PROJ/dist" ]; then
      if [ "$BUILD_PINNED" = true ]; then
        git fetch origin $IPEX_PINNED_COMMIT
        git checkout $IPEX_PINNED_COMMIT
        git submodule sync
        git submodule update --init --recursive
      fi
      pip install -r requirements.txt
      python setup.py bdist_wheel
    fi
    pip install dist/*.whl
  fi
  cd $BASE
  python -c "import intel_extension_for_pytorch as ipex;print(ipex.__version__)"
}

build() {
  if [ "$BUILD_PYTORCH" = true ]; then
    build_pytorch
  fi
  if [ "$BUILD_IPEX" = true ]; then
    build_ipex
  fi
}

build
