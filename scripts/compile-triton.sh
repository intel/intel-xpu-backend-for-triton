#!/usr/bin/env bash

set -euo pipefail

export PIP_DISABLE_PIP_VERSION_CHECK=1

# Select what to build.
BUILD_LLVM=false
BUILD_TRITON=false
CLEAN=false
VENV=false
CCACHE=false
for arg in "$@"; do
  case $arg in
    --llvm)
      BUILD_LLVM=true
      shift
      ;;
    --triton)
      BUILD_TRITON=true
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
    --ccache)
      CCACHE=true
      shift
      ;;
    --help)
      echo "Example usage: ./compile-triton.sh [--llvm | --triton | --clean | --venv | --ccache]"
      exit 1
      ;;
    *)
      echo "Unknown argument: $arg."
      exit 1
      ;;
  esac
done

if [ "$BUILD_LLVM" = false ] && [ "$BUILD_TRITON" = false ]; then
  BUILD_TRITON=true
fi

if [ ! -v BASE ]; then
  echo "**** BASE is not given *****"
  BASE=$(cd $(dirname "$0")/../.. && pwd)
  echo "**** Default BASE is set to $BASE ****"
fi

export PACKAGES_DIR=$BASE/packages
export LLVM_PROJ=$BASE/llvm
export LLVM_PROJ_BUILD=$LLVM_PROJ/build
export TRITON_PROJ=$BASE/intel-xpu-backend-for-triton
export TRITON_PROJ_BUILD=$TRITON_PROJ/python/build

if [ "$CLEAN" = true ]; then
  echo "**** Cleaning $PACKAGES_DIR, $LLVM_PROJ, and $TRITON_PROJ_BUILD before build ****"
  rm -rf $PACKAGES_DIR $LLVM_PROJ $TRITON_PROJ_BUILD
fi

if [ "$VENV" = true ]; then
  echo "**** Creating Python virtualenv ****"
  python3 -m venv .venv --prompt triton
  source .venv/bin/activate
  pip install ninja cmake wheel pybind11
fi

if [ ! -d "$PACKAGES_DIR" ]; then
  mkdir $PACKAGES_DIR
fi
if [ $BASE != $HOME ]; then
  ln -sfT $PACKAGES_DIR $HOME/packages
fi

############################################################################
# Clone the Triton project fork if it does not exists.

if [ ! -d "$TRITON_PROJ" ]; then
  echo "****** Cloning $TRITON_PROJ ******"
  cd $BASE
  git clone https://github.com/intel/intel-xpu-backend-for-triton.git
fi

############################################################################
## Configure and build the llvm project.

if [ ! -v C_COMPILER ]; then
  C_COMPILER=${GCC:-$(which gcc)}
  echo "**** C_COMPILER is set to $C_COMPILER ****"
fi
if [ ! -v CXX_COMPILER ]; then
  CXX_COMPILER=${GXX:-$(which g++)}
  echo "**** CXX_COMPILER is set to $CXX_COMPILER ****"
fi

build_llvm() {

  # Clone LLVM repository
  if [ ! -d "$LLVM_PROJ" ]; then
    echo "**** Cloning $LLVM_PROJ ****"
    cd $BASE
    LLVM_COMMIT_ID="$(<$BASE/intel-xpu-backend-for-triton/cmake/llvm-hash.txt)"
    git clone --recurse-submodules --jobs 8 https://github.com/llvm/llvm-project.git llvm
    cd llvm
    git checkout $LLVM_COMMIT_ID
    git submodule update --recursive
  fi

  echo "****** Configuring $LLVM_PROJ ******"

  ADDITIONAL_FLAGS=""
  if [ "$CCACHE" = true ]
  then
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS -DCMAKE_C_COMPILER_LAUNCHER=ccache"
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
  fi

  if [ ! -d "$LLVM_PROJ_BUILD" ]
  then
    mkdir $LLVM_PROJ_BUILD
  fi

  cd $LLVM_PROJ_BUILD
  cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_DUMP=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=true \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DLLVM_INSTALL_UTILS=true \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX=$PACKAGES_DIR/llvm \
    -DCMAKE_C_COMPILER=$C_COMPILER \
    -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
    $ADDITIONAL_FLAGS

  echo "****** Building $LLVM_PROJ ******"
  ninja
  ninja install
  ninja check-mlir
}

############################################################################
## Configure and build the Triton project.

if [ ! -d "$TRITON_PROJ_BUILD" ]
then
  # Remove the cached triton (see setup.py, get_triton_cache_path())
  if [ -n "$HOME" ]
  then
    rm -fr "$HOME/.triton/"
  elif [ -n "$USERPROFILE" ]
  then
    rm -fr "$USERPROFILE/.triton/"
  elif [ -n "$HOMEPATH" ]
  then
    rm -fr "$HOMEPATH/.triton/"
  fi
fi

build_triton() {
  echo "**** Configuring $TRITON_PROJ ****"
  cd $TRITON_PROJ

  if [ "$BUILD_LLVM" = true ]; then
    export LLVM_SYSPATH=$PACKAGES_DIR/llvm
  fi
  export DEBUG=1
  if [ "$CCACHE" = true ]
  then
    export TRITON_BUILD_WITH_CCACHE=true
  fi

  cd python
  # Install triton and its dependencies.
  pip install -v -e '.[build,tests]'

  # Copy compile_commands.json in the build directory (so that cland vscode plugin can find it).
  cp $(find $TRITON_PROJ_BUILD -name compile_commands.json) $TRITON_PROJ/
}

build() {
  if [ "$BUILD_LLVM" = true ]; then
    build_llvm
  fi

  if [ "$BUILD_TRITON" = true ]; then
    build_triton
  fi
}

build
