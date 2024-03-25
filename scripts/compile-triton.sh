#!/usr/bin/env bash

set -euo pipefail

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
      ARGS+="${arg} "
      shift
      ;;
  esac
done

if [ "$BUILD_LLVM" = false ] && [ "$BUILD_TRITON" = false ]; then
  BUILD_LLVM=true
  BUILD_TRITON=true
fi

if [ ! -v BASE ]; then
  echo "**** BASE is not given *****"
  BASE=$(cd $(dirname "$0")/../.. && pwd)
  echo "**** Default BASE is set to $BASE ****"
fi

export PACKAGES_DIR=$BASE/packages
export SPIRV_TOOLS=$PACKAGES_DIR/spirv-tools
export LLVM_PROJ=$BASE/llvm-project
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
  pip install ninja cmake wheel
elif [ -v VIRTUAL_ENV ]; then
  echo "**** Cleaning up Python virtualenv ****"
  deactivate
fi

check_rc() {
  if [ $? != 0 ]; then
    echo "Command failed with rc: $rc"
    exit 1
  fi
}

if [ ! -d "$PACKAGES_DIR" ]; then
  mkdir $PACKAGES_DIR
fi
if [ $BASE != $HOME ]; then
  ln -sfT $PACKAGES_DIR $HOME/packages
fi

############################################################################
# Download the Kronos SPIRV-Tools.

if [ ! -d "$SPIRV_TOOLS" ]; then
  cd $PACKAGES_DIR
  wget https://storage.googleapis.com/spirv-tools/artifacts/prod/graphics_shader_compiler/spirv-tools/linux-clang-release/continuous/2129/20230718-114856/install.tgz
  tar -xvf install.tgz
  mv install spirv-tools
  rm install.tgz
  # Fix up the prefix path.
  sed -i s#prefix=/tmpfs/src/install#prefix=$SPIRV_TOOLS# spirv-tools/lib/pkgconfig/SPIRV-Tools.pc
  sed -i s#prefix=/tmpfs/src/install#prefix=$SPIRV_TOOLS# spirv-tools/lib/pkgconfig/SPIRV-Tools-shared.pc
fi

############################################################################
# Clone the Triton project fork if it does not exists.

if [ ! -d "$TRITON_PROJ" ]; then
  echo "****** Cloning $TRITON_PROJ ******"
  cd $BASE
  git clone https://github.com/intel/intel-xpu-backend-for-triton.git -b llvm-target
fi

############################################################################
# Clone the LLVM repository if it does not exists, and checkout the commit used by Triton.

if [ ! -d "$LLVM_PROJ" ]; then
  echo "**** Cloning $LLVM_PROJ ****"
  cd $BASE
  git clone --recursive https://github.com/llvm/llvm-project.git

  TRITON_LLVM_COMMIT_FILE="$TRITON_PROJ/cmake/llvm-hash.txt"
  if [ ! -f "$TRITON_LLVM_COMMIT_FILE" ]; then
    echo "ERROR: TRITON LLVM commit file $TRITON_LLVM_COMMIT_FILE not found."
    abort
  fi

  TRITON_LLVM_COMMIT="$(<$TRITON_LLVM_COMMIT_FILE)"
  cd $LLVM_PROJ
  git checkout $TRITON_LLVM_COMMIT
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

if [ ! -d "$LLVM_PROJ_BUILD" ]
then
  mkdir $LLVM_PROJ_BUILD
fi

build_llvm() {
  echo "****** Configuring $LLVM_PROJ ******"

  ADDITIONAL_FLAGS=""
  if [ "$CCACHE" = true ]
  then
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS -DCMAKE_C_COMPILER_LAUNCHER=ccache"
    ADDITIONAL_FLAGS="$ADDITIONAL_FLAGS -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
  fi

  cd $LLVM_PROJ_BUILD
  cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_DUMP=1 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_ASSERTIONS=true \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER=$C_COMPILER \
    -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
    $ADDITIONAL_FLAGS

  echo "****** Building $LLVM_PROJ ******"
  ninja
  check_rc
  ninja install
  check_rc
  ninja check-mlir
  check_rc
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

  export LLVM_SYSPATH=$BASE/llvm-project/build
  export DEBUG=1
  if [ "$CCACHE" = true ]
  then
    export TRITON_BUILD_WITH_CCACHE=true
  fi

  cd python
  pip install -e .
  check_rc

  # Install triton tests.
  pip install -vvv -e '.[tests]'
  check_rc

  # Copy compile_commands.json in the build directory (so that cland vscode plugin can find it).
  cp $TRITON_PROJ_BUILD/"$(ls $TRITON_PROJ_BUILD)"/compile_commands.json $TRITON_PROJ/
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
