#!/usr/bin/env bash

set +o xtrace
if [ ! -z "$BASE" ]; then
  echo "**** BASE is not given *****"
  BASE=$(cd $(dirname "$0")/../.. && pwd)
  echo "**** Default BASE is set to $BASE ****"
fi

export PACKAGES_DIR=$BASE/packages
export SPIRV_TOOLS=$PACKAGES_DIR/spirv-tools
export LLVM_PROJ=$BASE/llvm
export TRITON_PROJ=$BASE/intel-xpu-backend-for-triton
export TRITON_PROJ_BUILD=$TRITON_PROJ/python/build

function check_rc {
  if [ $? != 0 ]; then
    echo "Command failed with rc: $rc"
    exit 1
  fi
}

CLEAN=false
VENV=false
SKIP_TRITON=false
for arg in "$@"; do
  case $arg in
    --clean)
      CLEAN=true
      shift
      ;;
    --venv)
      VENV=true
      shift
      ;;
    --skip-triton)
      SKIP_TRITON=true
      shift
      ;;
  esac
done

if [ "$CLEAN" = true ]; then
  echo "**** Cleaning $PACKAGES_DIR , $LLVM_PROJ , and $TRITON_PROJ_BUILD before build ****"
  rm -rf $PACKAGES_DIR $LLVM_PROJ $TRITON_PROJ_BUILD
fi

if [ ! -d "$PACKAGES_DIR" ]; then
  mkdir $PACKAGES_DIR
fi
if [ $BASE != $HOME ]; then
  ln -s $PACKAGES_DIR $HOME/packages
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
# Clone the LLVM repository (with GENX dialect).

export LLVM_PROJ_BUILD=$LLVM_PROJ/build

if [ ! -d "$LLVM_PROJ" ]; then
  echo "**** Cloning $LLVM_PROJ ****"
  cd $BASE

  git clone --recursive https://github.com/intel/llvm.git -b genx
fi

############################################################################
## Configure and build the llvm project.

if [ ! -z "$C_COMPILER" ]; then
  C_COMPILER=`which gcc`
  echo "**** C_COMPILER is set to $C_COMPILER ****"
fi
if [ ! -z "$CXX_COMPILER" ]; then
  CXX_COMPILER=`which g++`
  echo "**** CXX_COMPILER is set to $CXX_COMPILER ****"
fi

if [ ! -d "$LLVM_PROJ_BUILD" ]
then
  mkdir $LLVM_PROJ_BUILD
fi

function build_llvm {
  echo "****** Configuring $LLVM_PROJ ******"

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
    -DCMAKE_CXX_COMPILER=$CXX_COMPILER ..

  echo "****** Building $LLVM_PROJ ******"
  ninja
  check_rc
  ninja install
  check_rc
  ninja check-mlir
  check_rc
}
build_llvm

############################################################################
# Install libGenISAIntrinsics.a

cp $LLVM_PROJ/mlir/lib/Target/LLVMIR/Dialect/GENX/libGenISAIntrinsics.a $PACKAGES_DIR/llvm/lib

############################################################################
# Clone the Triton project fork if it does not exists.

if [ ! -d "$TRITON_PROJ" ]
then
  echo "****** Cloning $TRITON_PROJ ******"
  cd $BASE
  git clone https://github.com/intel/intel-xpu-backend-for-triton.git -b llvm-target
fi

############################################################################
## Configure and build the Triton project.

if [ "$SKIP_TRITON" = true ]; then
  exit 0
fi

if [ ! -d "$TRITON_PROJ_BUILD" ]
then
  # Remove the cached triton.
  rm -fr /iusers/$USERS/.triton
fi

function build_triton {
  echo "**** Configuring $TRITON_PROJ ****"
  cd $TRITON_PROJ

  if [ "$VENV" = true ]; then
    echo "**** Creating Python virtualenv ****"
    python3 -m venv .venv --prompt triton
    source .venv/bin/activate
    pip install ninja cmake wheel
  fi

  export LLVM_SYSPATH=$PACKAGES_DIR/llvm
  export DEBUG=1
  cd python
  pip install -e .
  check_rc

  # Install triton tests.
  pip install -vvv -e '.[tests]'
  check_rc

  # Copy compile_commands.json in the build directory (so that cland vscode plugin can find it).
  cp $TRITON_PROJ_BUILD/"$(ls $TRITON_PROJ_BUILD)"/compile_commands.json $TRITON_PROJ/
}
build_triton
