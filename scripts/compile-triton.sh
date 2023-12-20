#!/usr/bin/env bash

set +o xtrace
if [ ! -d "$BASE" ]; then
  echo "**** BASE is not given *****"
  echo "**** Default BASE is set to /iusers/$USER ****"
  BASE=/iusers/$USER
fi

export RSYNC_PROXY="proxy-us.intel.com:912"
export http_proxy="http://proxy-us.intel.com:912"
export https_proxy="http://proxy-us.intel.com:912"
export ftp_proxy="http://proxy-us.intel.com:912"
export socks_proxy="http://proxy-us.intel.com:1080"

CMAKE=/usr/bin/cmake
export PACKAGES_DIR=$BASE/packages
export SPIRV_TOOLS=$PACKAGES_DIR/spirv-tools
export LLVM_PROJ=$BASE/llvm
export SPIRV_LLVM_TRANSLATOR_PROJ=$BASE/SPIRV-LLVM-Translator
export TRITON_PROJ=$BASE/intel-xpu-backend-for-triton
export TRITON_PROJ_BUILD=$TRITON_PROJ/python/build

function check_rc {
  if [ $? != 0 ]; then
    echo "Command failed with rc: $rc"
    exit 1
  fi
}

if [[ "$1" == "--clean" ]]; then
  rm -rf $PACKAGES_DIR $LLVM_PROJ $SPIRV_LLVM_TRANSLATOR_PROJ $TRITON_PROJ_BUILD
fi

if [ ! -d "$PACKAGES_DIR" ]; then
  mkdir $PACKAGES_DIR
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

C_COMPILER=/usr/bin/gcc
CXX_COMPILER=/usr/bin/g++

if [ ! -d "$LLVM_PROJ_BUILD" ]
then
  mkdir $LLVM_PROJ_BUILD
fi

function build_llvm {
  echo "****** Configuring $LLVM_PROJ ******"

  cd $LLVM_PROJ_BUILD
  $CMAKE -G Ninja ../llvm \
    -DLLVM_ENABLE_DUMP=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=true \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="SPIRV" \
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
# Clone the SPIRV-LLVM translator fork if it does not exists.

export SPIRV_LLVM_TRANSLATOR_PROJ_BUILD=$SPIRV_LLVM_TRANSLATOR_PROJ/build

if [ ! -d "$SPIRV_LLVM_TRANSLATOR_PROJ" ]; then
  echo "****** Cloning $SPIRV_LLVM_TRANSLATOR_PROJ ******"
  cd $BASE
  git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
fi

############################################################################
## Configure and build the SPIRV-LLVM translator project.

if [ ! -d "$SPIRV_LLVM_TRANSLATOR_PROJ_BUILD" ]
then
  mkdir $SPIRV_LLVM_TRANSLATOR_PROJ_BUILD
fi

function build_spirv_translator {
  echo "**** Configuring $SPIRV_LLVM_TRANSLATOR_PROJ ****"

  cd $SPIRV_LLVM_TRANSLATOR_PROJ_BUILD
  PKG_CONFIG_PATH=$SPIRV_TOOLS/lib/pkgconfig/ $CMAKE -G Ninja ..\
    -DLLVM_DIR=$PACKAGES_DIR/llvm/lib/cmake/llvm \
    -DLLVM_SPIRV_BUILD_EXTERNAL=YES \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INSTALL_PREFIX=$PACKAGES_DIR/llvm-spirv ..

  echo "**** Building $SPIRV_LLVM_TRANSLATOR_PROJ ****"
  ninja
  check_rc
  ninja install
  check_rc
}
build_spirv_translator

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

if [ ! -d "$TRITON_PROJ_BUILD" ]
then
  # Remove the cached triton.
  rm -fr /iusers/$USERS/.triton
fi

function build_triton {
  echo "**** Configuring $TRITON_PROJ ****"
  cd $TRITON_PROJ
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
