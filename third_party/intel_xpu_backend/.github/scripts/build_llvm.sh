set -e -x
git clone https://github.com/quintinwang5/llvm-project.git -b triton_rebase_5e5a22ca ./private-llvm

CURRENT_DIR="$(pwd)"

SOURCE_DIR=$CURRENT_DIR/private-llvm
install_prefix=""
platform=""
build_config=""
arch="x86"
num_jobs=8

usage() {
  echo "Usage: bash build_llvm.bash -o INSTALL_PREFIX -p PLATFORM -c CONFIG [-a ARCH] [-j NUM_JOBS]"
  echo "Ex: bash build_llvm.bash -o llvm-16.0.0-x86_64-linux-gnu-ubuntu-18.04 -p docker_ubuntu-18.04 -c assert -j 16"
  echo "INSTALL_PREFIX = <string> # \${INSTALL_PREFIX}.tar.xz is created"
  echo "PLATFORM       = {local|docker_ubuntu_18.04|docker_centos7}"
  echo "CONFIG         = {release|assert|debug}"
  echo "ARCH           = {x86|arm64}"
  echo "NUM_JOBS       = {1|2|3|...}"
  exit 1;
}

while getopts "o:p:c:a:j:l" arg; do
	case "$arg" in
		o)
			install_prefix="$OPTARG"
			;;
		p)
			platform="$OPTARG"
			;;
		c)
			build_config="$OPTARG"
			;;
		a)
			arch="$OPTARG"
			;;
		j)
			num_jobs="$OPTARG"
			;;
		*)
			usage
			;;

	esac
done

if [ x"$install_prefix" == x ] || [ x"$platform" == x ] || [ x"$build_config" == x ]; then
  usage
fi

# Set up CMake configurations
CMAKE_CONFIGS="-DLLVM_ENABLE_PROJECTS=mlir -DLLVM_USE_LINKER=gold -DLLVM_ENABLE_LTO=OFF"
if [ x"$arch" == x"arm64" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS}"
else
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DLLVM_TARGETS_TO_BUILD=X86;NVPTX;AMDGPU"
fi

if [ x"$build_config" == x"release" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DCMAKE_BUILD_TYPE=Release"
elif [ x"$build_config" == x"assert" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DCMAKE_BUILD_TYPE=MinSizeRel -DLLVM_ENABLE_ASSERTIONS=True"
elif [ x"$build_config" == x"debug" ]; then
  CMAKE_CONFIGS="${CMAKE_CONFIGS} -DCMAKE_BUILD_TYPE=Debug"
else
  usage
fi


# Create a temporary build directory
BUILD_DIR="$(mktemp -d)"
echo "Using a temporary directory for the build: $BUILD_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"


# Build LLVM locally
pushd "$BUILD_DIR"
echo ${CMAKE_CONFIGS}
cmake "$SOURCE_DIR/llvm" -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/$install_prefix" $CMAKE_CONFIGS
make -j${num_jobs} install
tar -cJf "${SOURCE_DIR}/${install_prefix}.tar.xz" "$install_prefix"
popd

mv $SOURCE_DIR/$install_prefix.tar.xz $(pwd)
# Remove the temporary directory
rm -rf "$BUILD_DIR"

echo "Completed!"
