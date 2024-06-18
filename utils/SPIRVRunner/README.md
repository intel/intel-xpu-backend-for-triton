# SPIRV Runner

A utility program for running Triton-generated SPIR-V kernels with identical inputs outside of Triton.

## Building

`SPIRVRunner` depends on Torch. If you build Triton with virtualenvs, you can easily find your torch library path by running
```
find .venv -name TorchConfig.cmake
```
in the top level Triton directory.

```
mkdir build
cd build
CMAKE_PREFIX_PATH=/abs/path/to/TorchConfig.cmake/FromAbove/ cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j
```
