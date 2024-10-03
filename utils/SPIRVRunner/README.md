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

## Configuration

### Generate Data

In order to utilize this utility, Triton application must be run with following environment variables enabled
Provide the path to the directory where the serialized JSON, tensors and SPRI-V binary stored. It is recommended to clear triton cache.

```
export TRITON_XPU_DUMP_SPIRV_KERNEL_ARGS=< Absolute path to SPV Dumps >
```

Following input data is generated,

1. args_data.json - (Kernel Arguments / Grid Configuration)
2. tensors  (Tensors used by the kernel (.pt))
3. SPIR-V binary (.spv)


## Running

Help:
`./build/SPIRVRunner` < Output Tensor Title >

### Demo (01-vector-add.py)

`SPIRVRunner` is configured to run the `add_kernel.spv` SPIRV binary with inputs `tensor_0.pt` and `tensor_1.pt` and output `tensor_2.pt`. `add_kernel.spv` was generated from the `01-vector-add.py` tutorial.

SPIRVRunner Usage:
`./build/SPIRVRunner tensor_2`

Expected output follows:

```
Running on device: Intel(R) Data Center GPU Max 1100
Read 3772 byte kernel.
create kernel:add_kernel
Loaded kernel with 0 registers and 0 register spills.
Tensor output: [98432], Float (393728 bytes)
Kernel return output: 1.37129
[ CPUFloatType{} ]
```

The GPU hardware, shape and data type of each Tensor (along with number of bytes), and kernel information are printed. The shape and data type of the output Tensor is currently printed, along with the the first cell in the output. Ensuring the value of the first cell is non-zero allows for a quick sanity check. The output Tensor is written to a file `cpp_outs.pt` which is a Tensor in PyTorch format. Typically, we will create a quick Python script to read the input Tensor, run the same computations in PyTorch, and then compare the PyTorch result with the loaded `cpp_outs.pt` Tensor using the PyTorch testing API.
