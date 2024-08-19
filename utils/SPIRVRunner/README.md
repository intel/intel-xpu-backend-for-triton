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

## Configuring

`SPIRVRunner` is configured to run the `add_kernel.spv` SPIRV binary with inputs `x.py` and `y.py`. `add_kernel.spv` was generated from the `01-vector-add.py` tutorial. 

Kernels of different shapes require modifying parameters manually in the `SPIRVRunner`. Two places require modification:

1. `launchKernel`: Add input Tensors to the function signature, add arguments as variables within the function. Arguments can be pulled from the `args` variable to `XPULauncher.__call__` method in `driver.py`. Arguments should be passed to the `sycl_kernel_launch` function. Note that we currently rely on `sycl::memcpy` to move the PyTorch Tensor to XPU. In later versions of PyTorch we should be able to delegate this responsibility to `PyTorch`, and pass the raw XPU `data_ptr()` from `PyTorch` to the kernel.
2. `sycl_kernel_launch`: Place all `arg*` parameters into the `params` array and add an appropriate call to `set_scalar_arg` for each param, which tells `SYCL` what the arguments are for the kernel we are going to launch. 

## Running 

Once the `SPIRVRunner` has been appropriately configured for the kernel and inputs, run the binary with no arguments:

`./build/SPIRVRunner`

Expected output follows:

```
Running on device: Intel(R) Data Center GPU Max 1100
Tensor a: [98432], Float (393728 bytes)
Tensor b: [98432], Float (393728 bytes)
Read 3772 byte kernel.
Loaded kernel with 0 registers and 0 register spills.
Tensor output: [98432], Float (393728 bytes)
Kernel return output: 1.37129
[ CPUFloatType{} ]
```

The GPU hardware, shape and data type of each Tensor (along with number of bytes), and kernel information are printed. The shape and data type of the output Tensor is currently printed, along with the the first cell in the output. Ensuring the value of the first cell is non-zero allows for a quick sanity check. The output Tensor is written to a file `cpp_outs.pt` which is a Tensor in PyTorch format. Typically, we will create a quick Python script to read the input Tensor, run the same computations in PyTorch, and then compare the PyTorch result with the loaded `cpp_outs.pt` Tensor using the PyTorch testing API. 