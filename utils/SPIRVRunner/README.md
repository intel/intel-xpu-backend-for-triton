# SPIRV Runner

A utility program for running Triton-generated SPIR-V kernels with identical inputs outside of Triton.

## Building

`SPIRVRunner` depends on Torch.

If you build Triton with venv, you can easily find your torch library path by running the following command in the top level Triton directory:

```
find .venv -name TorchConfig.cmake
```

Alternatively, you can find `TorchConfig.cmake` with the following Python script:

```python
import importlib.metadata

for f in importlib.metadata.files('torch'):
  if 'TorchConfig.cmake' in f.name:
    print(f.locate().resolve())
```

`SPIRVRunner` depends on LLVM support library for argument parsing in order to use this run following in the top level Triton directory.

```
scripts/compile-triton.sh --llvm
```

SPIR-V Runner build steps:

```
mkdir build
cd build
CMAKE_PREFIX_PATH=/abs/path/to/TorchConfig.cmake/directory LLVM_DIR=/abs/path/to/packages/llvm cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j
```

## Configuration

### Generate Data

In order to utilize this utility, Triton application must be run with following environment variables enabled
Provide the path to the directory where the serialized JSON, tensors and SPIR-V binary stored. It is recommended to clear triton cache.

```
export TRITON_XPU_DUMP_SPIRV_KERNEL_ARGS=< Absolute path to SPV Dumps >
```

Following input data is generated,

1. args_data.json - (Kernel Arguments / Grid Configuration)
2. tensors  (Tensors used by the kernel (.pt))
3. SPIR-V binary (.spv)


## Running

Help:

```
USAGE: SPIRVRunner [options]

General options:

  -o <string> - <Specify Output Tensor Name>

  -p          - Enable kernel time profiling
  -v <string> - <Specify Expected Output Tensor Names (Ex: -v expected_tensor1.pt,expected_tensor2.pt or skip)>
 ```


Note: `Output Tensor Name`  is essentially a chosen tensor that needs to be copied back to the CPU and written to disk. Additionally, the name must match the tensor's name (tensor_) and number as specified in the JSON file. Please refer args_data.json file.

### Demo (01-vector-add.py)

`SPIRVRunner` is configured to run the `add_kernel.spv` SPIRV binary with inputs `tensor_0.pt` and `tensor_1.pt` and output `tensor_2.pt`. `add_kernel.spv` was generated from the `01-vector-add.py` tutorial.

SPIRVRunner Usage:
```
cd tests/add_kernel
`<abs path to SPIRVRunner executable> -o tensor_2 -p`

Note: Prior to this run test framework to generate serialized args/tensor information
```

Expected output follows:

```
Running on device: Intel(R) Data Center GPU Max 1100
Read 3772 byte kernel.
Loaded kernel with 0 registers and 0 register spills.
Tensor output: [98432], Float (393728 bytes)
Kernel execution time: 0.0096 ms
Output Tensor Path: /abs/path/utils/SPIRVRunner/cpp_outs.pt
```

The GPU hardware, shape and data type of each Tensor (along with number of bytes), and kernel information are printed. The shape and data type of the output Tensor is currently printed, along with the the first cell in the output. Ensuring the value of the first cell is non-zero allows for a quick sanity check. The output Tensor is written to a file `cpp_outs.pt` which is a Tensor in PyTorch format. Typically, we will create a quick Python script to read the input Tensor, run the same computations in PyTorch, and then compare the PyTorch result with the loaded `cpp_outs.pt` Tensor using the PyTorch testing API.

### Test Framework

In order to use the `SPIRVRunner` test framework set following environment varibles

```
   export SPIRV_RUNNER_PATH=<abs path to SPIRV Runner executable>
   export SPIRV_RUNNER_TESTS=<abs path to SPIRV Runner tests>
```

Run following command to execute,

```
    python3 -m pytest tests/test_spirv_runner.py
```

Expected output as follows:

```
(triton) intel-xpu-backend-for-triton/utils/SPIRVRunner$ python3 -m pytest tests/test_spirv_runner.py
============================================================================ test session starts =============================================================================
platform linux -- Python 3.9.18, pytest-8.3.4, pluggy-1.5.0
rootdir: /data/kballeda/Kali/0122_ci_enable/intel-xpu-backend-for-triton
configfile: pyproject.toml
plugins: xdist-3.6.1, forked-1.6.0
collected 4 items

tests/test_spirv_runner.py Test: utils/SPIRVRunner/tests/test_spirv_runner.py::test_argument_parsing, Status: PASS
Progress: 1/1 tests passed (100.00%)
.Test: utils/SPIRVRunner/tests/test_spirv_runner.py::test_invalid_argument, Status: PASS
Progress: 2/2 tests passed (100.00%)
.Test: utils/SPIRVRunner/tests/test_spirv_runner.py::test_spirv_execution[intel-xpu-backend-for-triton/utils/SPIRVRunner/tests/add_kernel], Status: PASS
Progress: 3/3 tests passed (100.00%)
.Test: utils/SPIRVRunner/tests/test_spirv_runner.py::test_spirv_execution[/intel-xpu-backend-for-triton/utils/SPIRVRunner/tests/dot], Status: PASS
Progress: 4/4 tests passed (100.00%)
```
