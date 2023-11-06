# Intel® XPU Backend for Triton\*

This is the development repository of Intel® XPU Backend for Triton\*, a new [OpenAI Triton](https://github.com/openai/triton) backend for Intel GPUs. Triton is a language and compiler for writing highly efficient custom deep learning primitives. The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs. Intel® XPU Backend for Triton\* is a module used by Triton to provide a reasonable tradeoff between performance and productivity on Intel GPUs.


- [Intel® XPU Backend for Triton\*](#intel-xpu-backend-for-triton)
- [Setup Guide](#setup-guide)
  - [Prerequisites](#prerequisites)
  - [Option1: Install From whl Packages](#option1-install-from-whl-packages)
  - [Option2: Build From the Source](#option2-build-from-the-source)
- [Usage Guide](#usage-guide)
  - [Code Modifications](#code-modifications)
    - [Example 1 : Triton Kernel](#example-1--triton-kernel)
    - [Example 2 : End-to-End Model](#example-2--end-to-end-model)
  - [More Examples on Tests](#more-examples-on-tests)
  - [Performance Analysis Guide](#performance-analysis-guide)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
  - [License](#license)
  - [Security](#security)

# Setup Guide

Intel® XPU Backend for Triton\* serves as a backend for [OpenAI Triton](https://github.com/openai/triton). There are two Options for installation.

## Prerequisites

Intel® XPU Backend for Triton\* requires the following two dependencies package:
1. [PyTorch](https://pytorch.org/get-started/locally/).
2. [Intel® Extension for PyTorch* ](https://github.com/intel/intel-extension-for-pytorch/).

Please follow [installation guide for Intel® Extension for PyTorch* ](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html#installation-guide) for the detailed process for **BOTH** PyTorch and Intel® Extension for PyTorch*. Please make sure that the associated driver and Intel® oneAPI Base Toolkit are installed correctly.

## Option1: Install From whl Packages
This method is the simplest way of getting things done.

Download the latest `.whl` according to your Python version. We provide `Cython` and `Pypy` version. By default, it should be `CPython`. You could check your Python implementation with the following command:

```Bash
python -c "import platform;print(platform.python_implementation())"
```
Then download the corresponding `.whl` at the [release page](https://github.com/intel/intel-xpu-backend-for-triton/releases) and install it locally, for example:

```Bash
wget https://github.com/intel/intel-xpu-backend-for-triton/releases/download/v2.1.0_rc1/triton-2.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

pip install triton-2.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

## Option2: Build From the Source

```Bash
# Clone OpenAI/Triton
git clone https://github.com/openai/triton.git
cd triton
# Clone submodules
git submodule sync && git submodule update --init --recursive --jobs 0
```
Since we are at the active development stage, it is recommended to check to latest commit for `intel-xpu-backend-for-triton`:

```Bash
cd third_party/intel_xpu_backend
git checkout main && git pull
```

Now Build Triton with Intel XPU backend enabled, note that it is important to make sure that the `triton` repo is checked to the pinned commit. This commit is the latest tested working commit.

```Bash
# cd to triton root folder and checkout to pinned commit
cd ../..
git checkout `cat third_party/intel_xpu_backend/triton_hash.txt`
# Build triton with XPU backend enabled
cd python
TRITON_CODEGEN_INTEL_XPU_BACKEND=1 python setup.py develop
```

We also provide a detailed page for the overall building process. It includes all source building methods. You could refer to [build_from_source.md](build_from_source.md) for more detail.
If you encountered any problem, please refer to the [Possible-Build-Bugs](https://github.com/intel/intel-xpu-backend-for-triton/wiki/Possible-Build-Bugs) page first.


# Usage Guide

## Code Modifications
Intel® XPU Backend for Triton\* only requires minor code changes. The user needs to do the following two things:

1. Add `import intel_extension_for_pytorch` for xpu support.
2. Put the tensor and models to XPU by calling `to('xpu')`. There are cases when PyTorch API needs to be changed, please refer to [API Documentation](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/api_doc.html#gpu-specific) from Intel® Extension for PyTorch* for more detail.

The following examples show modifications for the user code.

### Example 1 : Triton Kernel

This Example is a modified version of [Vector Add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#vector-addition) triton kernel. Please refer to [Vector Add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#vector-addition) for detailed comments and illustration about the code semantics.

Comparing to the original code, the following code modifies:

1. Add `import intel_extension_for_pytorch` for xpu support.
2. Put the tensor to XPU and change the API for manual_seed.


```Python
import torch
# Need to import intel_extension_for_pytorch for xpu support
import intel_extension_for_pytorch

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # Put the tensor to xpu
    output = torch.empty_like(x).xpu()
    assert x.is_xpu and y.is_xpu and output.is_xpu
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output

# For manual_seed, needs to use API for XPU
torch.xpu.manual_seed(0)
size = 512
# For tensors, needs to be put on XPU
x = torch.rand(size, device='xpu')
y = torch.rand(size, device='xpu')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(
    f'The maximum difference between torch and triton is '
    f'{torch.max(torch.abs(output_torch - output_triton))}'
)
```


### Example 2 : End-to-End Model
Triton is transparent for End-to-End models. One could easily use `torch.compile` with `inductor` as backend by default. It will automatically generates triton kernel and gets benefit from it.


```Python
import torch
# Need to import intel_extension_for_pytorch for xpu support
import intel_extension_for_pytorch
from torch._dynamo.testing import rand_strided

from torch.nn import *
class simpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # tensors inside model should be on xpu
        self.y = rand_strided((32, 8), (8, 1), device='xpu:0', dtype=torch.float32)

    def forward(self, x):
        z = x + self.y
        return z

# tensors passed to the model should be on xpu
x = rand_strided((32, 8), (8, 1), device='xpu:0', dtype=torch.float32)
xpu_model = simpleModel()
# Call torch.compile for optimization
optimized_mod = torch.compile(xpu_model)

graph_result = optimized_mod(x)
```

## More Examples on Tests
If you wish to take a look at more examples, please refer to the [Unit Tests](test_docs/unit_tests.md) and [End-to-End Benchmark Tests](test_docs/end_to_end_tests.md).


## Performance Analysis Guide

There are several ways of doing performance analysis. We recommend using `torch.profiler` for End-to-End performance analysis and using Intel® VTune™ Profiler for more detailed kernel analysis. We provide a comprehensive guide for those two:
1. [end_to_end_tests#profiling settings](test_docs/end_to_end_tests.md#profiling-settings) section for using `torch.profiler`.
2. [VTune Profiling Guide](VTune_Profiling.md) for kernel analysis.

Note that the user needs to explicitly set `TRITON_XPU_PROFILE=1` when the user needs to enable kernel profiling.

```Bash
export TRITON_XPU_PROFILE=1
```

# Known Limitations
For known limitations, please refer to the [wiki page for known limitations](https://github.com/intel/intel-xpu-backend-for-triton/wiki/Known-Limitations).

# Contributing
It is a warm welcome for any contributions from the community, please refer to the [contribution guidelines](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

## License

_MIT License_. As found in [LICENSE](https://github.com/intel/intel-xpu-backend-for-triton/blob/main/LICENSE) file.

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](security.md)
