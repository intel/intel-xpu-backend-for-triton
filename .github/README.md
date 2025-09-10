[![Build and test](https://github.com/intel/intel-xpu-backend-for-triton/actions/workflows/build-test.yml/badge.svg?branch=main)](https://github.com/intel/intel-xpu-backend-for-triton/actions/workflows/build-test.yml)
[![Triton wheels](https://github.com/intel/intel-xpu-backend-for-triton/actions/workflows/nightly-wheels.yml/badge.svg?branch=main)](https://github.com/intel/intel-xpu-backend-for-triton/actions/workflows/nightly-wheels.yml)

# Intel® XPU Backend for Triton\*

This is the development repository of Intel® XPU Backend for Triton\*, a new [Triton](https://github.com/triton-lang/triton) backend for Intel GPUs.
Intel® XPU Backend for Triton\* is a out of tree backend module for [Triton](https://github.com/triton-lang/triton) used to provide best-in-class performance and productivity on any Intel GPUs for [PyTorch](https://github.com/pytorch/pytorch) and standalone usage.

# Compatibility

* Operating systems:
  * [Ubuntu 22.04](http://releases.ubuntu.com/22.04)
* GPU Cards:
  * [Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html)
  * [Intel® Data Center Flex Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/flex-series.html)
  * [Intel® Arc A770](https://www.intel.com/content/www/us/en/products/sku/229151/intel-arc-a770-graphics-16gb/specifications.html)
  * [Intel® Arc B580](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html)
* GPU Drivers:
  * Latest [Long Term Support (LTS) Release](https://dgpu-docs.intel.com/driver/installation.html)
  * Latest [Rolling Release](https://dgpu-docs.intel.com/driver/installation-rolling.html)
* Toolchain:
  * [Intel® Deep Learning Essentials 2025.2.1](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=dl-essentials&dl-lin=offline&dl-essentials-os=linux)

Note that Intel® XPU Backend for Triton\* is not compatible with Intel® Extension for PyTorch\* and Intel® oneAPI Base Toolkit\*.

See also: [experimental support for Windows](WINDOWS.md).

# Triton is included in the PyTorch

If you have PyTorch on XPU installed from [binaries](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html#binaries), you already have Triton installed and don't need any additional installations, unless you want to use the latest version of Triton from `main`.

You can check existing installation by running one of the [tutorials](/python/tutorials/01-vector-add.py).

# Quick Installation

## Prerequisites

1. Latest [Rolling Release](https://dgpu-docs.intel.com/driver/installation-rolling.html) or [Long Term Support Release](https://dgpu-docs.intel.com/driver/installation.html) of GPU driver
2. [Intel® Deep Learning Essentials 2025.2.1](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=dl-essentials&dl-lin=offline&dl-essentials-os=linux)

## Install PyTorch and Triton from nightly wheels

Currently, Intel® XPU Backend for Triton\* requires a special version of PyTorch and both can be installed from nightly wheels.
Navigate to the [nightly wheels workflow](https://github.com/intel/intel-xpu-backend-for-triton/actions/workflows/nightly-wheels.yml),
select the most recent successful run on the top of the page and download an artifact for the corresponding Python version.
Extract the archive and in the extracted directory execute:

```shell
pip install torch-*.whl triton-*.whl
```

Before using Intel® XPU Backend for Triton\* you need to initialize the toolchain.
The default location is `/opt/intel/oneapi` (if installed as a `root` user) or `~/intel/oneapi` (if installed as a regular user).

```shell
# replace /opt/intel/oneapi with the actual location of Intel® Deep Learning Essentials
source /opt/intel/oneapi/setvars.sh
```

# Install from source

## Prerequisites

1. Latest [Rolling Release](https://dgpu-docs.intel.com/driver/installation-rolling.html) or [Long Term Support Release](https://dgpu-docs.intel.com/driver/installation.html) of GPU driver
2. Latest release of [Intel® Deep Learning Essentials](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=dl-essentials&dl-lin=offline&dl-essentials-os=linux)

## Compile PyTorch and Triton from source

Currently, Intel® XPU Backend for Triton\* requires a special version of PyTorch and both need to be compiled at the same time.

Before compiling PyTorch and Intel® XPU Backend for Triton\* you need to initialize the toolchain.
The default location is `/opt/intel/oneapi` (if installed as a `root` user) or `~/intel/oneapi` (if installed as a regular user).

```shell
# replace /opt/intel/oneapi with the actual location of Intel® Deep Learning Essentials
source /opt/intel/oneapi/setvars.sh
```

Clone this repository:

```shell
git clone https://github.com/intel/intel-xpu-backend-for-triton.git
cd intel-xpu-backend-for-triton
```

To avoid potential conflicts with installed packages it is recommended to create and activate a new Python virtual environment:

```shell
python -m venv .venv --prompt triton
source .venv/bin/activate
```

Compile and install PyTorch:

```shell
scripts/install-pytorch.sh --source
```

Compile and install Intel® XPU Backend for Triton\*:

```shell
scripts/compile-triton.sh
```

# Building with a custom LLVM

Triton uses LLVM to generate code for GPUs and CPUs.  Normally, the Triton build
downloads a prebuilt LLVM, but you can also build LLVM from source and use that.

LLVM does not have a stable API, so the Triton build will not work at an
arbitrary LLVM version.

1. Find the version of LLVM that Triton builds against.
Check `cmake/llvm-hash.txt` to see the current version.

2. Checkout LLVM at this revision to the directory `llvm`,
which must be in the same directory as `intel-xpu-backend-for-triton`:

3. In the directory `intel-xpu-backend-for-triton`, build Triton with custom LLVM:

    ```shell
    ./scripts/compile-triton.sh --llvm --triton
    ```

# Tips for building

- Set `TRITON_BUILD_WITH_CLANG_LLD=true` as an environment variable to use clang
  and lld.  lld in particular results in faster builds.

- Set `TRITON_BUILD_WITH_CCACHE=true` to build with ccache.

- Set `TRITON_HOME=/some/path` to change the location of the `.triton`
  directory where Triton's cache is located and downloads are stored
  during the build. By default, this is the user's home directory. It
  can be changed anytime.

- Pass `--no-build-isolation` to `pip install` to make nop builds faster.
  Without this, every invocation of `pip install` uses a different symlink to
  cmake, and this forces ninja to rebuild most of the `.a` files.

- VSCcode IntelliSense has some difficulty figuring out how to build Triton's C++
  (probably because, in our build, users don't invoke cmake directly, but
  instead use setup.py).  Teach vscode how to compile Triton as follows.

    - Do a local build. Run command `pip install -e .`
    - Get the full path to the `compile_commands.json` file produced by the build:
      `find build -name 'compile_commands.json' | xargs readlink -f`.
      You might get a full path similar to `/Users/{username}/triton/build/cmake.macosx-11.1-arm64-cpython-3.12/compile_commands.json`
    - In vscode, install the
      [C/C++
      extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools),
      then open the command palette (`Shift + Command + P` on Mac, or `Shift +
      Ctrl + P` on Windows/Linux) and open `C/C++: Edit Configurations (UI)`.
    - Open "Advanced Settings" and paste the full path to
      `compile_commands.json` into the "Compile Commands" textbox.

# Running tests

There currently isn't a turnkey way to run all the Triton tests, but you can
follow the following recipe.

```shell
scripts/test-triton.sh
```

# Tips for hacking

For detailed instructions on how to debug Triton's frontend, please refer to this [tutorial](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html). The following includes additional tips for hacking on Triton's backend.

**Helpful environment variables**

- `MLIR_ENABLE_DUMP=1` dumps the IR before every MLIR pass Triton runs, for all
   kernels. Use `MLIR_ENABLE_DUMP=kernelName` to dump for a specific kernel only.
  - Triton cache can interfere with the dump. In cases where `MLIR_ENABLE_DUMP=1` does not work, try cleaning your triton cache: `rm -r ~/.triton/cache/*`
- `LLVM_IR_ENABLE_DUMP=1` dumps the IR before every pass run over the LLVM IR.
- `TRITON_INTERPRET=1` uses the Triton interpreter instead of running on the
  GPU.  You can insert Python breakpoints in your kernel code!
- `TRITON_ENABLE_LLVM_DEBUG=1` passes `-debug` to LLVM, printing a lot of
  debugging information to stdout.  If this is too noisy, run with just
  `TRITON_LLVM_DEBUG_ONLY` instead to limit the output.

  An alternative way to reduce output noisiness is running with
  `LLVM_IR_ENABLE_DUMP=1`, extract the IR before the LLVM pass of interest, and
  then run LLVM's `opt` standalone, perhaps passing `-debug-only=foo` on the
  command line.
- `TRITON_LLVM_DEBUG_ONLY=<comma-separated>` is the equivalent of LLVM's
  `-debug-only` command-line option. This limits the LLVM debug output to
  specific pass or component names (which are specified using `#define
  DEBUG_TYPE` throughout LLVM and Triton) in order to allow the debug output to
  be less noisy. `TRITON_LLVM_DEBUG_ONLY` allows for one or more comma
  separated values to be specified (eg
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions` or
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`).
- `USE_IR_LOC={ttir,ttgir}` reparses the IR such that the location information
  will be the line number of the IR file with that particular extension,
  instead of line number of the python file. This can provide a direct mapping
  from the IR to llir/ptx. When used with performance tools, it can provide a
  breakdown on IR instructions.
- `TRITON_PRINT_AUTOTUNING=1` prints out the best autotuning config and total time
  spent for each kernel after autotuning is complete.
- `DISABLE_LLVM_OPT` will disable llvm optimizations for make_llir and make_ptx
  if its value is true when parsing as Bool. Otherwise, it will be parsed as a list
  of flags to disable llvm optimizations. One usage case is
  `DISABLE_LLVM_OPT="disable-lsr"`
  Loop strength reduction is known to cause up to 10% performance changes for
  certain kernels with register pressure.
- `TRITON_ALWAYS_COMPILE=1` forces to compile kernels regardless of cache hit.
- `MLIR_ENABLE_TIMING` dumps the timing information for each MLIR pass.
- `LLVM_ENABLE_TIMING` dumps the timing information for each LLVM pass.
- `TRITON_DEFAULT_FP_FUSION` overrides the default behavior of allowing fp fusion (mul+add->fma).
- `MLIR_ENABLE_REMARK` enables the performance warnings that are emitted as remarks.

# Usage Guide

## Code Modifications
Intel® XPU Backend for Triton\* requires a special version of PyTorch that can be built from sources or installed from nightly wheels.

1. Add `import torch` for xpu support.
2. Put the tensor and models to XPU by calling `to('xpu')`.

This repository contains modified [tutorials](https://github.com/intel/intel-xpu-backend-for-triton/tree/main/python/tutorials) that must be used with Intel® XPU Backend for Triton\*.

The following examples show modifications for the user code.

### Example 1 : Triton Kernel

This example is a modified version of [Vector Add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#vector-addition) triton kernel. Please refer to [Vector Add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#vector-addition) for detailed comments and illustration about the code semantics.

Comparing to the original code, the following code modifies:

```Python
import torch
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
Triton is transparent for end-to-end models. One could easily use `torch.compile` with `inductor` as backend by default. It will automatically generates triton kernel and gets benefit from it.

```Python
import torch
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

## Performance Analysis Guide

There are several ways of doing performance analysis.
We recommend using `torch.profiler` for end-to-end performance analysis and using Intel® VTune™ Profiler for more detailed kernel analysis.
Note that the user needs to explicitly set `TRITON_XPU_PROFILE=1` when the user needs to enable kernel profiling.

```Bash
export TRITON_XPU_PROFILE=1
```

# Contributing

Community contributions are more than welcome, whether it be to fix bugs or to add new features at [github](https://github.com/intel/intel-xpu-backend-for-triton). For more detailed instructions, please visit our [contributor's guide](https://github.com/intel/intel-xpu-backend-for-triton/blob/main/CONTRIBUTING.md).

## License

_MIT License_. As found in [LICENSE](https://github.com/intel/intel-xpu-backend-for-triton/blob/main/LICENSE) file.


## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](https://github.com/intel/intel-xpu-backend-for-triton/blob/main/SECURITY.md).
