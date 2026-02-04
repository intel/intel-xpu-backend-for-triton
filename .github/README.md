[![Build and test](https://github.com/intel/intel-xpu-backend-for-triton/actions/workflows/build-test.yml/badge.svg?branch=main)](https://github.com/intel/intel-xpu-backend-for-triton/actions/workflows/build-test.yml)
[![Triton wheels](https://github.com/intel/intel-xpu-backend-for-triton/actions/workflows/nightly-wheels.yml/badge.svg?branch=main)](https://github.com/intel/intel-xpu-backend-for-triton/actions/workflows/nightly-wheels.yml)

# Intel® XPU Backend for Triton\*

This is the development repository of Intel® XPU Backend for Triton\*, a new [Triton](https://github.com/triton-lang/triton) backend for Intel GPUs.
Intel® XPU Backend for Triton\* is an out of tree backend module for [Triton](https://github.com/triton-lang/triton) used to provide best-in-class performance and productivity on any Intel GPUs for [PyTorch](https://github.com/pytorch/pytorch) and standalone usage.

# Compatibility

* Operating systems:
  * [Ubuntu 22.04](http://releases.ubuntu.com/22.04)
  * [Ubuntu 24.04](http://releases.ubuntu.com/24.04)
  * [Ubuntu 25.04](http://releases.ubuntu.com/25.04)
* GPU Cards:
  * [Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html)
  * [Intel® Data Center Flex Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/flex-series.html)
  * [Intel® Arc A770](https://www.intel.com/content/www/us/en/products/sku/229151/intel-arc-a770-graphics-16gb/specifications.html)
  * [Intel® Arc B580](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html)
* GPU Drivers:
  * Latest [Long Term Support (LTS) Release](https://dgpu-docs.intel.com/driver/installation-lts2.html)
  * Latest [The Kobuk team Intel® Graphics PPA](https://launchpad.net/~kobuk-team/+archive/ubuntu/intel-graphics)
* Toolchain:
  * [Intel® Deep Learning Essentials 2025.3.1](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=dl-essentials&dl-lin=offline&dl-essentials-os=linux)

Note that Intel® XPU Backend for Triton\* is not compatible with Intel® Extension for PyTorch\* and Intel® oneAPI Base Toolkit\*.

See also: [experimental support for Windows](WINDOWS.md).

# Triton is included in the PyTorch

If you have PyTorch on XPU installed from [binaries](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html#binaries), you already have Triton installed and don't need any additional installations, unless you want to use the latest version of Triton from `main`.

You can check if triton is currently available by running one of the [tutorials](/python/tutorials/01-vector-add.py).

# Improving performance of your Triton source code

Basic rules:

1. **Use Tensor Descriptors:** For inputs and outputs of matmul operations (`tl.dot`), use Tensor Descriptors. This utilizes hardware-optimized DPAS operations and 2D block IO HW operations. You can often expect more than 2x performance improvement compared to the basic tensor of pointers approach. Use device side Tensor Descriptors (defined inside a kernel), not host side (defined in CPU code and passed to the kernel).
2. **Type Annotations:** Use proper type annotations for your kernels. Good type annotations allow for better optimization, but be careful to avoid excessive recompilation.
3. **Benchmark:** Experiment with the performance of your kernel. You can use `triton.testing.do_bench` for basic benchmarking, as demonstrated in the [tutorials](../python/tutorials/02-fused-softmax.py).
4. **Tiling and Autotuning:** Pick appropriate tiling for your machine and tensor shapes.

More details below.

### Use Tensor Descriptors to load tl.dot arguments and save results

For the Intel backend, use Tensor Descriptors to load matrices used in GEMM operations. A Tensor Descriptor can be created inside the kernel and used for loading as follows:

```
a_desc = tl.make_tensor_descriptor(
    # Base of a memory block that we want to work with.
    base=a_ptr,
    # Shape of a tensor that starts from that base, will be used for masking.
    shape=(M, K),
    # Tensor strides
    # It's important that the last stride (stride_ak) is known at compile time,
    # so it must have either `tl.constexpr` type annotation or no annotation at all.
    # We recommend the last dimension to be 1, more details below.
    strides=(stride_am, stride_ak),
    # Block size that will be actually loaded.
    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K)
)
m_offset = 0
# This will load (BLOCK_SIZE_M, BLOCK_SIZE_K) block from memory starting from
# a_ptr + m_offset * stride_am + k_offset * stride_ak
a = a_desc.load([m_offset, k_offset])
m_offset += BLOCK_SIZE_M
```

A Tensor Descriptor describes a piece of memory to load for processing. Loading happens in blocks, and the provided shape is used to mask out-of-bounds values to zero.


Similar code is used for saving results back to global memory:
```
c_desc = tl.make_tensor_descriptor(
    base=c_ptr,
    shape=(M, N),
    strides=(stride_cm, stride_cn),
    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N)
)
c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)
```

You can view a full example of a GEMM kernel with Tensor Descriptors [here](../benchmarks/triton_kernels_benchmark/gemm_benchmark.py).

**Before Tensor Descriptors:**
```
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offs_k = tl.arange(0, BLOCK_SIZE_K)
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
```

**After:**

```
a_desc = tl.make_tensor_descriptor(
    base=a_ptr, shape=(M, K),
    strides=(stride_am, stride_ak),
    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K)
)
b_desc = tl.make_tensor_descriptor(
    base=b_ptr, shape=(K, N),
    strides=(stride_bk, stride_bn),
    block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N)
)
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
off_k = 0
for _ in range(0, K, BLOCK_SIZE_K):
    a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
    b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
    accumulator += tl.dot(a, b)
    off_k += BLOCK_SIZE_K
```

Tensor Descriptors internally perform offset pointer calculations and masking, so manual calculation of these variables is no longer necessary.

---
In the base case, a Tensor Descriptor describes the whole `torch.Tensor` with full shape and strides, but you can also treat it as a "view" of a tensor. You can select a specific slice of a tensor using a mask specific to that block.

For example, consider a 3D tensor `A` where we want to get a slice `A[E, :M, :]` for a Mixture of Experts (MoE) kernel. The original code might look like this (adapted from [vllm](https://github.com/vllm-project/vllm/blob/8005e606bf280b7b6002f57e95ae3210ddc6f041/vllm/model_executor/layers/fused_moe/fused_batched_moe.py#L237)):

```
# Tensor A (a_ptr) has shape [E Experts, M Tokens, K features]
# We want to load K blocks from A[expert_id, cta_m_start:min(cta_m_start+BLOCK_M, e_num_tokens), :]

# We process just one expert in this block
expert_id = tl.program_id(axis=0)
# Defines how many tokens we need to process for this expert in total
e_num_tokens = tl.load(expert_num_tokens_ptr + expert_id)
if e_num_tokens == 0:
    return

cta_m_start = pid_m * BLOCK_M
if cta_m_start >= e_num_tokens:
    return # Early exit

# Start of a current block, A[expert_id, cta_m_start:cta_m_start+BLOCK_M, :]
a_ptr_block = a_ptr + expert_id * stride_ae + cta_m_start * stride_am

offs_m = tl.arange(0, BLOCK_M)
offs_k = tl.arange(0, BLOCK_K)

# Actual block size of M dimension that we need to process
cta_m_size = min(BLOCK_M, e_num_tokens - cta_m_start)

mask_m = offs_m < cta_m_size

# Each expert needs to process e_num_tokens
# Pointers to our block A[expert_id, cta_m_start:cta_m_start+BLOCK_M, 0:BLOCK_K]
a_ptrs = a_ptr_block + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

offs_k = tl.arange(0, BLOCK_K)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(
      a_ptrs,
      # We mask tokens outside of e_num_token range and features larger than K
      mask=mask_m[:, None] & (offs_k[None, :] < K - k * BLOCK_K),
      other=0.0)
    a_ptrs += BLOCK_K * stride_ak
```


We can migrate to Tensor Descriptors to simplify the source code. Let's investigate what Tensor Descriptors we will need here.

From `a_ptrs` shape we can infer block shapes and strides of a possible Tensor Descriptor:
```
block_size=(BLOCK_M, BLOCK_K)
strides=(stride_am, stride_ak)
```

Using Tensor Descriptors we can directly describe a slice that we want to process:

`A[expert_id, :e_num_tokens, :]`

and then only load our block:

 `A[expert_id, cta_m_start:cta_m_start+BLOCK_M, k*BLOCK_K:(k+1)*BLOCK_K]`.

We also want to mask tokens outside of `e_num_tokens` (`mask_m`) and features outside of dimension size `K`.
We can just pass that information as tensor shape and avoid manual masking: `shape=(e_num_tokens, K)`

Base of a tensor descriptor can be inferred from `a_ptrs` as well:

`base=a_ptr + expert_id * stride_ae`

Note that we don't add `cta_m_start * stride_am` to the base, because we can pass that offset directly during loading.

So we can rewrite that code with Tensor Descriptors, which will look much cleaner and will work faster on XPU:

```
expert_id = tl.program_id(axis=0)
e_num_tokens = tl.load(expert_num_tokens + expert_id)
if e_num_tokens == 0:
    # Early exit
    return

cta_m_start = pid_m * BLOCK_M
if cta_m_start >= e_num_tokens:
    # Early exit
    return

a_desc = tl.make_tensor_descriptor(
    base=a_ptr + expert_id * stride_ae,
    shape=(e_num_tokens, K),
    strides=(stride_am, stride_ak),
    block_shape=(BLOCK_M, BLOCK_K))

for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
```

---


Tensor Descriptors support shapes up to 5 dimensions, however the Intel XPU backend currently optimizes well only the 2D case, so avoid using higher dimensionality Tensor Descriptors whenever possible.

Consider this example based on the unified attention kernel from [vllm](https://github.com/vllm-project/vllm/blob/9a161307f5f096c63ae4134c5055d87a36d224a8/vllm/attention/ops/triton_unified_attention.py#L52). This code loads a block of K values from a cache of shape `[NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_SIZE]`:


```
offs_d = tl.arange(0, HEAD_SIZE_PADDED)
dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
block_table_offset = seq_idx * block_table_stride

# iterate through tiles
for j in range(0, num_blocks):
    physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

    offs_n = tl.arange(0, BLOCK_SIZE)

    k_offset = (physical_block_idx * stride_k_cache_0 +
                kv_head_idx * stride_k_cache_2 +
                offs_d[:, None] * stride_k_cache_3 +
                offs_n[None, :] * stride_k_cache_1)

    # K : (HEAD_SIZE, BLOCK_SIZE)
    K_load = tl.load(key_cache_ptr + k_offset,
                        mask=dim_mask[:, None],
                        other=0.0)

```

The code above loads 2D block of `K[physical_block_idx, :BLOCK_SIZE, kv_head_idx, :HEAD_SIZE].T`.

A 3D Tensor Descriptor implementation following the tensor shape might look like this:

```
# iterate through tiles
for j in range(0, num_blocks):
    physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)
    k_base = key_cache_ptr + physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2
    k_desc = tl.make_tensor_descriptor(base=k_base, shape=(BLOCK_SIZE, 1, HEAD_SIZE),
                                       strides=(stride_k_cache_1, stride_k_cache_2, stride_k_cache_3),
                                       block_shape=(BLOCK_SIZE, 1, HEAD_SIZE_PADDED))
    K_load = k_desc.load([0, 0, 0]).reshape(BLOCK_SIZE, HEAD_SIZE_PADDED).T
```

However, describing this memory as a 2D block yields significantly better performance:

```
# iterate through tiles
for j in range(0, num_blocks):
    physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)
    k_base = key_cache_ptr + physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2
    k_desc = tl.make_tensor_descriptor(base=k_base, shape=(BLOCK_SIZE, HEAD_SIZE),
                                       strides=(stride_k_cache_1, stride_k_cache_3),
                                       block_shape=(BLOCK_SIZE, HEAD_SIZE_PADDED))
    K_load = k_desc.load([0, 0]).T
```

---

**Set last stride to 1**

Intel backend for Triton can work with non-contiguous tensors, but to maximize performance we recommend
using Tensor Descriptors on contiguous blocks of memory. So for best performance one of the strides has to be 1.
At the same time to maximize performance you only want to use 2d Tensor Descriptors. Nvidia backend requires the last
stride of a Tensor Descriptor to be 1, same is true for triton interpreter (enabled with `TRITON_INTERPRET=1`).
Combining all these requirements and recommendations, the best practice is to set the last stride of a Tensor Descriptor to 1
for performance and compatibility reasons. If you want the transpose version of the same 2D block, use transposition (`a.T`).

---
**Use kernel side Tensor Descriptors, not host side**

Tensor Descriptors can be defined inside of a kernel (called **device side Tensor Descriptors** (like in all the examples above) or outside of a triton kernel, in the launching utility (called **host side Tensor Descriptors**), like it is done in the upstream Triton ([example](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)).
You should only use device side Tensor Descriptors on XPU for now.

---
**Summary:**
1. **Use Tensor Descriptors to load memory required for `tl.dot` and to save results.**
2. **Use kernel side Tensor Descriptors, not host side.**
3. **Strive to only use 2D Tensor Descriptors with last stride set to 1.**
4. **Ideally, annotate strides with `tl.constexpr`.** Basic preference is `tl.constexpr` > no annotation > `tl.int64`/ `tl.int32` (for non-last strides) >> `tl.int64` / `tl.int32` for the last stride. Avoid annotating the last strides with `tl.int64` or `tl.int32`!

### Use Proper Type Annotations
1. Set `tl.constexpr` type annotation for block sizes and boolean flags to let the compiler optimize. Ideally, also use `tl.constexpr` for strides and tensor shapes that will cover limited number of values (up to 10). Each combination of arguments with this annotation is compiled separately, so avoid setting it for values that vary widely at runtime (like the number of tokens) to prevent excessive recompilation. For GEMM kernels of modern LLMs, for example, number of input and output features will likely be the same or cover just 2-4 unique values, but number of tokens can have more than 100 unique values. So you should use `tl.constexpr` for number of features, but not for the number of tokens.
2. No Annotation: You can keep type annotations empty and let the compiler guess. This is good for parameters that change often, like number of tokens or total number of elements, to avoid excessive recompilation.
3. Avoid annotating with `tl.int64` or `tl.int32`. It can prevent many optimizations.
4. Never annotate the last tensor stride with `tl.int32` or `tl.int64`!

Example of a reasonable type annotation for a GEMM kernel:
```
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr
):
```

It's even better to use more `tl.constexpr` declarations if we know that the kernel will be called for the same input shapes. That is often the case for DL models including LLMs. Usually only the batch size, and the number of tokens varies.

This is especially important for cases when you have `tl.dot` operations without Tensor Descriptors (which is discouraged) as compiler will try to guess if it can perform Tensor Descriptor conversion internally and will need shapes and strides
fixed during compilation.
```
...
        # M will correspond to the number of tokens and will vary during prefill
        # setting it to tl.constexpr would mean recompiling
        # the kernel every time the input size changes
        M,
        # GEMM will transform K features -> N features and these are constant
        N: tl.constexpr,
        K: tl.constexpr,
        # Strides depend on remaining shapes, stride_am = K and also will not change
        stride_am: tl.constexpr, stride_ak: tl.constexpr,
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
```

---

**Avoid int32 Overflow**

If you replace `tl.int64` with `tl.constexpr` you will now get int32 value for your strides (instead of int64) and you might end up with an overflow in your offset calculation. That can happen for MOE deepseek R1 model for example.

Original source code:
```
# stride_ae: tl.int64
# For deepseek R1 stride_ae = 7168 * 2048 * 2 = 2.9e7, which fits in the maximum int32 value 2.1e9

# this will be tl.int32
expert_id = tl.program_id(axis=0)
# this will be calculated as tl.int64
# expert_id goes up to 256, so we get at maximum 7.5e9, which fits int64 type, no error here
expert_offset = expert_id * stride_ae
```

Now, if we use tl.constexpr:

```
# stride_ae: tl.constexpr, will be treated as int32 under the hood, because the initial value fits within int32
# For deepseek R1 stride_ae = 7168 * 2048 * 2 = 2.9e7

# this will be tl.int32
expert_id = tl.program_id(axis=0)
# this will be calculated as tl.int32 now
# expert_id goes up to 256, so we get at maximum 7.5e9, which doesn't fit int32 type,
# there WILL BE AN OVERFLOW.
expert_offset = expert_id * stride_ae
```

For large values, cast manually into `tl.int64`:
```
# This will not overflow even if stride_ae is int32
expert_offset = expert_id.to(tl.int64) * stride_ae
```

### Benchmark and Tune Kernel Configuration

Just like with any other Triton kernels, you want to pick appropriate block sizes to your hardware and tensor shapes. Use `triton.autotune` to try various combinations and find the best one. Key parameters to tune include block sizes, `num_warps`, `num_stages`, and `grf_mode`.

See existing [benchmarks](../benchmarks/triton_kernels_benchmark/) for reasonable configuration grids for [GEMM](../benchmarks/triton_kernels_benchmark/gemm_benchmark.py) and [Flash Attention](../benchmarks/triton_kernels_benchmark/flash_attention_benchmark.py) kernels.

**GRF Mode**:
The Intel-specific option `grf_mode` determines the number of registers allocated to a kernel.
Setting it higher can be good for a kernel that uses many registers, but it will decrease hardware utilization.
You can set high value with `grf_mode = "256"` if you have to.

# Quick Installation

## Prerequisites

1. Latest [Rolling Release](https://dgpu-docs.intel.com/driver/installation-rolling.html) or [Long Term Support Release](https://dgpu-docs.intel.com/driver/installation-lts2.html) of GPU driver
2. [Intel® Deep Learning Essentials 2025.3.1](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=dl-essentials&dl-lin=offline&dl-essentials-os=linux)

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

1. Latest [Rolling Release](https://dgpu-docs.intel.com/driver/installation-rolling.html) or [Long Term Support Release](https://dgpu-docs.intel.com/driver/installation-lts2.html) of GPU driver
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
