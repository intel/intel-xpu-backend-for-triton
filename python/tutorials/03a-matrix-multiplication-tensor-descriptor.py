"""
Tensor Descriptors
==================
This tutorial will guide you through writing a matrix multiplication algorithm that utilizes tensor descriptor
semantics. Tensor descriptors provide hardware-aware, bounds-checked block loads and stores that give the compiler
direct knowledge of the memory access pattern, enabling it to emit optimal hardware instructions.

"""

# %%
# Motivations
# -----------
# In the previous matrix multiplication tutorial, we constructed blocks of values by de-referencing blocks of pointers,
# i.e., :code:`load(block<pointer_type<element_type>>) -> block<element_type>`, which involved loading blocks of
# elements from memory. This approach allowed for flexibility in using hardware-managed cache and implementing complex
# data structures, such as tensors of trees or unstructured look-up tables.
#
# However, the drawback of this approach is that it relies heavily on complex optimization passes by the compiler to
# optimize memory access patterns. This can result in brittle code that may suffer from performance degradation when the
# optimizer fails to perform adequately. Additionally, as memory controllers specialize to accommodate dense spatial
# data structures commonly used in machine learning workloads, this problem is likely to worsen.
#
# *Tensor descriptors* offer a higher-level alternative: you describe the memory region once — its shape, strides, and
# tile size — and then simply request tiles by their logical offset. The compiler has direct visibility into the access
# pattern, which allows it to emit optimally scheduled hardware instructions with no manual masking needed.
#
# Let's demonstrate how to write the matrix multiplication kernel using tensor descriptors.

# %%
# Make a Tensor Descriptor
# ------------------------
# A tensor descriptor is constructed with :code:`tl.make_tensor_descriptor` and takes the following arguments:
#
# * :code:`base`: the base pointer to the start of the memory region;
#
# * :code:`shape`: the shape of the tensor — used for automatic out-of-bounds masking;
#
# * :code:`strides`: the strides of the tensor in number of elements per dimension;
#
# * :code:`block_shape`: the shape of the tile to load or store at each access.
#
# For example, to create a tensor descriptor for a :code:`BLOCK_SIZE_M x BLOCK_SIZE_K` tile in a row-major
# 2D matrix A with strides :code:`(stride_am, stride_ak)`:
#
# .. code-block:: python
#
#     a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
#                                        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
#
# The memory layout is fully described by :code:`strides`, and the tile offset is supplied at load/store time.

# %%
# Load/Store with a Tensor Descriptor
# ------------------------------------
# To load a tile, call :code:`.load()` on the descriptor and pass the offset of the top-left corner:
#
# .. code-block:: python
#
#     a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
#
# Out-of-bounds accesses are automatically masked to zero using the :code:`shape` provided at construction time —
# no explicit mask argument is required.
#
# To store a result tile back to memory, call :code:`.store()`:
#
# .. code-block:: python
#
#     c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)
#
# Boundary masking on store is also handled automatically.

# %%
# Iterating over K
# ----------------
# The offset of the tile to load is passed directly to :code:`.load()` at each iteration. A simple integer
# counter :code:`off_k` tracks the current position along the K dimension:
#
# .. code-block:: python
#
#     off_k = 0
#     for _ in range(0, K, BLOCK_SIZE_K):
#         a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
#         b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
#         accumulator += tl.dot(a, b)
#         off_k += BLOCK_SIZE_K

# %%
# Final Result
# ------------

import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256'},
            num_stages=s, num_warps=32) for s in [1, 2, 3]
    ] + [
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': m},
                      num_stages=s, num_warps=w) for s in [2, 3, 4] for (m, w) in ([('256', 32), ('128', 64)])
    ] + [
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256'},
            num_stages=s, num_warps=32) for s in [2]
    ] + [
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': m},
                      num_stages=s, num_warps=w) for s in [2, 3] for (m, w) in ([('256', 32), ('128', 64)])
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_with_tensor_descriptors(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak: tl.constexpr,  #
        stride_bk, stride_bn: tl.constexpr,  #
        stride_cm, stride_cn: tl.constexpr,  #
        ACCUMULATOR_DTYPE: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See the matrix multiplication tutorial for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create tensor descriptors for A and B.
    # The descriptor encodes the base pointer, tensor shape, strides, and tile shape.
    # The tile offset is supplied later at each descriptor.load([row, col]) call.
    # See above `Make a Tensor Descriptor` section for details.
    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
    b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                       block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_DTYPE)
    off_k = 0
    for _ in range(0, K, BLOCK_SIZE_K):
        # Load the next tile of A and B. Out-of-bounds accesses are masked to zero automatically.
        # See above `Load/Store with a Tensor Descriptor` section for details.
        a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b, out_dtype=ACCUMULATOR_DTYPE)
        # Advance the K offset for the next iteration.
        # See above `Iterating over K` section for details.
        off_k += BLOCK_SIZE_K
    c = accumulator.to(c_ptr.type.element_ty)
    # ----------------------------------------------------------------
    # Write back the block of the output matrix C.
    # Tensor descriptor store handles boundary masking automatically.
    # See above `Load/Store with a Tensor Descriptor` section for details.
    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2,
                      num_warps=32),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=3,
                      num_warps=32),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2,
                      num_warps=32),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2,
                      num_warps=32),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_with_tensor_descriptors_batched(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        B, M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_az, stride_am, stride_ak: tl.constexpr,  #
        stride_bz, stride_bk, stride_bn: tl.constexpr,  #
        stride_cz, stride_cm, stride_cn: tl.constexpr,  #
        ACCUMULATOR_DTYPE: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Kernel for computing the batched matmul C = A x B.
    A has shape (B, M, K), B has shape (B, K, N) and C has shape (B, M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See the matrix multiplication tutorial for details.
    bid = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offset_a = bid.to(tl.int64) * stride_az
    offset_b = bid.to(tl.int64) * stride_bz
    # ----------------------------------------------------------
    # Create tensor descriptors for this batch slice of A and B.
    # The batch offset is folded into the base pointer so that each
    # descriptor covers a single [M, K] / [K, N] slice.
    # See above `Make a Tensor Descriptor` section for details.
    a_desc = tl.make_tensor_descriptor(base=a_ptr + offset_a, shape=(M, K), strides=(stride_am, stride_ak),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
    b_desc = tl.make_tensor_descriptor(base=b_ptr + offset_b, shape=(K, N), strides=(stride_bk, stride_bn),
                                       block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_DTYPE)
    off_k = 0
    for _ in range(0, K, BLOCK_SIZE_K):
        # Load the next tile of A and B. Out-of-bounds accesses are masked to zero automatically.
        # See above `Load/Store with a Tensor Descriptor` section for details.
        a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b, out_dtype=ACCUMULATOR_DTYPE)
        # Advance the K offset for the next iteration.
        # See above `Iterating over K` section for details.
        off_k += BLOCK_SIZE_K
    c = accumulator.to(c_ptr.type.element_ty)
    # ----------------------------------------------------------------
    # Write back the block of the output matrix C.
    # Tensor descriptor store handles boundary masking automatically.
    # See above `Load/Store with a Tensor Descriptor` section for details.
    offset_c = bid.to(tl.int64) * stride_cz
    c_desc = tl.make_tensor_descriptor(base=c_ptr + offset_c, shape=(M, N), strides=(stride_cm, stride_cn),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)


# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.
def matmul(a, b, accum_dtype, res_dtype):
    # Check constraints.
    if len(a.shape) == 3 and len(b.shape) == 3:
        assert a.shape[0] == b.shape[0], "Incompatible Batch dimension"
        assert a.shape[2] == b.shape[1], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert b.is_contiguous(), "Matrix B must be contiguous"
        B, M, K = a.shape
        B, K, N = b.shape
        c = torch.empty((B, M, N), device=a.device, dtype=res_dtype)
        # Map accumulator type, e.g. `torch.float16` -> `tl.fp16`
        triton_accum_dtype = tl.dtype(str(accum_dtype)[6:].replace('bfloat', 'bf').replace('float', 'fp'))
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            B,
        )
        matmul_kernel_with_tensor_descriptors_batched[grid](
            a, b, c,  #
            B, M, N, K,  #
            a.stride(0), a.stride(1), a.stride(2),  #
            b.stride(0), b.stride(1), b.stride(2),  #
            c.stride(0), c.stride(1), c.stride(2),  #
            ACCUMULATOR_DTYPE=triton_accum_dtype)
    elif len(a.shape) == 2 and len(b.shape) == 2:
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert b.is_contiguous(), "Matrix B must be contiguous"
        M, K = a.shape
        K, N = b.shape
        B = 1
        c = torch.empty((M, N), device=a.device, dtype=res_dtype)
        # Map accumulator type, e.g. `torch.float16` -> `tl.fp16`
        triton_accum_dtype = tl.dtype(str(accum_dtype)[6:].replace('bfloat', 'bf').replace('float', 'fp'))
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        matmul_kernel_with_tensor_descriptors[grid](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            ACCUMULATOR_DTYPE=triton_accum_dtype)
    else:
        assert False, "Input matrixs dimensions mismatch"
    # Allocates output.
    return c


# %%
# Unit Test
# ---------
#
# We can test our matrix multiplication with tensor descriptors against a native torch implementation.

FP16_TYPES = [(torch.float16, torch.float16, torch.float16), (torch.float16, torch.float32, torch.float16),
              (torch.float16, torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16, torch.bfloat16),
              (torch.bfloat16, torch.float32, torch.float32), (torch.bfloat16, torch.float32, torch.bfloat16)]

FP32_TYPES = [(torch.float32, torch.float32, torch.float32)]

INT8_TYPES = [(torch.int8, torch.int32, torch.int32)]

FP8_TYPES = [(torch.float8_e4m3fn, torch.float32, torch.float16)]

torch.manual_seed(0)
for dtype, accum_dtype, res_dtype in FP16_TYPES + FP32_TYPES + INT8_TYPES + FP8_TYPES:
    for shape in [(512, 512), (4, 512, 512)]:
        assert shape[-1] == shape[-2], "Only square matrices are supported"
        if dtype.is_floating_point:
            if accum_dtype in [torch.float16, torch.bfloat16]:
                # 16-bit accumulation across 512 multiplications is error-prone,
                # hence we multiply a matrix of random numbers with
                # [[ 1  1  0 ... ],
                #  [ 1  1  1 ... ],
                #  [ 0  1  1 ... ], ... ]
                # in order only add 3 values per result matrix element.
                a = torch.randn(shape, device='xpu', dtype=dtype)
                b = torch.eye(shape[-2], device='xpu', dtype=dtype) + torch.diag(
                    torch.ones(shape[-2] - 1, device='xpu', dtype=dtype), diagonal=1) + torch.diag(
                        torch.ones(shape[-2] - 1, device='xpu', dtype=dtype), diagonal=-1)
                # duplicate b on batch dimension.
                if len(shape) == 3:
                    b = b.unsqueeze(0).repeat(shape[0], 1, 1)
            elif dtype is torch.float8_e4m3fn:
                a = torch.randn(shape, device='xpu', dtype=torch.float16).to(torch.float8_e4m3fn)
                b = torch.randn(shape, device='xpu', dtype=torch.float16).to(torch.float8_e4m3fn)
            else:
                a = torch.randn(shape, device='xpu', dtype=dtype)
                b = torch.randn(shape, device='xpu', dtype=dtype)

            if dtype is torch.float8_e4m3fn:
                torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16)).to(dtype=res_dtype)
            else:
                torch_output = torch.matmul(a, b).to(dtype=res_dtype)
        else:
            a = torch.randint(low=-127, high=128, size=shape, device='xpu', dtype=dtype)
            b = torch.randint(low=-127, high=128, size=shape, device='xpu', dtype=dtype)
            # torch.matmul clamps values to input dtype; IPEX doesn't support int32 matmul
            torch_output = torch.matmul(a.to(device='cpu', dtype=accum_dtype),
                                        b.to(device='cpu', dtype=accum_dtype)).to(device='xpu', dtype=res_dtype)

        triton_output = matmul(a, b, accum_dtype, res_dtype)

        print(f"triton_output={triton_output}")
        print(f"torch_output={torch_output}")

        # Note: the torch.matmul and Triton implementations uses different
        # algorithms so we need to adjust tolerance.
        rtol = 1e-2 if dtype == torch.bfloat16 or accum_dtype in [torch.float16, torch.bfloat16] else 1e-3
        # FIXME: Remove 1e-1 tolerance for fp32, once fp32 math mode is implemented at pytorch:
        # https://github.com/intel/intel-xpu-backend-for-triton/issues/1957
        atol = 1e-1 if dtype == torch.float32 else 1e-2 if accum_dtype == torch.bfloat16 else 1e-3 if accum_dtype == torch.float16 else 1e-4
        if torch.allclose(triton_output, torch_output, atol=atol, rtol=rtol):
            print("✅ Triton and Torch match")
        else:
            exit("❌ Triton and Torch differ")

# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Compare the performance of our kernel against that of the library
ref_lib = 'oneDNN'
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=[ref_lib.lower(), "triton"],  # Label name for the lines
        line_names=[ref_lib, "Triton"],  # Line styles
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-fp16",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='xpu', dtype=torch.float16)
    b = torch.randn((K, N), device='xpu', dtype=torch.float16)

    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, torch.float32, torch.float16),
                                                     quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)
