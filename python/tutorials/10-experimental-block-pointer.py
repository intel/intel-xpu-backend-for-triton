"""
Block Pointer (Experimental)
============================
This tutorial will guide you through writing a matrix multiplication algorithm that utilizes block pointer semantics.
These semantics are more friendly for Triton to optimize and can result in better performance on specific hardware.
Note that this feature is still experimental and may change in the future.

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
# To address this issue, we will use block pointers :code:`pointer_type<block<element_type>>` and load them into
# :code:`block<element_type>`, in which way gives better friendliness for the compiler to optimize memory access
# patterns.
#
# Let's start with the previous matrix multiplication example and demonstrate how to rewrite it to utilize block pointer
# semantics.

# %%
# Make a Block Pointer
# --------------------
# A block pointer pointers to a block in a parent tensor and is constructed by :code:`make_block_ptr` function,
# which takes the following information as arguments:
#
# * :code:`base`: the base pointer to the parent tensor;
#
# * :code:`shape`: the shape of the parent tensor;
#
# * :code:`strides`: the strides of the parent tensor, which means how much to increase the pointer by when moving by 1 element in a specific axis;
#
# * :code:`offsets`: the offsets of the block;
#
# * :code:`block_shape`: the shape of the block;
#
# * :code:`order`: the order of the block, which means how the block is laid out in memory.
#
# For example, to a block pointer to a :code:`BLOCK_SIZE_M * BLOCK_SIZE_K` block in a row-major 2D matrix A by
# offsets :code:`(pid_m * BLOCK_SIZE_M, 0)` and strides :code:`(stride_am, stride_ak)`, we can use the following code
# (exactly the same as the previous matrix multiplication tutorial):
#
# .. code-block:: python
#
#     a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
#                                     offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
#                                     order=(1, 0))
#
# Note that the :code:`order` argument is set to :code:`(1, 0)`, which means the second axis is the inner dimension in
# terms of storage, and the first axis is the outer dimension. This information may sound redundant, but it is necessary
# for some hardware backends to optimize for better performance.

# %%
# Load/Store a Block Pointer
# --------------------------
# To load/store a block pointer, we can use :code:`load/store` function, which takes a block pointer as an argument,
# de-references it, and loads/stores a block. You may mask some values in the block, here we have an extra argument
# :code:`boundary_check` to specify whether to check the boundary of each axis for the block pointer. With check on,
# out-of-bound values will be masked according to the :code:`padding_option` argument (load only), which can be
# :code:`zero` or :code:`nan`. Temporarily, we do not support other values due to some hardware limitations. In this
# mode of block pointer load/store does not support :code:`mask` or :code:`other` arguments in the legacy mode.
#
# So to load the block pointer of A in the previous section, we can simply write
# :code:`a = tl.load(a_block_ptr, boundary_check=(0, 1))`. Boundary check may cost extra performance, so if you can
# guarantee that the block pointer is always in-bound in some axis, you can turn off the check by not passing the index
# into the :code:`boundary_check` argument. For example, if we know that :code:`M` is a multiple of
# :code:`BLOCK_SIZE_M`, we can replace with :code:`a = tl.load(a_block_ptr, boundary_check=(1, ))`, since axis 0 is
# always in bound.

# %%
# Advance a Block Pointer
# -----------------------
# To advance a block pointer, we can use :code:`advance` function, which takes a block pointer and the increment for
# each axis as arguments and returns a new block pointer with the same shape and strides as the original one,
# but with the offsets advanced by the specified amount.
#
# For example, to advance the block pointer by :code:`BLOCK_SIZE_K` in the second axis
# (no need to multiply with strides), we can write :code:`a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))`.

# %%
# Final Result
# ------------

import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=s, num_warps=32) for s in [1, 2, 3]
    ] + [
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': m},
                      num_stages=s, num_warps=w) for s in [2, 3, 4] for (m, w) in ([('large', 32), ('small', 64)])
    ] + [
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
            num_stages=s, num_warps=32) for s in [2]
    ] + [
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': m},
                      num_stages=s, num_warps=w) for s in [2, 3] for (m, w) in ([('large', 32), ('small', 64)])
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_with_block_pointers(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
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
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    # See above `Make a Block Pointer` section for details.
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_DTYPE)
    for k in range(0, K, BLOCK_SIZE_K):
        # Load with boundary checks, no need to calculate the mask manually.
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        # See above `Load/Store a Block Pointer` section for details.
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b, out_dtype=ACCUMULATOR_DTYPE)
        # Advance the block pointer to the next K block.
        # See above `Advance a Block Pointer` section for details.
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    c = accumulator.to(c_ptr.type.element_ty)
    # ----------------------------------------------------------------
    # Write back the block of the output matrix C with boundary checks.
    # See above `Load/Store a Block Pointer` section for details.
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


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
def matmul_kernel_with_block_pointers_batched(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        B, M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_az, stride_am, stride_ak,  #
        stride_bz, stride_bk, stride_bn,  #
        stride_cz, stride_cm, stride_cn,  #
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
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    # See above `Make a Block Pointer` section for details.
    a_block_ptr = tl.make_block_ptr(base=a_ptr + offset_a, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr + offset_b, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_DTYPE)
    for k in range(0, K, BLOCK_SIZE_K):
        # Load with boundary checks, no need to calculate the mask manually.
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        # See above `Load/Store a Block Pointer` section for details.
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b, out_dtype=ACCUMULATOR_DTYPE)
        # Advance the block pointer to the next K block.
        # See above `Advance a Block Pointer` section for details.
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    c = accumulator.to(c_ptr.type.element_ty)
    # ----------------------------------------------------------------
    # Write back the block of the output matrix C with boundary checks.
    # See above `Load/Store a Block Pointer` section for details.
    offset_c = bid.to(tl.int64) * stride_cz
    c_block_ptr = tl.make_block_ptr(base=c_ptr + offset_c, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


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
        matmul_kernel_with_block_pointers_batched[grid](
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
        matmul_kernel_with_block_pointers[grid](
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
# Still we can test our matrix multiplication with block pointers against a native torch implementation (i.e., cuBLAS).

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
