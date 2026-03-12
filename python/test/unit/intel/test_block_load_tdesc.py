"""Tests for DescriptorLoadOp → 2D block IO lowering.

Mirrors the structure of test_block_load.py but uses ``tt.make_tensor_descriptor``
/ ``tt.descriptor_load`` / ``tt.descriptor_store`` (TTGIR tests) and
``tl.make_tensor_descriptor`` / ``desc.load`` / ``desc.store`` (JIT tests)
instead of ``tt.make_tensor_ptr`` / ``tt.load`` / ``tt.store``.
"""

import pytest
import torch
import pathlib
from functools import partial

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


# ---------------------------------------------------------------------------
# Load/store with DPAS dot_op layout (mirrors test_block_load_dpas_layout)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M, N",
                         [[256, 64], [256, 32], [128, 32], [128, 16], [128, 8], [64, 64], [64, 32], [32, 32], [16, 64]])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.skipif(not is_xpu(), reason="Block descriptor tests are specific to the XPU backend")
@pytest.mark.xfail(not triton.runtime.driver.active.get_current_target().arch['has_subgroup_2d_block_io'],
                   reason="Block loads not supported on this architecture", run=False)
def test_block_tdesc_load_dpas_layout(M, N, dtype_str, device, tmp_path: pathlib.Path):
    """Load A and B matrices through tensor descriptors with DPAS dot_op encoding.

    Unlike test_block_load_dpas_layout, tensor descriptors do not carry per-op
    block_io or order attributes. Memory layout is inferred from strides, and
    boundary protection is always enabled via the descriptor shape.
    """
    if dtype_str == "int8":
        A_width = 2
        B_width = 4
        layouts = "#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 2]}>"
        num_warps = 4
    elif dtype_str == "float32":
        A_width = 1
        B_width = 1
        layouts = "#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2]}>"
        num_warps = 32
    else:
        A_width = 1
        B_width = 2
        layouts = "#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [4, 2]}>"
        num_warps = 32

    ty = {"float32": "f32", "float16": "f16", "int8": "i8"}[dtype_str]

    # Tensor descriptor does not have the `order` attribute or `ttig.block_io`
    # attribute – the layout is inferred from the strides in the descriptor.
    # Row-major A: strides = [N, 1]
    # Row-major B: strides = [M, 1]
    ir = layouts + f"""
    module attributes {{ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32}} {{
        tt.func public @block_load_tdesc_dpas_layout(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg3: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) attributes {{noinline = false}} {{
            %0 = tt.get_program_id x : i32
            %M_i64 = arith.constant {M} : i64
            %N_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32
            %cM_i32 = arith.constant {M} : i32
            %cN_i32 = arith.constant {N} : i32

            // A matrix: row-major MxN, encoding = dot_op opIdx=0
            %src_a = tt.make_tensor_descriptor %arg0, [%cM_i32, %cN_i32], [%N_i64, %c1_i64]
                     : !tt.ptr<{ty}>, !tt.tensordesc<tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>

            %a = tt.descriptor_load %src_a [%0, %c0_i32]
                 : !tt.tensordesc<tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>
                 -> tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>

            %dst_a = tt.make_tensor_descriptor %arg1, [%cM_i32, %cN_i32], [%N_i64, %c1_i64]
                     : !tt.ptr<{ty}>, !tt.tensordesc<tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>

            tt.descriptor_store %dst_a [%0, %c0_i32], %a
                                : !tt.tensordesc<tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>,
                                  tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>

            // B matrix: row-major NxM, encoding = dot_op opIdx=1
            %src_b = tt.make_tensor_descriptor %arg2, [%cN_i32, %cM_i32], [%M_i64, %c1_i64]
                     : !tt.ptr<{ty}>, !tt.tensordesc<tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>

            %b = tt.descriptor_load %src_b [%c0_i32, %0]
                 : !tt.tensordesc<tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>
                 -> tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>

            %dst_b = tt.make_tensor_descriptor %arg3, [%cN_i32, %cM_i32], [%M_i64, %c1_i64]
                     : !tt.ptr<{ty}>, !tt.tensordesc<tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>

            tt.descriptor_store %dst_b [%c0_i32, %0], %b
                                : !tt.tensordesc<tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>,
                                  tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>

            tt.return
        }}
    }}
    """

    torch_dtype = getattr(torch, dtype_str)
    if torch_dtype.is_floating_point:
        a = torch.randn((M, N), dtype=torch_dtype, device=device)
        b = torch.randn((N, M), dtype=torch_dtype, device=device)
    else:
        a = torch.randint(low=-127, high=128, size=(M, N), dtype=torch_dtype, device=device)
        b = torch.randint(low=-127, high=128, size=(N, M), dtype=torch_dtype, device=device)

    x = torch.empty_like(a)
    y = torch.empty_like(b)

    temp_file = tmp_path / "test_block_tdesc_dpas_layout.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x, b, y)
    assert torch.equal(a, x) and torch.equal(b, y)


# ---------------------------------------------------------------------------
# Matmul via tensor descriptors (mirrors test_block_load_dot_product)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K",
                         [[256, 256, 32], [256, 64, 32], [64, 256, 32], [64, 128, 32], [64, 64, 32], [32, 32, 32],
                          [32, 32, 16], [16, 16, 16], [8, 32, 16], [8, 512, 64]])
@pytest.mark.parametrize("GROUP_SIZE_M", [4, 1])
@pytest.mark.parametrize("TRANSPOSE_A", [True, False])
@pytest.mark.parametrize("TRANSPOSE_B", [True, False])
@pytest.mark.skipif(not is_xpu(), reason="Block descriptor tests are specific to the XPU backend")
@pytest.mark.xfail(
    not (triton.runtime.driver.active.get_current_target().arch['has_subgroup_2d_block_io']
         and triton.runtime.driver.active.get_current_target().arch['has_subgroup_matrix_multiply_accumulate']),
    reason="Block loads and/or DPAS not supported on this architecture", run=False)
def test_block_tdesc_dot_product(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, TRANSPOSE_A, TRANSPOSE_B,
                                 device):
    if GROUP_SIZE_M == 1 and (BLOCK_SIZE_M > 64 or BLOCK_SIZE_N > 64):
        pytest.xfail("Skipping slow combinations")

    @triton.jit
    def matmul_kernel_with_tensor_descriptors(
            # Pointers to matrices
            a_ptr, b_ptr, c_ptr,
            # Matrix dimensions
            M, N, K,
            # Raw strides of A (dim0, dim1 of the storage layout)
            stride_a0, stride_a1,
            # Raw strides of B (dim0, dim1 of the storage layout)
            stride_b0, stride_b1,
            # Strides of C (always row-major)
            stride_cm, stride_cn,
            # Meta-parameters
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
            GROUP_SIZE_M: tl.constexpr, TRANSPOSE_A: tl.constexpr, TRANSPOSE_B: tl.constexpr):
        """Kernel for computing the matmul C = A x B using tensor descriptors.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N).
        When TRANSPOSE_A is True, A is stored as (K, M) in memory.
        When TRANSPOSE_B is True, B is stored as (N, K) in memory.
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create tensor descriptors for A and B.
        # Tensor descriptors require the last stride to be 1 (contiguous),
        # so for transposed matrices we describe the storage layout directly
        # and transpose the loaded block.
        if TRANSPOSE_A:
            # A stored as (K, M) in memory
            a_desc = tl.make_tensor_descriptor(a_ptr, shape=[K, M], strides=[stride_a0, 1],
                                               block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_M])
        else:
            # A stored as (M, K) in memory
            a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[stride_a0, 1],
                                               block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])

        if TRANSPOSE_B:
            # B stored as (N, K) in memory
            b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[stride_b0, 1],
                                               block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
        else:
            # B stored as (K, N) in memory
            b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[stride_b0, 1],
                                               block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            if TRANSPOSE_A:
                a = tl.trans(a_desc.load([k, pid_m * BLOCK_SIZE_M]))
            else:
                a = a_desc.load([pid_m * BLOCK_SIZE_M, k])

            if TRANSPOSE_B:
                b = tl.trans(b_desc.load([pid_n * BLOCK_SIZE_N, k]))
            else:
                b = b_desc.load([k, pid_n * BLOCK_SIZE_N])

            accumulator += tl.dot(a, b)
        c = accumulator.to(tl.float32)

        c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[stride_cm, 1],
                                           block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
        c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c.to(tl.float16))

    def triton_mm(X, Y, b=None, transpose_x=False, transpose_y=False):
        if transpose_x:
            K, M = X.shape
        else:
            M, K = X.shape
        if transpose_y:
            N, _ = Y.shape
        else:
            _, N = Y.shape

        Z = torch.empty((M, N), device=X.device, dtype=X.dtype)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

        matmul_kernel_with_tensor_descriptors[grid](X, Y, Z, M, N, K,
                                                    X.stride(0), X.stride(1), Y.stride(0), Y.stride(1), Z.stride(0),
                                                    Z.stride(1), BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                                    BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M,
                                                    TRANSPOSE_A=transpose_x, TRANSPOSE_B=transpose_y)

        return Z

    M = 512
    K = 64
    N = 512
    dtype = torch.float16
    torch.manual_seed(0)

    X = torch.randn((M, K) if not TRANSPOSE_A else (K, M), device=device, dtype=dtype, requires_grad=False)
    Y = torch.randn((K, N) if not TRANSPOSE_B else (N, K), device=device, dtype=dtype, requires_grad=False)

    fn_tor = partial(torch.mm, X if not TRANSPOSE_A else X.T, Y if not TRANSPOSE_B else Y.T)
    fn_tri = partial(triton_mm, X, Y, transpose_x=TRANSPOSE_A, transpose_y=TRANSPOSE_B)

    result_tor = fn_tor()
    result_tri = fn_tri()
    torch.testing.assert_close(result_tri, result_tor, atol=1e-2, rtol=1e-3)
