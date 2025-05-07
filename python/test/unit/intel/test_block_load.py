import itertools

import numpy as np
import pytest
import torch
import pathlib
from functools import partial

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


class DpasLayout:

    def __init__(self, repeatCount, systolic_depth, execution_size, ops_per_chan, threads_per_warp, warps_per_cta,
                 rep_cluster):
        self.repeatCount = repeatCount
        self.systolic_depth = systolic_depth
        self.execution_size = execution_size
        self.ops_per_chan = ops_per_chan
        self.threads_per_warp = threads_per_warp
        self.warps_per_cta = warps_per_cta
        self.rep_cluster = rep_cluster

    def __str__(self):
        return f"#triton_intel_gpu.dpas<{{repeatCount={self.repeatCount}, systolicDepth={self.systolic_depth}, executionSize = {self.execution_size}, opsPerChan = {self.ops_per_chan}, threadsPerWarp = {self.threads_per_warp}, warpsPerCTA={self.warps_per_cta}, repCluster={self.rep_cluster}}}>"


def warps_per_cta(layout):
    return layout.warps_per_cta


@pytest.mark.parametrize("M, N",
                         [[256, 64], [256, 32], [128, 32], [128, 16], [128, 8], [64, 64], [64, 32], [32, 32], [16, 64]])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
@pytest.mark.xfail(not torch.xpu.get_device_capability()['has_subgroup_2d_block_io'],
                   reason="Block loads not supported on this architecture")
def test_block_load_dpas_layout(M, N, dtype_str, transpose, device, tmp_path: pathlib.Path):
    # modify the layouts to ensure the correct OCL/SPIRV intrinsic is called for each datatype
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

    block_io = "\"column_major\"" if transpose else "\"row_major\""

    ty = {"float32": "f32", "float16": "f16", "int8": "i8"}[dtype_str]

    ir = layouts + f"""
    module attributes {{ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32}} {{
        tt.func public @block_load_dpas_layout(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}, %arg3: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}) attributes {{noinline = false}} {{
            %0 = tt.get_program_id x : i32
            %M_i64 = arith.constant {M} : i64
            %N_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %c0_i32 = arith.constant 0 : i32

            // A matrix
            %1 = tt.make_tensor_ptr %arg0, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {{order = array<i32: 1, 0>}} : <tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>
            %2 = tt.load %1 {{boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"}} : !tt.ptr<tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>
            %3 = tt.make_tensor_ptr %arg1, [%M_i64, %N_i64], [%N_i64, %c1_i64], [%0, %c0_i32] {{order = array<i32: 1, 0>}} : <tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>
            tt.store %3, %2 {{boundaryCheck = array<i32: 0, 1>}} : !tt.ptr<tensor<{M}x{N}x{ty}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>>>

            // B matrix
            %4 = tt.make_tensor_ptr %arg2, [%N_i64, %M_i64], {"[%c1_i64, %N_i64]" if transpose else "[%M_i64, %c1_i64]"}, [%c0_i32, %0] {{order = array<i32: 1, 0>}} : <tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>
            %5 = tt.load %4 {{boundaryCheck = array<i32: 0, 1>, ttig.block_io = {block_io} }} : !tt.ptr<tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>
            %6 = tt.make_tensor_ptr %arg3, [%N_i64, %M_i64], {"[%c1_i64, %N_i64]" if transpose else "[%M_i64, %c1_i64]"}, [%c0_i32, %0] {{order = array<i32: 1, 0>}} : <tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>
            tt.store %6, %5 {{boundaryCheck = array<i32: 0, 1>}} : !tt.ptr<tensor<{N}x{M}x{ty}, #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>>>

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
    y = torch.empty_like(b.T if transpose else b)

    temp_file = tmp_path / "test_block_load_dpas_layout.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x, b, y)
    assert torch.equal(a, x) and torch.equal(b.T if transpose else b, y)


@pytest.mark.parametrize("BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K",
                         [[256, 256, 32], [256, 64, 32], [64, 256, 32], [64, 128, 32], [64, 64, 32], [32, 32, 32],
                          [32, 32, 16], [16, 16, 16], [8, 32, 16], [8, 512, 64]])
@pytest.mark.parametrize("GROUP_SIZE_M", [4, 1])
@pytest.mark.parametrize("TRANSPOSE_A", [True, False])
@pytest.mark.parametrize("TRANSPOSE_B", [True, False])
@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
@pytest.mark.xfail(
    not (torch.xpu.get_device_capability()['has_subgroup_2d_block_io']
         and torch.xpu.get_device_capability()['has_subgroup_matrix_multiply_accumulate']),
    reason="Block loads and/or DPAS not supported on this architecture")
def test_block_load_dot_product(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, TRANSPOSE_A, TRANSPOSE_B,
                                device):
    if GROUP_SIZE_M == 1 and (BLOCK_SIZE_M > 64 or BLOCK_SIZE_N > 64):
        # skip large block sizes as they will be too slow
        pytest.xfail("Skipping slow combinations")

    @triton.jit
    def matmul_kernel_with_block_pointers(
            # Pointers to matrices
            a_ptr, b_ptr, c_ptr,
            # Matrix dimensions
            M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            # The stride variables represent how much to increase the ptr by when moving by 1
            # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
            # by to get the element one row down (A has M rows).
            stride_am: tl.constexpr, stride_ak: tl.constexpr,  #
            stride_bk: tl.constexpr, stride_bn: tl.constexpr,  #
            stride_cm: tl.constexpr, stride_cn: tl.constexpr,  #
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
        pid_m = first_pid_m + (pid % group_size_m)
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
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            # Load with boundary checks, no need to calculate the mask manually.
            # For better performance, you may remove some axis from the boundary
            # check, if you can guarantee that the access is always in-bound in
            # that axis.
            # See above `Load/Store a Block Pointer` section for details.
            a = tl.load(a_block_ptr, boundary_check=(0, 1))
            b = tl.load(b_block_ptr, boundary_check=(0, 1))
            # We accumulate along the K dimension.
            accumulator += tl.dot(a, b)
            # Advance the block pointer to the next K block.
            # See above `Advance a Block Pointer` section for details.
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
        c = accumulator.to(tl.float32)

        c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
        tl.store(c_block_ptr, c.to(tl.float16), boundary_check=(0, 1))

    def triton_mm(X, Y, b=None, transpose_x=False, transpose_y=False):
        if transpose_x:
            K, M = X.shape
            Xstride0, Xstride1 = X.stride(1), X.stride(0)
        else:
            M, K = X.shape
            Xstride0, Xstride1 = X.stride(0), X.stride(1)
        if transpose_y:
            N, _ = Y.shape
            Wstride0, Wstride1 = Y.stride(1), Y.stride(0)
        else:
            _, N = Y.shape
            Wstride0, Wstride1 = Y.stride(0), Y.stride(1)
        # Allocates output.
        Z = torch.empty((M, N), device=X.device, dtype=X.dtype)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

        matmul_kernel_with_block_pointers[grid](X, Y, Z, M, N, K, Xstride0, Xstride1, Wstride0, Wstride1, Z.stride(0),
                                                Z.stride(1), BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                                BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M)

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


layouts = [
    # Layout for Xe2 and Xe2+
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=4, threads_per_warp=16,
               warps_per_cta=[1, 4], rep_cluster=[1, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=16,
               warps_per_cta=[8, 4], rep_cluster=[4, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=16,
               warps_per_cta=[8, 4], rep_cluster=[1, 1]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=4, threads_per_warp=32,
               warps_per_cta=[1, 4], rep_cluster=[1, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=2, threads_per_warp=32,
               warps_per_cta=[8, 4], rep_cluster=[4, 2]),
    DpasLayout(repeatCount=8, systolic_depth=8, execution_size=16, ops_per_chan=1, threads_per_warp=32,
               warps_per_cta=[8, 4], rep_cluster=[1, 1]),
    # Layout for Xe
]


@pytest.mark.parametrize("M, N", [[M, N] for M, N in itertools.product([32, 64, 128, 256], [32, 64, 128, 256])])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.parametrize("layout", layouts)
@pytest.mark.skipif(not is_xpu(), reason="Block load tests are specific to the XPU backend")
def test_tensor_pointer_block_load(M, N, dtype_str, layout, device, tmp_path: pathlib.Path):

    warps = warps_per_cta(layout)
    num_warps = int(np.prod(warps))
    threads_per_warp = layout.threads_per_warp
    ops_per_chan = layout.ops_per_chan
    A_width = 1 if ops_per_chan == 1 else ops_per_chan // 2
    B_width = ops_per_chan

    ty = {"float32": "f32", "float16": "f16", "int8": "i8"}[dtype_str]

    support_block_io = torch.xpu.get_device_capability()['has_subgroup_2d_block_io']

    ir = f"""
    #mma = {layout}
    #dot_a = #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = {A_width}}}>
    #dot_b = #ttg.dot_op<{{opIdx = 1, parent = #mma, kWidth = {B_width}}}>
    module attributes {{triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_bf16_conversion, triton_intel_gpu.support_dpas, {"triton_intel_gpu.support_sg_2d_block," if support_block_io else ""} triton_intel_gpu.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
        tt.func public @tensor_pointer_block_load(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg6: i32 {{tt.divisibility = 16 : i32}}, %arg2: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}, %arg3: !tt.ptr<{ty}> {{tt.divisibility = 16: i32}}, %arg7: i32 {{tt.divisibility = 16 : i32}}) attributes {{noinline = false}} {{
            // A matrix
            %1 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dot_a}}>>
            %2 = tt.expand_dims %1 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dot_a}}>> -> tensor<{M}x1xi32, #dot_a>
            %3 = tt.splat %arg6 : i32 -> tensor<{M}x1xi32, #dot_a>
            %4 = arith.muli %2, %3 : tensor<{M}x1xi32, #dot_a>
            %5 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dot_a}}>>
            %6 = tt.expand_dims %5 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dot_a}}>> -> tensor<1x{N}xi32, #dot_a>
            %7 = tt.broadcast %4 : tensor<{M}x1xi32, #dot_a> -> tensor<{M}x{N}xi32, #dot_a>
            %8 = tt.broadcast %6 : tensor<1x{N}xi32, #dot_a> -> tensor<{M}x{N}xi32, #dot_a>
            %9 = arith.addi %7, %8 : tensor<{M}x{N}xi32, #dot_a>

            %10 = tt.splat %arg0 : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>
            %11 = tt.addptr %10, %9 : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>, tensor<{M}x{N}xi32, #dot_a>
            %12 = tt.load %11 {{triton_intel_gpu.block_io = "row_major"}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>
            %13 = tt.splat %arg1 : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>
            %14 = tt.addptr %13, %9 : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>, tensor<{M}x{N}xi32, #dot_a>
            tt.store %14, %12 {{boundaryCheck = array<i32: 0, 1>}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_a>

            // B matrix
            %22 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dot_b}}>>
            %44 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dot_b}}>>
            %46 = tt.expand_dims %44 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #dot_b}}>> -> tensor<{M}x1xi32, #dot_b>
            %48 = tt.splat %arg7 : i32 -> tensor<{M}x1xi32, #dot_b>
            %49 = arith.muli %46, %48 : tensor<{M}x1xi32, #dot_b>
            %50 = tt.expand_dims %22 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #dot_b}}>> -> tensor<1x{N}xi32, #dot_b>
            %51 = tt.broadcast %49 : tensor<{M}x1xi32, #dot_b> -> tensor<{M}x{N}xi32, #dot_b>
            %52 = tt.broadcast %50 : tensor<1x{N}xi32, #dot_b> -> tensor<{M}x{N}xi32, #dot_b>
            %53 = arith.addi %51, %52 : tensor<{M}x{N}xi32, #dot_b>

            %54 = tt.splat %arg2 : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>
            %55 = tt.addptr %54, %53 : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>, tensor<{M}x{N}xi32, #dot_b>
            %56 = tt.load %55 {{triton_intel_gpu.block_io = "row_major"}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>
            %57 = tt.splat %arg3 : !tt.ptr<{ty}> -> tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>
            %58 = tt.addptr %57, %53 : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>, tensor<{M}x{N}xi32, #dot_b>
            tt.store %58, %56 {{boundaryCheck = array<i32: 0, 1>}} : tensor<{M}x{N}x!tt.ptr<{ty}>, #dot_b>

            tt.return
        }}
    }}
    """

    torch_dtype = getattr(torch, dtype_str)
    if torch_dtype.is_floating_point:
        a = torch.randn((M, N), dtype=torch_dtype, device=device)
    else:
        a = torch.randint(low=-127, high=128, size=(M, N), dtype=torch_dtype, device=device)

    x = torch.empty_like(a)
    y = torch.empty_like(a)

    temp_file = tmp_path / "test_tensor_pointer_block_load.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    if support_block_io:
        # assert '2d block io' in kernel.asm['llir']
        pass

    kernel[(1, 1, 1)](a, x, a.stride(0), a, y, a.stride(0))

    assert torch.equal(a, x) and torch.equal(a, y)
