import pytest
import torch
import pathlib

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


@pytest.mark.parametrize("M, N",
                         [[256, 64], [256, 32], [128, 32], [128, 16], [64, 64], [64, 32], [32, 32], [16, 64], [16, 16]])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor tests are specific to the XPU backend")
def test_tdesc_load_store(M, N, dtype_str, device, tmp_path: pathlib.Path):
    num_warps = 4
    threads_per_warp = 32

    ty = {"float32": "f32", "float16": "f16", "int8": "i8"}[dtype_str]

    ir = f"""
    #blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [1, {threads_per_warp}], warpsPerCTA = [1, {num_warps}], order = [1, 0]}}>
    module attributes {{ttg.target = "xpu", "ttg.num-warps" = {num_warps} : i32, "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
        tt.func public @descriptor_load_store(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) {{
            %stride_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %cM_i32 = arith.constant {M} : i32
            %cN_i32 = arith.constant {N} : i32
            %c0_i32 = arith.constant 0 : i32

            %src_desc = tt.make_tensor_descriptor %arg0, [%cM_i32, %cN_i32], [%stride_i64, %c1_i64]
                        : !tt.ptr<{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #blocked>

            %data = tt.descriptor_load %src_desc [%c0_i32, %c0_i32]
                    : !tt.tensordesc<{M}x{N}x{ty}, #blocked> -> tensor<{M}x{N}x{ty}, #blocked>

            %dst_desc = tt.make_tensor_descriptor %arg1, [%cM_i32, %cN_i32], [%stride_i64, %c1_i64]
                        : !tt.ptr<{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #blocked>

            tt.descriptor_store %dst_desc [%c0_i32, %c0_i32], %data
                                : !tt.tensordesc<{M}x{N}x{ty}, #blocked>, tensor<{M}x{N}x{ty}, #blocked>

            tt.return
        }}
    }}
    """
    torch.manual_seed(42)

    torch_dtype = getattr(torch, dtype_str)
    if torch_dtype.is_floating_point:
        a = torch.randn((M, N), dtype=torch_dtype, device=device)
    else:
        a = torch.randint(low=-127, high=128, size=(M, N), dtype=torch_dtype, device=device)

    x = torch.empty_like(a)

    temp_file = tmp_path / "test_tdesc_load_store.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x)
    assert torch.equal(a, x)


@pytest.mark.parametrize("M, N",
                         [[256, 64], [256, 32], [128, 32], [128, 16], [64, 64], [64, 32], [32, 32], [16, 64], [16, 16]])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor tests are specific to the XPU backend")
def test_tdesc_load_zero_padding(M, N, dtype_str, device, tmp_path: pathlib.Path):
    """Load a MxN block through a descriptor whose shape is (M-1)x(N-1).

    The last row and last column are out of bounds and must be zero-padded.
    Input is filled with ones so any zero in the output indicates padding.
    """
    num_warps = 4
    threads_per_warp = 32

    ty = {"float32": "f32", "float16": "f16", "int8": "i8"}[dtype_str]

    ir = f"""
    #blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [1, {threads_per_warp}], warpsPerCTA = [1, {num_warps}], order = [1, 0]}}>
    module attributes {{ttg.target = "xpu", "ttg.num-warps" = {num_warps} : i32, "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
        tt.func public @descriptor_load_store_pad(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) {{
            %stride_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %cM_minus1 = arith.constant {M - 1} : i32
            %cN_minus1 = arith.constant {N - 1} : i32
            %cM_i32 = arith.constant {M} : i32
            %cN_i32 = arith.constant {N} : i32
            %c0_i32 = arith.constant 0 : i32

            // Source descriptor with shape (M-1)x(N-1) — last row/col out of bounds
            %src_desc = tt.make_tensor_descriptor %arg0, [%cM_minus1, %cN_minus1], [%stride_i64, %c1_i64]
                        : !tt.ptr<{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #blocked>

            %data = tt.descriptor_load %src_desc [%c0_i32, %c0_i32]
                    : !tt.tensordesc<{M}x{N}x{ty}, #blocked> -> tensor<{M}x{N}x{ty}, #blocked>

            // Destination descriptor with full shape so we can store everything
            %dst_desc = tt.make_tensor_descriptor %arg1, [%cM_i32, %cN_i32], [%stride_i64, %c1_i64]
                        : !tt.ptr<{ty}>, !tt.tensordesc<{M}x{N}x{ty}, #blocked>

            tt.descriptor_store %dst_desc [%c0_i32, %c0_i32], %data
                                : !tt.tensordesc<{M}x{N}x{ty}, #blocked>, tensor<{M}x{N}x{ty}, #blocked>

            tt.return
        }}
    }}
    """

    torch_dtype = getattr(torch, dtype_str)
    a = torch.ones((M, N), dtype=torch_dtype, device=device)
    x = torch.empty_like(a)

    temp_file = tmp_path / "test_tdesc_load_zero_padding.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x)

    # Build expected: ones everywhere except last row and last column are zero-padded
    expected = torch.ones((M, N), dtype=torch_dtype, device=device)
    expected[M - 1, :] = 0
    expected[:, N - 1] = 0

    assert torch.equal(x, expected)


@pytest.mark.parametrize("M, N", [[64, 32], [32, 16], [16, 16]])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor tests are specific to the XPU backend")
def test_tdesc_rank_reducing_load_store(M, N, dtype_str, device, tmp_path: pathlib.Path):
    num_warps = 4
    threads_per_warp = 32

    ty = {"float32": "f32", "float16": "f16", "int8": "i8"}[dtype_str]
    mn = M * N

    ir = f"""
    #blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [1, {threads_per_warp}], warpsPerCTA = [1, {num_warps}], order = [1, 0]}}>
    module attributes {{ttg.target = "xpu", "ttg.num-warps" = {num_warps} : i32, "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
        tt.func public @descriptor_rank_reduce_load_store(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) {{
            %c1_i32 = arith.constant 1 : i32
            %c0_i32 = arith.constant 0 : i32
            %cM_i32 = arith.constant {M} : i32
            %cN_i32 = arith.constant {N} : i32
            %c1_i64 = arith.constant 1 : i64
            %cN_i64 = arith.constant {N} : i64
            %cMN_i64 = arith.constant {mn} : i64

            // 4D descriptor reduced to a 2D tensor: leading singleton dimensions
            // use stride M*N so their zero offsets keep the same contiguous view.
            %src_desc = tt.make_tensor_descriptor %arg0, [%c1_i32, %c1_i32, %cM_i32, %cN_i32], [%cMN_i64, %cMN_i64, %cN_i64, %c1_i64]
                        : !tt.ptr<{ty}>, !tt.tensordesc<1x1x{M}x{N}x{ty}>
            %data = tt.descriptor_load %src_desc [%c0_i32, %c0_i32, %c0_i32, %c0_i32]
                    : !tt.tensordesc<1x1x{M}x{N}x{ty}> -> tensor<{M}x{N}x{ty}, #blocked>

            %dst_desc = tt.make_tensor_descriptor %arg1, [%c1_i32, %c1_i32, %cM_i32, %cN_i32], [%cMN_i64, %cMN_i64, %cN_i64, %c1_i64]
                        : !tt.ptr<{ty}>, !tt.tensordesc<1x1x{M}x{N}x{ty}>
            tt.descriptor_store %dst_desc [%c0_i32, %c0_i32, %c0_i32, %c0_i32], %data
                                : !tt.tensordesc<1x1x{M}x{N}x{ty}>, tensor<{M}x{N}x{ty}, #blocked>
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
    temp_file = tmp_path / "test_tdesc_rank_reducing_load_store.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x)
    assert torch.equal(a, x)


# ------------------------------------------------------------------------------------------------
# Regression test for issue #7990: https://github.com/intel/intel-xpu-backend-for-triton/issues/7990
#
# The stride-one descriptor index was proved through `tt::intel::getFinalValue`, which resolves an
# `scf.for` iteration argument to its *init* operand and never inspects the yielded update. So
# `off = 0; ...; off += STEP` was proved aligned from its initial `0` for any `STEP`, and with an odd
# `STEP` every other iteration issued a 2D block load at an odd X, which the message cannot express
# for 16-bit elements: on BMG that silently returned 3776 of 4096 output elements wrong.
#
# step=2 is legal and must KEEP using 2D block loads -- the tightest legal case, so it guards against
# a fix that refuses too much. step=3 must be refused and its fallback must be exact.
# ------------------------------------------------------------------------------------------------


def _has_2d_block_io():
    """Check if current device supports 2D block I/O."""
    if not is_xpu():
        return False
    return triton.runtime.driver.active.get_current_target().arch.get('has_2d_block_io', False)


def _count_2d_block_loads(llir):
    """Occurrences of the 2D block load symbol in the LLVM IR, declaration included.

    SPIR-V builtin or GenISA fallback. Used only for > 0 / == 0, so the exact count
    does not matter; it is not a call count.
    """
    return llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead')


@triton.jit
def _loop_carried_index_kernel(a_ptr, b_ptr, c_ptr, M, N, KA, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                               BLOCK_K: tl.constexpr, NITER: tl.constexpr, STEP: tl.constexpr):
    """Descriptor GEMM whose K index is loop-carried: init 0, yielded as `off + STEP`."""
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, KA], strides=[KA, 1], block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, KA], strides=[KA, 1], block_shape=[BLOCK_N, BLOCK_K])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    off = 0
    for _ in range(NITER):
        acc += tl.dot(a_desc.load([0, off]), b_desc.load([0, off]).T)
        off += STEP

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    tl.store(c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], acc)


# Both gates are needed and are not redundant: skipif covers a non-XPU backend, xfail covers an XPU
# that lacks the capability (the test would genuinely fail there, not be inapplicable). is_xpu() is
# evaluated twice because _has_2d_block_io needs it to stay collection-safe -- `.arch` is an int, not
# a dict, on a CUDA target.
# The odd arm is expected to fail until the descriptor fallback is fixed: refusing the 2D block
# message hands the load to a fallback that derives its vector width from the descriptor's base and
# pitch divisibility and ignores the load-time index, so it issues a 128-bit load at a 2-mod-4 byte
# address for an odd 16-bit index. That is pre-existing and unrelated to this gate (the lowering of
# the refused path is bit-identical to main), but it means #7990's symptom survives this change on
# affected hardware. Not strict: the fallback may be correct on other architectures.
@pytest.mark.parametrize("step", [
    2,
    pytest.param(
        3, marks=pytest.mark.xfail(
            reason="descriptor fallback over-vectorizes at an odd "
            "stride-one index; #7990 symptom persists", strict=False)),
])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor block I/O is specific to the XPU backend")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_tdesc_loop_carried_index(step, device, with_allocator):
    # `with_allocator` (python/test/conftest.py) registers a global scratch allocator and
    # restores the null one afterwards. Device-side tl.make_tensor_descriptor only consults it
    # when the backend asks for scratch, so this may be inert -- but it is the sanctioned
    # fixture, and setting an allocator inline would leak into every later test in the worker.
    M = N = KA = 64
    BLOCK_M = BLOCK_N = 64
    BLOCK_K = 32
    NITER = 8

    # `off` reaches (NITER - 1) * step, so for step=3 the last tile ends at 21 + BLOCK_K
    # = 53 <= KA: every load is fully in bounds and no arm depends on descriptor
    # out-of-bounds padding semantics.
    assert (NITER - 1) * step + BLOCK_K <= KA
    # KA is also the descriptor's stride-one extent, which the separate extent gate proves
    # via tt.divisibility. Keep it a multiple of 16 so the specializer stamps that hint and
    # this test is decided by the index gate it is about.
    assert KA % 16 == 0

    # Small integers in [-2, 2] held in f16 with an fp32 accumulator: every product and
    # every partial sum is an exactly representable integer (|acc| <= 4 * BLOCK_K * NITER
    # = 1024, far below 2**24). Both the 2D block path and the fallback must therefore
    # agree bit-for-bit, so any nonzero difference is a defect and a tolerance could only
    # reduce sensitivity -- which matters because a misaligned block load returns a shifted
    # slice of the operand rather than noise, so the difference need not be large.
    generator = torch.Generator().manual_seed(17)
    a = torch.randint(-2, 3, (M, KA), generator=generator, dtype=torch.int8).to(torch.float16).to(device)
    b = torch.randint(-2, 3, (N, KA), generator=generator, dtype=torch.int8).to(torch.float16).to(device)
    c = torch.zeros((M, N), dtype=torch.float32, device=device)

    ref = torch.zeros((M, N), dtype=torch.float32, device=device)
    for i in range(NITER):
        k0 = i * step
        ref += a[:, k0:k0 + BLOCK_K].float() @ b[:, k0:k0 + BLOCK_K].float().T

    kernel = _loop_carried_index_kernel[(1, )](a, b, c, M, N, KA, BLOCK_M, BLOCK_N, BLOCK_K, NITER, step, num_warps=8)
    torch.xpu.synchronize()

    block_loads = _count_2d_block_loads(kernel.asm["llir"])
    num_wrong = int((c != ref).sum())

    assert torch.equal(c, ref), \
        f"step={step}: {num_wrong}/{M * N} elements wrong with {block_loads} 2D block load(s)"

    # Prove the gate's decision from the compiled artifact, so the test still means
    # something on a device or driver where the odd-X load happens not to corrupt. This
    # assumes the NITER loop still carries `off` in iter_args when MaterializeBlockPointer
    # runs; if it were ever unrolled to constant indices first, the even iterations would
    # be legitimately admitted and this assert would fail for an unrelated reason.
    #
    # `== 0` is exact only because BOTH descriptors index their stride-one dimension with
    # `off` (strides=[KA, 1], loaded at [0, off]), so every 2D block load in the kernel has
    # to go. An operand indexed by `off` in the ROW position needs no alignment and would
    # legitimately keep its message -- adding one would change this test's premise, not
    # break the fix.
    if step % 2 == 0:
        assert block_loads > 0, \
            f"step={step}: index is 2-aligned and must stay on 2D block I/O, but none was emitted"
    else:
        assert block_loads == 0, \
            f"step={step}: odd index must be refused, but {block_loads} 2D block load(s) were emitted"
