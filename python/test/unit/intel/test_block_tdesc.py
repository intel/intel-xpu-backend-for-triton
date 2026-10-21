import functools
import re
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

            %data = tt.descriptor_load %src_desc [%c0_i32, %c0_i32] {{ttig.desc_padding = 1 : i32}}
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

            %data = tt.descriptor_load %src_desc [%c0_i32, %c0_i32] {{ttig.desc_padding = 1 : i32}}
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
            %data = tt.descriptor_load %src_desc [%c0_i32, %c0_i32, %c0_i32, %c0_i32] {{ttig.desc_padding = 1 : i32}}
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
# The stride-one descriptor index was proved through `tt::intel::getFinalValue`, which resolved an
# `scf.for` iteration argument to its *init* operand and never inspected the yielded update. So
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


# Both arms must be exact. The even arm rides the 2D block message; the odd arm is refused and
# lands on the gather fallback, which used to derive its vector width from the descriptor's base
# and pitch divisibility alone and so issued a 128-bit load at a 2-mod-4 byte address for an odd
# 16-bit index. getDescriptorVecSize now folds the load-time index in, so the odd arm is exact too.
@pytest.mark.parametrize("step", [2, 3])
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

    # Prove the gate's decision from the compiled artifact FIRST, so it is checked on both arms
    # and stays meaningful on a device or driver where the odd-X load happens not to corrupt. It
    # assumes the NITER loop still carries `off` in iter_args when MaterializeBlockPointer runs; if it
    # were ever unrolled to constant indices first, the even iterations would be legitimately
    # admitted and this assert would fail for an unrelated reason.
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

    assert torch.equal(c, ref), \
        f"step={step}: {num_wrong}/{M * N} elements wrong with {block_loads} 2D block load(s)"


# Negative column offsets (issue #8479). The descriptor under test is the kernel's only memory read (the other dot
# operand is an arithmetic identity), so the output is the loaded tile and 2D call counts belong to that descriptor.
def _count_2d_calls(llir, kind):
    """Call sites (not declarations) of a SPIR-V or GenISA 2D block message; kind is "Load" or "Store"."""
    names = (f"Subgroup2DBlock{kind}", "GenISA.LSC2DBlockRead" if kind == "Load" else "GenISA.LSC2DBlockWrite")
    return sum(1 for line in llir.splitlines() if re.search(r"\bcall\b", line) and any(n in line for n in names))


@triton.jit
def _neg_col_kernel(c_ptr, x_ptr, LD, R0, OFF, OFFC: tl.constexpr, WHICH: tl.constexpr, TRANS: tl.constexpr,
                    PAD: tl.constexpr, RW: tl.constexpr, CL: tl.constexpr, ACC: tl.constexpr):
    col = OFFC if OFFC is not None else OFF
    t = tl.make_tensor_descriptor(x_ptr, shape=(RW, CL), strides=(LD, 1), block_shape=(RW, CL),
                                  padding_option=PAD).load([R0, col])
    if TRANS:
        t = tl.trans(t)
    TR: tl.constexpr = CL if TRANS else RW
    TC: tl.constexpr = RW if TRANS else CL
    r = tl.arange(0, TR)
    c = tl.arange(0, TC)
    if WHICH == 0:
        acc = tl.dot(t, (c[:, None] == c[None, :]).to(x_ptr.dtype.element_ty), out_dtype=ACC)
    else:
        acc = tl.dot((r[:, None] == r[None, :]).to(x_ptr.dtype.element_ty), t, out_dtype=ACC)
    tl.store(c_ptr + r[:, None] * TC + c[None, :], acc)


def _run_neg_col(dtype_str, which, trans, mis, col, device, pad="zero", rw=64, cl=64, runtime_col=False, r0=0):
    """Load an (rw, cl) tile at [r0, col] from a base `mis` bytes past a 64-byte boundary; return (out, expected, kernel)."""
    dtype = getattr(torch, dtype_str)
    eb = torch.empty(0, dtype=dtype).element_size()
    ld = cl + 64  # elements; keeps the pitch 16-byte aligned for every dtype
    front = 4096  # bytes of poison before the tensor, so a leaked read is visible
    buf = torch.full(((front + mis) // eb + rw * ld + 4096, ), 7, dtype=dtype, device=device)
    assert buf.data_ptr() % 64 == 0
    x = buf[(front + mis) // eb:]
    assert x.data_ptr() % 64 == mis
    data = torch.randint(-2, 3, (rw, cl), generator=torch.Generator().manual_seed(8479)).to(dtype)
    x[:rw * ld].view(rw, ld)[:, :cl] = data.to(device)

    tr, tc = (cl, rw) if trans else (rw, cl)
    is_float = dtype.is_floating_point
    out = torch.full((tr, tc), -999, dtype=torch.float32 if is_float else torch.int32, device=device)
    acc = tl.float32 if is_float else tl.int32
    kernel = _neg_col_kernel[(1, )](out, x, ld, r0, col, None if runtime_col else col, which, trans, pad, rw, cl, acc,
                                    num_warps=4, warp_size=16)

    # tile[i, j] = data[r0 + i, col + j] inside the descriptor, else the padding value.
    ii = torch.arange(rw)[:, None] + r0
    jj = torch.arange(cl)[None, :] + col
    inside = (ii >= 0) & (ii < rw) & (jj >= 0) & (jj < cl)
    fill = float("nan") if pad == "nan" else 0
    tile = torch.where(inside, data[ii.clamp(0, rw - 1), jj.clamp(0, cl - 1)].double(), torch.tensor(fill).double())
    t = tile.T if trans else tile
    if pad == "nan":  # a NaN poisons its whole row (A operand) or column (B operand) of the identity product
        poisoned = t.isnan().any(1, keepdim=True) if which == 0 else t.isnan().any(0, keepdim=True)
        t = torch.where(poisoned, torch.tensor(float("nan")).double(), t)
    return out.cpu().double(), t, kernel


# x is divided by the packing factor; an unsigned divide made a negative x out of bounds for the whole tile. The base is
# aligned, so the compensation is a no-op.
@pytest.mark.parametrize("dtype_str, which, trans", [("float16", 0, 1), ("float16", 1, 1), ("bfloat16", 0, 1),
                                                     ("int8", 0, 0), ("int8", 1, 1)])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor block I/O is specific to the XPU backend")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_tdesc_negative_col_packed_aligned_base(dtype_str, which, trans, device, with_allocator):
    # -4 is legal for every dtype here (int8 needs a multiple of 4 elements).
    out, expected, kernel = _run_neg_col(dtype_str, which, trans, mis=0, col=-4, device=device)
    assert _count_2d_calls(kernel.asm["llir"], "Load") > 0, "the load must stay on 2D block I/O"
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)


# The pad_nan mask must use the logical (pre-compensation) column offset and width.
@pytest.mark.parametrize("which, col", [(0, -8), (1, -16)])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor block I/O is specific to the XPU backend")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_tdesc_nan_padding_negative_col_misaligned_base(which, col, device, with_allocator):
    out, expected, kernel = _run_neg_col("float16", which, trans=0, mis=32, col=col, device=device, pad="nan")
    assert _count_2d_calls(kernel.asm["llir"], "Load") > 0, "the load must stay on 2D block I/O"
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)


# A transposed load pairs the descriptor width with the tile's row dimension; a non-square tile exposes a wrong pairing.
@pytest.mark.parametrize("which, r0", [(0, 0), (1, 16)])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor block I/O is specific to the XPU backend")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_tdesc_nan_padding_transposed_non_square(which, r0, device, with_allocator):
    out, expected, kernel = _run_neg_col("float16", which, trans=1, mis=0, col=0, device=device, pad="nan", rw=32,
                                         cl=64, r0=r0)
    assert _count_2d_calls(kernel.asm["llir"], "Load") > 0, "the load must stay on 2D block I/O"
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)


@functools.lru_cache(maxsize=None)
def _unaffected_load_calls(dtype_str, which, trans, rw, cl, device):
    """2D load calls of the same kernel at column 0, where no negative-column handling is needed."""
    kernel = _run_neg_col(dtype_str, which, trans, mis=0, col=0, device=device, rw=rw, cl=cl)[2]
    calls = _count_2d_calls(kernel.asm["llir"], "Load")
    assert calls > 0, "the reference kernel must use 2D block loads"
    return calls


# Messages wholly left of 0 (-64, -32) are redirected; ones that may straddle 0 (-8, runtime -16) are masked. Both stay 2D.
# -32 mixes messages left of 0 with ones inside the tensor, so each is judged by its own start.
@pytest.mark.parametrize("dtype_str, which, trans, col, runtime, mis", [
    ("float16", 0, 0, -8, False, 32),
    ("float16", 0, 0, -64, False, 32),
    ("float16", 1, 0, -32, False, 16),
    ("float16", 0, 1, -16, True, 32),
    ("bfloat16", 0, 0, -32, False, 16),
    ("int8", 0, 0, -32, False, 16),
    ("int8", 1, 1, -8, False, 32),
])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor block I/O is specific to the XPU backend")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_tdesc_negative_col_misaligned_base(dtype_str, which, trans, col, runtime, mis, device, with_allocator):
    out, expected, kernel = _run_neg_col(dtype_str, which, trans, mis, col, device, runtime_col=runtime)
    assert _count_2d_calls(kernel.asm["llir"], "Load") == _unaffected_load_calls(dtype_str, which, trans, 64, 64,
                                                                                 device)
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("which, col", [(0, -8), (1, -64)])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor block I/O is specific to the XPU backend")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_tdesc_negative_col_transposed_non_square(which, col, device, with_allocator):
    out, expected, kernel = _run_neg_col("float16", which, 1, 32, col, device, rw=32, cl=64)
    assert _count_2d_calls(kernel.asm["llir"], "Load") == _unaffected_load_calls("float16", which, 1, 32, 64, device)
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)


@triton.jit
def _neg_col_store_kernel(x_ptr, LD, OFF, OFFC: tl.constexpr, RW: tl.constexpr, CL: tl.constexpr):
    col = OFFC if OFFC is not None else OFF
    r = tl.arange(0, RW)
    c = tl.arange(0, CL)
    dt = x_ptr.dtype.element_ty
    v = ((r[:, None] * CL + c[None, :]) % 5 + 1).to(dt)
    tile = tl.dot((r[:, None] == r[None, :]).to(dt), v).to(dt)  # the identity dot puts `v` in the accumulator layout
    tl.make_tensor_descriptor(x_ptr, shape=(RW, CL), strides=(LD, 1), block_shape=(RW, CL)).store([0, col], tile)


# Stores cannot be masked per element: redirect when no message can straddle 0, else scatter. Everything outside the
# tensor, including the poison before it and between rows, must stay unchanged.
@pytest.mark.parametrize("dtype_str, col, runtime, stays_2d, mis", [
    ("float16", -8, False, False, 32),
    ("float16", -16, True, True, 16),
    ("float16", -64, False, True, 32),
    ("int8", -32, False, True, 16),
])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor block I/O is specific to the XPU backend")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_tdesc_store_negative_col_misaligned_base(dtype_str, col, runtime, stays_2d, mis, device, with_allocator):
    dtype = getattr(torch, dtype_str)
    eb = torch.empty(0, dtype=dtype).element_size()
    rw = cl = 64
    ld = cl + 64
    front = 4096
    start = (front + mis) // eb  # element index of the tensor within the allocation
    buf = torch.full((start + rw * ld + 4096, ), 7, dtype=dtype, device=device)
    assert buf.data_ptr() % 64 == 0
    kernel = _neg_col_store_kernel[(1, )](buf[start:], ld, col, None if runtime else col, rw, cl, num_warps=4,
                                          warp_size=16)

    v = ((torch.arange(rw)[:, None] * cl + torch.arange(cl)[None, :]) % 5 + 1).to(dtype)
    expected = torch.full(buf.shape, 7, dtype=dtype)
    tensor = expected[start:start + rw * ld].view(rw, ld)
    for j in range(cl):
        if 0 <= col + j < cl:
            tensor[:, col + j] = v[:, j]
    if dtype_str == "float16":  # int8 tiles are wider, so whether they straddle is not asserted
        assert (_count_2d_calls(kernel.asm["llir"], "Store") > 0) == stays_2d
    assert torch.equal(buf.cpu(), expected)
