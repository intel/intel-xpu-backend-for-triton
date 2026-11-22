import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu

pytestmark = pytest.mark.skipif(
    not is_xpu() or not triton.runtime.driver.active.get_current_target().arch.get('has_2d_block_io', False),
    reason="2D block IO is specific to the XPU backend")


def _assert_block_io(k, no_slm=True):
    ttgir = k.asm["ttgir"]
    block_loads = k.asm["llir"].count('spirv_Subgroup2DBlockLoad') + k.asm["llir"].count('GenISA.LSC2DBlockRead')
    assert block_loads > 0, "no 2D block load in the LLVM IR"
    if no_slm:
        assert "ttg.local_alloc" not in ttgir, "the tile goes through shared memory"
        assert k.metadata.shared == 0, "the kernel uses shared memory"


@triton.jit
def _gemm(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, lda: tl.constexpr, ldb: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, TRANS_B: tl.constexpr, IV: tl.constexpr,
          SPLITS: tl.constexpr):
    # The shape of #8814: the last column is read in the middle of a split's K loop.
    split = tl.program_id(0)
    iters = tl.cdiv(tl.cdiv(K, BK), SPLITS)
    k0 = split * iters * BK if SPLITS > 1 else 0
    a_desc = tl.make_tensor_descriptor(a_ptr, (M, K), (lda, 1), (BM, BK))
    b_desc = tl.make_tensor_descriptor(b_ptr, (N, K) if TRANS_B else (K, N), (ldb, 1), (BN, BK) if TRANS_B else
                                       (BK, BN))
    acc = tl.zeros((BM, BN), tl.int32 if a_ptr.dtype.element_ty == tl.int8 else tl.float32)
    if IV:
        for k in range(k0, k0 + iters * BK, BK):
            b = b_desc.load([0, k]).T if TRANS_B else b_desc.load([k, 0])
            acc = tl.dot(a_desc.load([0, k]), b, acc)
    else:
        for i in range(iters):
            k = k0 + i * BK
            b = b_desc.load([0, k]).T if TRANS_B else b_desc.load([k, 0])
            acc = tl.dot(a_desc.load([0, k]), b, acc)
    tl.store(c_ptr + split * BM * BN + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], acc)


@triton.jit
def _tile(a_ptr, c_ptr, M: tl.constexpr, W: tl.constexpr, lda: tl.constexpr, y, xq, G: tl.constexpr, BM: tl.constexpr,
          BK: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, (M, W), (lda, 1), (BM, BK))
    a = a_desc.load([y, xq * G])
    k = tl.arange(0, BK)
    c = tl.dot(a, (k[:, None] == k[None, :]).to(a.dtype), out_dtype=tl.int32 if a.dtype == tl.int8 else tl.float32)
    tl.store(c_ptr + tl.arange(0, BM)[:, None] * BK + k[None, :], c)


def _view(rows, cols, dtype, device):
    # Padding that must never be read: NaN, or an int8 value that skews sums.
    pitch = (cols + 63) // 64 * 64 + 64
    buf = torch.full((rows, pitch), 3 if dtype == torch.int8 else float("nan"), dtype=dtype, device=device)
    buf[:, :cols] = torch.randint(-8, 8, (rows, cols), device=device).to(dtype)
    return buf[:, :cols]


def _run_gemm(dtype, K, trans_b, iv, splits, device):
    M, N, BK = 64, 32, 64 if dtype == torch.int8 else 32
    a, b = _view(M, K, dtype, device), _view(N, K, dtype, device) if trans_b else _view(K, N, dtype, device)
    c = torch.empty((splits, M, N), dtype=torch.int32 if dtype == torch.int8 else torch.float32, device=device)
    k = _gemm[(splits, )](a, b, c, M, N, K, a.stride(0), b.stride(0), BM=M, BN=N, BK=BK, TRANS_B=trans_b, IV=iv,
                          SPLITS=splits)
    # Not every XPU supports fp64.
    a, b = a.cpu().double(), (b.T if trans_b else b).cpu().double()
    iters = triton.cdiv(triton.cdiv(K, BK), splits)
    for s in range(splits):
        lo, hi = s * iters * BK, min((s + 1) * iters * BK, K)
        ref = a[:, lo:hi] @ b[lo:hi] if lo < hi else torch.zeros((M, N))
        torch.testing.assert_close(c[s].cpu().double(), ref, rtol=0, atol=0)
    return k, iters, BK


@pytest.mark.parametrize("iv", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("dtype, K", [(torch.float16, 7347), (torch.float16, 33), (torch.bfloat16, 255),
                                      (torch.int8, 131)])
def test_gemm(dtype, K, trans_b, iv, device):
    if dtype == torch.int8 and trans_b:
        pytest.skip("int8 B is not loaded transposed")
    k, _, _ = _run_gemm(dtype, K, trans_b, iv, 1, device)
    # The transposed B tile goes through shared memory regardless.
    _assert_block_io(k, no_slm=not trans_b)


@pytest.mark.parametrize("dtype, K", [(torch.float16, 7347), (torch.bfloat16, 1315), (torch.int8, 2631),
                                      (torch.int8, 2629)])
def test_splitk(dtype, K, device):
    k, iters, BK = _run_gemm(dtype, K, False, False, 4, device)
    assert (K - 1) // BK % iters < iters - 1
    assert k.asm["ttgir"].count("scf.for") == 3, "the K loop is not split in three"
    _assert_block_io(k)


@pytest.mark.parametrize("dtype, W, x", [(torch.float16, 67, x) for x in (-32, 36, 66, 68)] + [(torch.int8, 66, x)
                                                                                               for x in (-64, 4, 64)])
def test_tile_offsets(dtype, W, x, device):
    M, BM, BK, G, y = 48, 64, 64 if dtype == torch.int8 else 32, 4 // dtype.itemsize, -3
    a = _view(M, W, dtype, device)
    c = torch.empty((BM, BK), dtype=torch.int32 if dtype == torch.int8 else torch.float32, device=device)
    k = _tile[(1, )](a, c, M, W, a.stride(0), y, x // G, G=G, BM=BM, BK=BK)
    _assert_block_io(k)
    ref = torch.zeros((BM, BK), dtype=torch.float64)
    r0, r1, c0, c1 = max(y, 0), min(y + BM, M), max(x, 0), min(x + BK, W)
    if r0 < r1 and c0 < c1:
        ref[r0 - y:r1 - y, c0 - x:c1 - x] = a[r0:r1, c0:c1].cpu().double()
    torch.testing.assert_close(c.cpu().double(), ref, rtol=0, atol=0)
