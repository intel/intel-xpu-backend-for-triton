"""2D block stores with rows narrower than a 64 B cache line are written back through L1 on parts whose driver
default L1 store policy bypasses L1 (BMG); elsewhere they keep the default. A kernel that hands data to
another work-group keeps the default too, since a gpu-scope release does not flush L1 on those parts."""
import re

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


def _has_2d_block_io():
    if not is_xpu():
        return False
    return triton.runtime.driver.active.get_current_target().arch.get("has_2d_block_io", False)


def _writes_back():
    from triton.backends.intel.compiler import L1_STORE_WRITE_BYPASS_ARCHES, XPUBackend
    from triton import knobs
    target = triton.runtime.driver.active.get_current_target()
    return XPUBackend(
        target).device_arch in L1_STORE_WRITE_BYPASS_ARCHES and not knobs.intel.disable_store_l1_write_back


def _2d_store_counts(llir):
    """(all 2D block store calls, calls with L1 write-back) over the SPIR-V builtin and GenISA (LTS) forms."""
    spv = re.findall(r"call spir_func void @\w*Subgroup2DBlockStoreINTEL\w*\([^\n]*", llir)
    gen = re.findall(r"call (?:spir_func )?void @llvm\.genx\.GenISA\.LSC2DBlockWrite\.\w+\([^\n]*", llir)
    wb = sum("DecorationCacheControlINTEL" in s for s in spv) + sum(", i1 false, i1 false, i32 7," in g for g in gen)
    return len(spv) + len(gen), wb


@triton.jit
def _gemm(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        acc = tl.dot(a_desc.load([pid_m * BLOCK_M, k]), b_desc.load([k, pid_n * BLOCK_N]), acc)
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(c_desc.dtype))


@pytest.mark.parametrize("dtype_str, narrow", [("float16", True), ("bfloat16", True), ("float32", False)])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor block I/O is specific to the XPU backend")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_gemm_epilogue_store_policy(dtype_str, narrow, device, with_allocator):
    dtype = getattr(torch, dtype_str)
    M = N = 512
    K = 64
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16, device=device)
    b = torch.randn((K, N), dtype=torch.float16, device=device)
    c = torch.empty((M, N), dtype=dtype, device=device)
    kernel = _gemm[(M // 128, N // 128)](a, b, c, M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8)
    stores, wb = _2d_store_counts(kernel.asm["llir"])
    assert stores > 0, "expected 2D block stores"
    # 16-bit rows of an 8x16 block are 32 B and get write-back on write-bypass parts; 32-bit rows fill the line.
    assert wb == (stores if narrow and _writes_back() else 0)
    torch.testing.assert_close(c.float(), a.float() @ b.float(), atol=2e-2, rtol=2e-2)


@triton.jit
def _producer_consumer(x_ptr, out_ptr, flag_ptr, val, LAST: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                       LDST: tl.constexpr):
    pid = tl.program_id(0)
    r = tl.arange(0, BM)
    c = tl.arange(0, BN)
    if pid == 0:
        tile = (tl.zeros((BM, BN), dtype=tl.float32) + val + r[:, None] * BN + c[None, :]).to(tl.float16)
        tile = tl.dot((r[:, None] == r[None, :]).to(tl.float16), tile).to(tl.float16)  # DPAS layout: 2D block store
        tl.make_tensor_descriptor(x_ptr, shape=[BM, BN], strides=[BN, 1], block_shape=[BM, BN]).store([0, 0], tile)
        if LDST:
            tl.atomic_store(flag_ptr, 1, sem="release", scope="gpu")
        else:
            tl.atomic_xchg(flag_ptr, 1, sem="release", scope="gpu")
    elif pid == LAST:
        if LDST:
            while tl.atomic_load(flag_ptr, sem="acquire", scope="gpu") == 0:
                pass
        else:
            while tl.atomic_add(flag_ptr, 0, sem="acquire", scope="gpu") == 0:
                pass
        tl.store(out_ptr + r[:, None] * BN + c[None, :], tl.load(x_ptr + r[:, None] * BN + c[None, :]))


@pytest.mark.parametrize("ldst", [False, True])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor block I/O is specific to the XPU backend")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_cross_work_group_handoff_keeps_default(ldst, device, with_allocator):
    """With write-back the consumer read a stale tile in 200/200 trials on a B570; the atomics keep the default."""
    BM = BN = 64
    want = (torch.arange(BM, device=device)[:, None] * BN + torch.arange(BN, device=device)[None, :]).to(torch.float16)
    for trial in range(20):
        x = torch.full((BM, BN), -1.0, dtype=torch.float16, device=device)
        out = torch.full((BM, BN), -2.0, dtype=torch.float16, device=device)
        flag = torch.zeros(1, dtype=torch.int32, device=device)
        kernel = _producer_consumer[(64, )](x, out, flag, 0.0, 63, BM, BN, ldst, num_warps=4)
        torch.xpu.synchronize()
        assert torch.equal(out, want), f"trial {trial}: the consumer read a stale tile"
    stores, wb = _2d_store_counts(kernel.asm["llir"])
    assert stores > 0 and wb == 0


@triton.jit
def _producer(x_ptr, BM: tl.constexpr, BN: tl.constexpr):
    r = tl.arange(0, BM)
    c = tl.arange(0, BN)
    tile = (tl.zeros((BM, BN), dtype=tl.float32) + r[:, None] * BN + c[None, :]).to(tl.float16)
    tile = tl.dot((r[:, None] == r[None, :]).to(tl.float16), tile).to(tl.float16)
    tl.make_tensor_descriptor(x_ptr, shape=[BM, BN], strides=[BN, 1], block_shape=[BM, BN]).store([0, 0], tile)


@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor block I/O is specific to the XPU backend")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_next_kernel_sees_write_back_store(device, with_allocator):
    BM = BN = 64
    want = (torch.arange(BM, device=device)[:, None] * BN + torch.arange(BN, device=device)[None, :]).to(torch.float16)
    for _ in range(20):
        x = torch.full((BM, BN), -1.0, dtype=torch.float16, device=device)
        kernel = _producer[(1, )](x, BM, BN, num_warps=4)
        assert torch.equal(x.clone(), want)
    stores, wb = _2d_store_counts(kernel.asm["llir"])
    assert stores > 0 and wb == (stores if _writes_back() else 0)
