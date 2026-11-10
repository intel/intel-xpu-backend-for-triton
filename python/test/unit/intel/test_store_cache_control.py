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
    from triton.backends.intel.compiler import XPUBackend
    from triton import knobs
    target = triton.runtime.driver.active.get_current_target()
    return XPUBackend(target).properties["l1_store_write_bypass"] and not knobs.intel.disable_store_l1_write_back


def _2d_store_counts(llir):
    """(all 2D block store calls, calls with L1 write-back) over the SPIR-V builtin and GenISA (LTS) forms."""
    spv = re.findall(r"call spir_func void @\w*Subgroup2DBlockStoreINTEL\w*\([^\n]*", llir)
    gen = re.findall(r"call (?:spir_func )?void @llvm\.genx\.GenISA\.LSC2DBlockWrite\.\w+\([^\n]*", llir)
    # SPIR-V: a CacheControlStoreINTEL (6443) decoration for level 0 (L1) with policy 2 (write-back).
    l1wb = set(re.findall(r"^(!\d+) = !\{i32 6443, i32 0, i32 2, i32 \d+\}$", llir, re.M))

    def spv_wb(call):
        m = re.search(r"!spirv\.DecorationCacheControlINTEL (!\d+)", call)
        node = m and re.search(rf"^{m.group(1)} = !\{{(.*)\}}$", llir, re.M)
        return bool(node) and (m.group(1) in l1wb or any(d in l1wb for d in re.findall(r"!\d+", node.group(1))))

    # GenISA takes the cache control after the transpose and VNNI flags; 7 is L1WB_L3WB.
    wb = sum(map(spv_wb, spv)) + sum(", i1 false, i1 false, i32 7," in g for g in gen)
    return len(spv) + len(gen), wb


@triton.jit
def _gemm(a_ptr, b_ptr, c_ptr, M, N, K, stride_cm, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[stride_cm, 1], block_shape=[BLOCK_M, BLOCK_N])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        acc = tl.dot(a_desc.load([pid_m * BLOCK_M, k]), b_desc.load([k, pid_n * BLOCK_N]), acc)
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(c_desc.dtype))


# Full tiles; then partial last tiles with a row pitch that is not a multiple of 64 B in f16 (528 x 2 B).
@pytest.mark.parametrize("M, N, pitch", [(512, 512, 512), (496, 496, 528)])
@pytest.mark.parametrize("dtype_str, narrow", [("float16", True), ("float32", False)])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor block I/O is specific to the XPU backend")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_gemm_epilogue_store_policy(M, N, pitch, dtype_str, narrow, device, with_allocator):
    dtype = getattr(torch, dtype_str)
    K = 64
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16, device=device)
    b = torch.randn((K, N), dtype=torch.float16, device=device)
    out = torch.full((M, pitch), -7.0, dtype=dtype, device=device)
    c = out[:, :N]
    kernel = _gemm[(triton.cdiv(M, 128), triton.cdiv(N, 128))](a, b, c, M, N, K, pitch, BLOCK_M=128, BLOCK_N=128,
                                                               BLOCK_K=64, num_warps=8)
    stores, wb = _2d_store_counts(kernel.asm["llir"])
    assert stores > 0, "expected 2D block stores"
    # 16-bit rows of an 8x16 block are 32 B and get write-back on write-bypass parts; 32-bit rows fill the line.
    assert wb == (stores if narrow and _writes_back() else 0)
    torch.testing.assert_close(c.float(), a.float() @ b.float(), atol=2e-2, rtol=2e-2)
    assert (out[:, N:] == -7.0).all(), "a store wrote past the end of a row"


@pytest.mark.skipif(not is_xpu(), reason="The L1 store policy is specific to the XPU backend")
def test_inductor_target_matches_driver_target():
    """Inductor passes torch.xpu.get_device_capability() as the target arch; it lacks the driver's "arch" key."""
    from triton.backends.compiler import GPUTarget
    from triton.backends.intel.compiler import XPUBackend
    driver_target = triton.runtime.driver.active.get_current_target()
    inductor_target = GPUTarget("xpu", torch.xpu.get_device_capability(), driver_target.warp_size)
    prop = "l1_store_write_bypass"
    assert XPUBackend(inductor_target).properties[prop] == XPUBackend(driver_target).properties[prop]


@pytest.mark.parametrize("device_arch, disable, writes_back", [("bmg", False, True), ("bmg", True, False),
                                                               ("lnl", False, False), ("ptl", False, False),
                                                               ("pvc", False, False)])
@pytest.mark.skipif(not is_xpu(), reason="The L1 store policy is specific to the XPU backend")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_write_back_policy(device_arch, disable, writes_back, monkeypatch):
    """Only bmg writes narrow rows back, and the knob turns it off. The arch is forced, so any XPU can run this."""
    from triton.backends.intel.compiler import XPUBackend
    from triton.compiler import ASTSource
    monkeypatch.setattr(XPUBackend, "arch_to_impl", {})
    monkeypatch.setenv("TRITON_INTEL_DEVICE_ARCH", device_arch)
    monkeypatch.setenv("TRITON_INTEL_DISABLE_STORE_L1_WRITE_BACK", "1" if disable else "0")
    src = ASTSource(fn=_producer, signature={"x_ptr": "*fp16", "BM": "constexpr", "BN": "constexpr"},
                    constexprs={"BM": 64, "BN": 64})
    stores, wb = _2d_store_counts(
        triton.compile(src, target=triton.runtime.driver.active.get_current_target()).asm["llir"])
    assert stores > 0 and wb == (stores if writes_back else 0)


@triton.jit
def _producer_consumer(x_ptr, out_ptr, flag_ptr, val, LAST: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    r = tl.arange(0, BM)
    c = tl.arange(0, BN)
    if pid == 0:
        tile = (tl.zeros((BM, BN), dtype=tl.float32) + val + r[:, None] * BN + c[None, :]).to(tl.float16)
        tile = tl.dot((r[:, None] == r[None, :]).to(tl.float16), tile).to(tl.float16)  # DPAS layout: 2D block store
        tl.make_tensor_descriptor(x_ptr, shape=[BM, BN], strides=[BN, 1], block_shape=[BM, BN]).store([0, 0], tile)
        tl.atomic_xchg(flag_ptr, 1, sem="release", scope="gpu")
    elif pid == LAST:
        # Bounded, so a producer that is never scheduled fails the test instead of hanging the job.
        seen = tl.atomic_add(flag_ptr, 0, sem="acquire", scope="gpu")
        spins = 0
        while (seen == 0) & (spins < 1 << 20):
            seen = tl.atomic_add(flag_ptr, 0, sem="acquire", scope="gpu")
            spins += 1
        if seen != 0:
            tl.store(out_ptr + r[:, None] * BN + c[None, :], tl.load(x_ptr + r[:, None] * BN + c[None, :]))


@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor block I/O is specific to the XPU backend")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_cross_work_group_handoff_keeps_default(device, with_allocator):
    """With write-back the consumer read a stale tile in 200/200 trials on a B570, because the release does not evict
    L1 lines written back on request. The atomics keep the default; the guard can go once the release evicts them."""
    BM = BN = 64
    want = (torch.arange(BM, device=device)[:, None] * BN + torch.arange(BN, device=device)[None, :]).to(torch.float16)
    for trial in range(20):
        x = torch.full((BM, BN), -1.0, dtype=torch.float16, device=device)
        out = torch.full((BM, BN), -2.0, dtype=torch.float16, device=device)
        flag = torch.zeros(1, dtype=torch.int32, device=device)
        kernel = _producer_consumer[(64, )](x, out, flag, 0.0, 63, BM, BN, num_warps=4)
        torch.xpu.synchronize()
        assert not (out == -2.0).all(), f"trial {trial}: the consumer timed out waiting for the producer"
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


@pytest.mark.parametrize("reader", ["same_stream", "host_copy", "other_stream"])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor block I/O is specific to the XPU backend")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_write_back_store_visible_after_kernel(reader, device, with_allocator):
    """Write-back lines are visible once the kernel ends, to whichever reader comes first: a kernel on the same stream,
    a host copy, or a kernel on another stream."""
    BM = BN = 64
    want = (torch.arange(BM)[:, None] * BN + torch.arange(BN)[None, :]).to(torch.float16)
    other = torch.xpu.Stream()
    for _ in range(20):
        x = torch.full((BM, BN), -1.0, dtype=torch.float16, device=device)
        kernel = _producer[(1, )](x, BM, BN, num_warps=4)
        if reader == "host_copy":
            got = x.cpu()
        elif reader == "other_stream":
            other.wait_stream(torch.xpu.current_stream())
            with torch.xpu.stream(other):
                got = (x + 0).cpu()
        else:
            got = (x + 0).cpu()
        assert torch.equal(got, want), f"{reader}: read a stale tile"
    stores, wb = _2d_store_counts(kernel.asm["llir"])
    assert stores > 0 and wb == (stores if _writes_back() else 0)
