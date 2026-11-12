"""fp8 dot operands that the DPAS lowering converts to fp16 one DPAS operand at a time must give the same
results, bit for bit, as converting the whole operand ahead of the dot
(TRITON_INTEL_DISABLE_LAZY_FP8_DPAS_OPERANDS=1)."""

import pytest
import torch

import triton
import triton.language as tl

KNOB = "TRITON_INTEL_DISABLE_LAZY_FP8_DPAS_OPERANDS"
E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2


@triton.jit
def _fp8_desc_gemm(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0) * BM
    pn = tl.program_id(1) * BN
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
    b_desc = tl.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [BK, BN])
    acc = tl.zeros((BM, BN), c_ptr.dtype.element_ty)
    for k in range(0, K, BK):
        acc = tl.dot(a_desc.load([pm, k]), b_desc.load([k, pn]), acc)
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BM, BN])
    c_desc.store([pm, pn], acc)


def _gemm(a, b, acc_dtype, BM, BN, BK, num_warps):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), dtype=acc_dtype, device=a.device)
    kernel = _fp8_desc_gemm[(M // BM, N // BN)](a, b, c, M, N, K, BM, BN, BK, num_warps=num_warps)
    return c, kernel


@pytest.mark.parametrize("acc_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("a_dtype, b_dtype", [(E4M3, E4M3), (E4M3, E5M2), (E5M2, E4M3), (E5M2, E5M2)])
@pytest.mark.parametrize("BM, BN, BK, num_warps", [(128, 128, 32, 8), (64, 128, 64, 8), (256, 128, 64, 16)])
def test_lazy_fp8_dpas_operands_bitwise(a_dtype, b_dtype, acc_dtype, BM, BN, BK, num_warps, device, monkeypatch,
                                        fresh_triton_cache_scope):
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=device))
    M = N = K = 512
    gen = torch.Generator().manual_seed(0)
    a = torch.randn((M, K), generator=gen).to(a_dtype).to(device)
    b = torch.randn((K, N), generator=gen).to(b_dtype).to(device)

    # The knob is read when compiling, so compile each variant afresh: the in-memory kernel cache is keyed by
    # the kernel's arguments only.
    monkeypatch.delenv(KNOB, raising=False)
    _fp8_desc_gemm.device_caches.clear()
    with fresh_triton_cache_scope():
        lazy, lazy_kernel = _gemm(a, b, acc_dtype, BM, BN, BK, num_warps)
    ttgir = lazy_kernel.asm["ttgir"]
    if "#ttig.dpas" not in ttgir or "ttig.support_subgroup_matrix_multiply_accumulate_bf8" in ttgir:
        pytest.skip("fp8 dots are only extended to fp16 for DPAS without fp8 DPAS")

    monkeypatch.setenv(KNOB, "1")
    _fp8_desc_gemm.device_caches.clear()
    with fresh_triton_cache_scope():
        eager, eager_kernel = _gemm(a, b, acc_dtype, BM, BN, BK, num_warps)

    assert lazy_kernel.asm["llir"] != eager_kernel.asm["llir"], "the dot did not convert its fp8 operands itself"
    assert torch.equal(lazy, eager)
