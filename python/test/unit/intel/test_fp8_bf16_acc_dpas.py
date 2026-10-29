"""fp8 x fp8 -> bf16 dots use DPAS on targets without fp8 DPAS.

DPAS has a bf16 accumulator only for bf16 operands (and for fp8 on targets with
fp8 DPAS). Without fp8 DPAS, fp8 operands with a bf16 accumulator are extended
to fp16, which is exact for e5m2 and e4m3, and the dot accumulates in fp32 on
DPAS and rounds the result to bf16 once (RTNE); mixed fp8 formats are extended
the same way. The result must match, bit for bit, the same kernel written with
fp16 operands, an fp32 accumulator and an explicit cast to bf16.

Mixed fp8 formats with an fp32 accumulator are extended to fp16 like
same-format fp8, so they use DPAS too instead of the FMA path.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu

FP8_PAIRS = [(torch.float8_e5m2, torch.float8_e5m2), (torch.float8_e4m3fn, torch.float8_e4m3fn),
             (torch.float8_e4m3fn, torch.float8_e5m2), (torch.float8_e5m2, torch.float8_e4m3fn)]


@triton.jit
def _fp8_dot(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, REFERENCE: tl.constexpr):
    off_m = tl.program_id(0) * BLOCK_M
    off_n = tl.program_id(1) * BLOCK_N
    rm = off_m + tl.arange(0, BLOCK_M)
    rn = off_n + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    c_ptrs = c_ptr + rm[:, None] * N + rn[None, :]
    acc = tl.load(c_ptrs)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :])
        b = tl.load(b_ptr + (k + rk)[:, None] * N + rn[None, :])
        if REFERENCE:
            acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc.to(tl.float32)).to(c_ptr.dtype.element_ty)
        else:
            acc = tl.dot(a, b, acc)
    tl.store(c_ptrs, acc)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("a_dtype, b_dtype", FP8_PAIRS)
def test_fp8_dot_bf16_acc_dpas(a_dtype, b_dtype, device):
    arch = triton.runtime.driver.active.get_current_target().arch
    if not arch["has_subgroup_matrix_multiply_accumulate"]:
        pytest.skip("needs DPAS")
    if arch.get("has_subgroup_matrix_multiply_accumulate_bfloat8", False):
        pytest.skip("fp8 x fp8 -> bf16 uses fp8 DPAS on this target")
    M, N, K = 128, 128, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    generator = torch.Generator(device="cpu").manual_seed(0)
    a = (torch.randn((M, K), generator=generator) / 2).to(a_dtype)
    b = (torch.randn((K, N), generator=generator) / 2).to(b_dtype)
    c = torch.randn((M, N), generator=generator).to(torch.bfloat16)

    def run(reference):
        out = c.to(device)
        kernel = _fp8_dot[(M // BLOCK_M, N // BLOCK_N)](a.to(device), b.to(device), out, M, N, K, BLOCK_M, BLOCK_N,
                                                        BLOCK_K, REFERENCE=reference)
        return out, kernel

    out, kernel = run(reference=False)
    ref_out, ref_kernel = run(reference=True)
    assert "#ttig.dpas" in kernel.asm["ttgir"], "fp8 x fp8 -> bf16 dot does not use DPAS"
    assert kernel.metadata.threads_per_warp == ref_kernel.metadata.threads_per_warp
    assert torch.equal(out, ref_out), "result differs from the fp32-accumulator dot with a cast to bf16"

    expected = c.float() + a.float() @ b.float()
    torch.testing.assert_close(out.float().cpu(), expected, atol=0.05, rtol=1e-2)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("a_dtype, b_dtype", [(torch.float8_e4m3fn, torch.float8_e5m2),
                                              (torch.float8_e5m2, torch.float8_e4m3fn)])
def test_mixed_fp8_dot_fp32_acc_dpas(a_dtype, b_dtype, device):
    arch = triton.runtime.driver.active.get_current_target().arch
    if not arch["has_subgroup_matrix_multiply_accumulate"]:
        pytest.skip("needs DPAS")
    if arch.get("has_subgroup_matrix_multiply_accumulate_bfloat8", False):
        pytest.skip("mixed fp8 x fp8 -> fp32 uses the FMA path on this target")
    M, N, K = 128, 128, 128
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    generator = torch.Generator(device="cpu").manual_seed(0)
    a = (torch.randn((M, K), generator=generator) / 2).to(a_dtype)
    b = (torch.randn((K, N), generator=generator) / 2).to(b_dtype)
    c = torch.randn((M, N), generator=generator)

    def run(reference):
        out = c.to(device)
        kernel = _fp8_dot[(M // BLOCK_M, N // BLOCK_N)](a.to(device), b.to(device), out, M, N, K, BLOCK_M, BLOCK_N,
                                                        BLOCK_K, REFERENCE=reference)
        return out, kernel

    out, kernel = run(reference=False)
    ref_out, ref_kernel = run(reference=True)
    assert "#ttig.dpas" in kernel.asm["ttgir"], "mixed fp8 x fp8 -> fp32 dot does not use DPAS"
    assert kernel.metadata.threads_per_warp == ref_kernel.metadata.threads_per_warp
    assert torch.equal(out, ref_out), "result differs from the dot with the operands cast to fp16"

    expected = c + a.float() @ b.float()
    torch.testing.assert_close(out.cpu(), expected, atol=1e-4, rtol=1e-4)
