"""`out_dtype=tl.bfloat16` is only accepted for bf16 x bf16 and fp8 x fp8 dots.

A bf16 accumulator is an Intel-only feature (other backends reject
`out_dtype=tl.bfloat16`). The frontend accepts it, explicitly or inferred from a
bf16 `acc`, only for the operand types DPAS has a bf16 accumulator for: bf16, and
fp8 (on targets with fp8 DPAS; elsewhere fp8 dots take the FMA path). Other
operand types are rejected: accumulate in fp32 and cast the result with
`.to(tl.bfloat16)`.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu

FP8_PAIRS = [(torch.float8_e5m2, torch.float8_e5m2), (torch.float8_e4m3fn, torch.float8_e4m3fn),
             (torch.float8_e4m3fn, torch.float8_e5m2)]


def _arch():
    return triton.runtime.driver.active.get_current_target().arch


@triton.jit
def _dot_bf16_out(a_ptr, b_ptr, c_ptr, N: tl.constexpr, WITH_ACC: tl.constexpr):
    offs = tl.arange(0, N)
    idx = offs[:, None] * N + offs[None, :]
    a = tl.load(a_ptr + idx)
    b = tl.load(b_ptr + idx)
    if WITH_ACC:
        # out_dtype is inferred from the bf16 accumulator.
        out = tl.dot(a, b, tl.load(c_ptr + idx))
    else:
        out = tl.dot(a, b, out_dtype=tl.bfloat16)
    tl.store(c_ptr + idx, out)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("in_dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("with_acc", [False, True])
def test_dot_bf16_out_dtype_rejected(in_dtype, with_acc, device):
    if in_dtype == torch.float64 and not _arch()["has_fp64"]:
        pytest.skip("device has no fp64 support")
    N = 32
    a = torch.ones((N, N), dtype=in_dtype, device=device)
    out = torch.zeros((N, N), dtype=torch.bfloat16, device=device)
    with pytest.raises(triton.CompilationError) as exc_info:
        _dot_bf16_out[(1, )](a, a, out, N, WITH_ACC=with_acc)
    assert "out_dtype=bfloat16 is unsupported" in str(exc_info.value.__cause__ or exc_info.value)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("a_dtype, b_dtype", [(torch.bfloat16, torch.bfloat16)] + FP8_PAIRS)
@pytest.mark.parametrize("with_acc", [False, True])
def test_dot_bf16_out_dtype_supported(a_dtype, b_dtype, with_acc, device):
    N = 32
    generator = torch.Generator(device="cpu").manual_seed(0)
    a = (torch.randn((N, N), generator=generator) / 2).to(a_dtype)
    b = (torch.randn((N, N), generator=generator) / 2).to(b_dtype)
    c = torch.randn((N, N), generator=generator).to(torch.bfloat16) if with_acc else torch.zeros(
        (N, N), dtype=torch.bfloat16)
    expected = c.float() + a.float() @ b.float()

    out = c.to(device)
    kernel = _dot_bf16_out[(1, )](a.to(device), b.to(device), out, N, WITH_ACC=with_acc)
    if a_dtype == torch.bfloat16 and _arch()["has_subgroup_matrix_multiply_accumulate"]:
        assert "#ttig.dpas" in kernel.asm["ttgir"], "bf16 accumulator dot no longer uses DPAS"
    # Dots that do not use DPAS (fp8 without fp8 DPAS, mixed fp8 formats, or no
    # DPAS on the target) take the FMA path, which accumulates in bf16 and
    # truncates after every multiply-add (intel/intel-xpu-backend-for-triton#8466).
    uses_dpas = "#ttig.dpas" in kernel.asm["ttgir"]
    torch.testing.assert_close(out.float().cpu(), expected, atol=0.1 if uses_dpas else 0.5, rtol=2e-2)
