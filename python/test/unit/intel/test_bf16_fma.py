"""bf16 `tl.fma` is computed in fp32 and rounded once to bf16.

`llvm.fma.bf16` needs SPV_INTEL_bfloat16_arithmetic, which LTS drivers do not
allow (intel/intel-xpu-backend-for-triton#8642), and IGC rounds it toward zero,
like the bf16 `mad` in #8466. The result must equal the exact `a * b + c`
rounded to nearest even, on LTS and non-LTS drivers.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _fma_kernel(a_ptr, b_ptr, c_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    c = tl.load(c_ptr + offs)
    tl.store(out_ptr + offs, tl.fma(a, b, c))


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("lts", [False, True])
def test_bf16_fma(lts, device):
    N = 1024
    target = triton.runtime.driver.active.get_current_target()
    arch = dict(target.arch)
    if lts:
        # Pretend to be an LTS driver: no native bf16 arithmetic.
        arch["driver_version"] = "1.3.29735+27"
        arch.pop("has_bfloat16_arithmetic", None)
    target = GPUTarget(target.backend, arch, target.warp_size)
    signature = {name: "*bf16" for name in ("a_ptr", "b_ptr", "c_ptr", "out_ptr")} | {"N": "constexpr"}
    kernel = triton.compile(ASTSource(_fma_kernel, signature, constexprs={"N": N}), target=target)
    assert "llvm.fma.bf16" not in kernel.asm["llir"]

    generator = torch.Generator(device="cpu").manual_seed(0)
    a, b, c = (torch.randn(N, generator=generator).to(torch.bfloat16) for _ in range(3))
    # Let half of the addends nearly cancel the product.
    c[N // 2:] = -(a[N // 2:].float() * b[N // 2:].float()).to(torch.bfloat16)
    out = torch.empty(N, dtype=torch.bfloat16, device=device)
    kernel[(1, 1, 1)](a.to(device), b.to(device), c.to(device), out)

    # The fp64 result is exact for these operands, so this rounds once.
    ref = (a.double() * b.double() + c.double()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref, atol=0, rtol=0)
