"""tt.dot_scaled with fewer than 8 rows (issue #8773).

Block scaled DPAS only has an 8-row form. A scaled dot with M = 1, 2 or 4 keeps a small-M DPAS layout and is
padded to 8-row instructions, so its rows must equal the leading rows of the same dot with M = 8, bit for bit.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_compile_warmup, is_xpu


@triton.jit
def _dot_scaled(a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr, d_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                A_FMT: tl.constexpr, B_FMT: tl.constexpr):
    A_K: tl.constexpr = K // 2 if A_FMT == "e2m1" else K
    B_K: tl.constexpr = K // 2 if B_FMT == "e2m1" else K
    rm, rn, rs = tl.arange(0, M), tl.arange(0, N), tl.arange(0, K // 32)
    a = tl.load(a_ptr + rm[:, None] * A_K + tl.arange(0, A_K)[None, :])
    b = tl.load(b_ptr + tl.arange(0, B_K)[:, None] * N + rn[None, :])
    a_scale = tl.load(a_scale_ptr + rm[:, None] * (K // 32) + rs[None, :])
    b_scale = tl.load(b_scale_ptr + rn[:, None] * (K // 32) + rs[None, :])
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :])
    d = tl.dot_scaled(a, a_scale, A_FMT, b, b_scale, B_FMT, acc=c)
    tl.store(d_ptr + rm[:, None] * N + rn[None, :], d)


def _operand(fmt, rows, cols, pack_dim, generator, device):
    if fmt == "e4m3":
        return (torch.randn((rows, cols), generator=generator) * 4).to(torch.float8_e4m3fn).to(device)
    # e2m1: two values per byte along the packed dimension.
    shape = [rows, cols]
    shape[pack_dim] //= 2
    return torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8).to(device)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("a_fmt, b_fmt", [("e4m3", "e2m1"), ("e2m1", "e2m1"), ("e4m3", "e4m3")])
@pytest.mark.parametrize("M", [1, 2, 4])
@pytest.mark.enable_warmup
def test_dot_scaled_small_m_matches_m8(M, a_fmt, b_fmt, device):
    # K spans several DPAS instructions, so the padded accumulator is chained through the K loop.
    N, K = 64, 128
    generator = torch.Generator(device="cpu").manual_seed(0)
    a = _operand(a_fmt, 8, K, 1, generator, device)
    b = _operand(b_fmt, K, N, 0, generator, device)
    a_scale = torch.randint(122, 133, (8, K // 32), generator=generator, dtype=torch.uint8).to(device)
    b_scale = torch.randint(122, 133, (N, K // 32), generator=generator, dtype=torch.uint8).to(device)
    c = torch.randn((8, N), generator=generator).to(device)

    def run(m):
        d = torch.empty((m, N), dtype=torch.float32, device=device)
        kernel = _dot_scaled[(1, )](a[:m], a_scale[:m], b, b_scale, c[:m], d, m, N, K, a_fmt, b_fmt)
        return d, kernel

    d8, _ = run(8)
    d, kernel = run(M)
    if is_compile_warmup():
        # Warmup launches return no compiled kernel; only the compilations matter.
        return
    if triton.runtime.driver.active.get_current_target().arch.get("has_subgroup_scaled_matrix_multiply_accumulate"):
        # The scaled dot is not decomposed and keeps its small-M DPAS layout.
        ttgir = kernel.asm["ttgir"]
        assert "tt.dot_scaled" in ttgir and f"repeatCount = {M}," in ttgir
    torch.testing.assert_close(d, d8[:M], atol=0, rtol=0)
