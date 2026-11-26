"""Native descriptor scatters (#8868) on XPU: a tt.dot result scattered through an in-kernel descriptor, at a column
offset proven 16-byte aligned, lowers as a ttig.descriptor_scatter and drops out-of-range rows and columns exactly as
the pointer expansion (TRITON_INTEL_DISABLE_NATIVE_DESCRIPTOR_SCATTER=1) does.
"""

import os
import re

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu

ENV = "TRITON_INTEL_DISABLE_NATIVE_DESCRIPTOR_SCATTER"
pytestmark = [
    pytest.mark.skipif(not is_xpu(), reason="XPU-specific lowering"),
    pytest.mark.skipif(
        os.environ.get(ENV, "").lower() in ("1", "true", "on"), reason="native descriptor scatters disabled"),
]


@pytest.fixture(autouse=True)
def _allocator(device):
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=device))


@triton.jit
def _scatter_dot(out_ptr, a_ptr, eye_ptr, x_ptr, y, X, w_ptr, P, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
    # The identity B makes the dot result its A tile exactly, and loaded rows keep the scatter from becoming a
    # descriptor store. multiple_of drops a hint on a kernel argument, so the hinted width is a loaded scalar.
    r = tl.arange(0, BLOCK_X)
    c = tl.arange(0, BLOCK_Y)
    a = tl.load(a_ptr + r[:, None] * BLOCK_Y + c[None, :])
    eye = tl.load(eye_ptr + c[:, None] * BLOCK_Y + c[None, :])
    desc = tl.make_tensor_descriptor(out_ptr, [X, tl.multiple_of(tl.load(w_ptr), 8)], [P, 1], [1, BLOCK_Y])
    desc.scatter(tl.dot(a, eye).to(tl.float16), tl.load(x_ptr + r), y)


def _scatter_ref(out, src, x, y, n_rows, width, pad):
    rows, cols = x.long(), torch.arange(src.shape[1], device=src.device) + y
    keep_rows, keep_cols = (rows >= 0) & (rows < n_rows), (cols >= 0) & (cols < width)
    ref = out.clone()
    ref[pad + rows[keep_rows][:, None], cols[keep_cols][None, :]] = src[keep_rows][:, keep_cols]
    return ref


# A 32x64 tile on 4 warps, whose DPAS rows are 32-column runs, and a 16x32 tile ("warps") that 8 warps cover twice.
@pytest.mark.parametrize("case", ["rows", "cols_partial", "cols_negative", "warps"])
def test_scatter_dot(case, device, monkeypatch):
    block_x, block_y, num_warps = (16, 32, 8) if case == "warps" else (32, 64, 4)
    n_rows, pitch, pad, y, width = block_x + 16, block_y + 32, 8, 0, block_y
    torch.manual_seed(0)
    a = torch.randint(-2048, 2048, (block_x, block_y), device=device).to(torch.float16)
    eye = torch.eye(block_y, device=device, dtype=torch.float16)
    x = torch.randperm(n_rows, device=device)[:block_x]
    if case == "rows":  # valid, above and negative rows interleaved
        x[1::4] = n_rows + torch.arange(block_x // 4, device=device)
        x[3::4] = -1 - torch.arange(block_x // 4, device=device)
    elif case == "cols_partial":
        width = 56  # not a multiple of 16, so each row's second run crosses it
    elif case == "cols_negative":
        y = -16  # each row's first run starts before column 0
    x = x.to(torch.int32)
    # Padding rows and columns past the width hold NaN, so any out-of-range write shows up in the comparison.
    out = torch.full((pad + n_rows + pad, pitch), float("nan"), device=device, dtype=torch.float16)
    outs = [out.clone(), out.clone()]
    args = (a, eye, x, y, n_rows, torch.tensor([width], device=device, dtype=torch.int32), pitch, block_x, block_y)
    monkeypatch.setenv(ENV, "1")
    # A separate JITFunction, as the in-process kernel cache ignores the environment.
    pointer = triton.jit(_scatter_dot.fn)[(1, )](outs[0][pad:], *args, num_warps=num_warps)
    monkeypatch.delenv(ENV)
    native = _scatter_dot[(1, )](outs[1][pad:], *args, num_warps=num_warps)
    assert "ttig.descriptor_scatter" not in pointer.asm["ttgir"]
    assert "ttig.descriptor_scatter" in native.asm["ttgir"]
    if native.metadata.threads_per_warp == 16:  # DPAS: the block-write arm, to global memory
        assert re.search(r"intel_sub_group_block_write_\w+PU3AS1", native.asm["llir"])
    ref = _scatter_ref(out, a, x, y, n_rows, width, pad).view(torch.int16)
    torch.testing.assert_close(outs[0].view(torch.int16), ref)
    torch.testing.assert_close(outs[1].view(torch.int16), ref)
