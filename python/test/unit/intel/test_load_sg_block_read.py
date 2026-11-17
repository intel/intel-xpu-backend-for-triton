"""Single-row tl.load lowered to sub-group block reads.

Each test compiles a fresh `triton.jit` into its own TRITON_CACHE_DIR, because the in-memory JIT cache ignores
TRITON_INTEL_DISABLE_DESCRIPTOR_SG_BLOCK_READ. Cache-control annotation is turned off: a block read carries no cache
control, so loads with a cache policy keep the gathers.
"""
import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu

pytestmark = pytest.mark.skipif(not is_xpu(), reason="Sub-group block reads are specific to the XPU backend")

DISABLE_ENV = "TRITON_INTEL_DISABLE_DESCRIPTOR_SG_BLOCK_READ"


def _row_copy_fn(in_ptr, out_ptr, rows, N: tl.constexpr, CG: tl.constexpr):
    # The +1 shifts each row off 16-byte alignment, so each lane holds one element.
    pid = tl.program_id(0)
    offs = tl.arange(0, N)
    mask = tl.full([N], pid < rows, tl.int1)
    ptrs = in_ptr + pid * (N + 1) + 1 + offs
    if CG:
        x = tl.load(ptrs, mask=mask, other=-1.0, cache_modifier=".cg")
    else:
        x = tl.load(ptrs, mask=mask, other=-1.0)
    tl.store(out_ptr + pid * N + offs, x)


def _row_copy(N, warp_size, monkeypatch, tmp_path, device, disable=False, cg=False):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / ("off" if disable else "on")))
    monkeypatch.setenv("TRITON_INTEL_DISABLE_ANNOTATE_CACHE_CONTROL", "1")
    if disable:
        monkeypatch.setenv(DISABLE_ENV, "1")
    else:
        monkeypatch.delenv(DISABLE_ENV, raising=False)
    rows = 64
    x = torch.randn(rows + 8, N + 1, device=device)
    out = torch.empty(rows + 8, N, device=device)
    k = triton.jit(_row_copy_fn)[(rows + 8, )](x, out, rows, N=N, CG=cg, warp_size=warp_size)
    ref = x[:, 1:].clone()
    ref[rows:] = -1.0
    return k, out, ref


def _block_reads(k):
    return k.asm["llir"].count("intel_sub_group_block_read")


@pytest.mark.parametrize("N, warp_size", [(128, 32), (64, 16)])
def test_row_copy(N, warp_size, monkeypatch, tmp_path, device):
    k, out, ref = _row_copy(N, warp_size, monkeypatch, tmp_path, device)
    assert _block_reads(k) > 0
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_cache_modifier_keeps_gathers(monkeypatch, tmp_path, device):
    k, out, ref = _row_copy(128, 32, monkeypatch, tmp_path, device, cg=True)
    assert _block_reads(k) == 0
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_disabled(monkeypatch, tmp_path, device):
    k, out, ref = _row_copy(128, 32, monkeypatch, tmp_path, device, disable=True)
    assert _block_reads(k) == 0
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
