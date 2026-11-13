"""Single-row descriptor loads lowered to sub-group block reads (#8667).

Each test wraps a plain function in a fresh `triton.jit` and compiles into its own TRITON_CACHE_DIR, because the
in-memory JIT cache ignores TRITON_INTEL_DISABLE_DESCRIPTOR_SG_BLOCK_READ.
"""
import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu

pytestmark = pytest.mark.skipif(not is_xpu(), reason="Sub-group block reads are specific to the XPU backend")

DISABLE_ENV = "TRITON_INTEL_DISABLE_DESCRIPTOR_SG_BLOCK_READ"


def _row_copy_fn(in_ptr, out_ptr, rows, pitch_units, row_off, N: tl.constexpr, PITCH_MULT: tl.constexpr,
                 NAN: tl.constexpr):
    # pitch_units * PITCH_MULT proves the row pitch divisible by exactly PITCH_MULT elements.
    pid = tl.program_id(0)
    desc = tl.make_tensor_descriptor(in_ptr, shape=[rows, N], strides=[pitch_units * PITCH_MULT, 1], block_shape=[1, N],
                                     padding_option="nan" if NAN else "zero")
    x = desc.load([pid + row_off, 0]).reshape(N)
    tl.store(out_ptr + pid * N + tl.arange(0, N), x)


def _compile_run(fn, grid, args, kwargs, monkeypatch, tmp_path, disable=False):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / ("off" if disable else "on")))
    if disable:
        monkeypatch.setenv(DISABLE_ENV, "1")
    else:
        monkeypatch.delenv(DISABLE_ENV, raising=False)
    return triton.jit(fn)[grid](*args, **kwargs)


def _block_reads(k):
    return k.asm["llir"].count("intel_sub_group_block_read")


def _row_copy(dtype, N, warp_size, pitch_pad, mult, monkeypatch, tmp_path, device, nan=False, disable=False):
    """Copies rows -4 .. rows+3 of a [rows, N + pitch_pad] tensor of random bits; rows outside the descriptor pad."""
    rows, off = 64, -4
    bits = torch.randint(-2**15, 2**15, (rows, N + pitch_pad), dtype=torch.int16, device=device)
    x = bits.view(torch.bfloat16) if dtype == torch.bfloat16 else bits.to(torch.int32).mul(65537).view(dtype)
    out = torch.full((rows + 8, N), 7, dtype=dtype, device=device)
    k = _compile_run(_row_copy_fn, (rows + 8, ), (x, out, rows, (N + pitch_pad) // mult, off),
                     dict(N=N, PITCH_MULT=mult, NAN=nan, warp_size=warp_size), monkeypatch, tmp_path, disable)
    idx = torch.arange(rows + 8, device=device) + off
    inside = (idx >= 0) & (idx < rows)
    ref = torch.where(inside[:, None], x[idx.clamp(0, rows - 1), :N], float("nan") if nan else 0.0)
    return k, out, ref, inside


def _bits(t):
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


@pytest.mark.parametrize("dtype, N, warp_size, pitch_pad, mult, block_read",
                         [(torch.float32, 128, 32, 1, 1, True),  # 32-bit words, 4-byte rows
                          (torch.float32, 256, 32, 2, 2, True),  # 64-bit words, 8-byte rows
                          (torch.bfloat16, 256, 32, 2, 2, True),  # 32-bit words of two bf16
                          (torch.bfloat16, 512, 32, 4, 4, True),  # 64-bit words of four bf16
                          (torch.float32, 64, 16, 1, 1, True), (torch.float32, 128, 16, 2, 2, True),
                          (torch.bfloat16, 128, 32, 0, 1, False),  # 16-bit words keep the gathers
                          (torch.float32, 256, 32, 1, 1, False),  # 64-bit words on 4-byte rows keep the gathers
                          ])
def test_row_copy(dtype, N, warp_size, pitch_pad, mult, block_read, device, with_allocator, monkeypatch, tmp_path):
    k, out, ref, _ = _row_copy(dtype, N, warp_size, pitch_pad, mult, monkeypatch, tmp_path, device)
    assert (_block_reads(k) > 0) == block_read
    assert torch.equal(_bits(out), _bits(ref))


def test_nan_padding(device, with_allocator, monkeypatch, tmp_path):
    k, out, ref, inside = _row_copy(torch.float32, 128, 32, 0, 1, monkeypatch, tmp_path, device, nan=True)
    assert _block_reads(k) > 0
    assert torch.equal(_bits(out[inside]), _bits(ref[inside]))
    assert torch.isnan(out[~inside]).all()


def test_disabled(device, with_allocator, monkeypatch, tmp_path):
    k_on, out_on, ref_on, _ = _row_copy(torch.float32, 128, 32, 0, 1, monkeypatch, tmp_path, device)
    k_off, out_off, ref_off, _ = _row_copy(torch.float32, 128, 32, 0, 1, monkeypatch, tmp_path, device, disable=True)
    assert _block_reads(k_on) > 0 and _block_reads(k_off) == 0
    assert torch.equal(_bits(out_on), _bits(ref_on)) and torch.equal(_bits(out_off), _bits(ref_off))


def _split_sum_fn(x_ptr, out_ptr, S, CONDITIONAL: tl.constexpr, N: tl.constexpr):
    # A _fwd_kernel_stage2-like loop: one [1, N] row per split, accumulated in f32.
    pid = tl.program_id(0)
    desc = tl.make_tensor_descriptor(x_ptr + pid * S * N, shape=[S, N], strides=[N, 1], block_shape=[1, N])
    acc = tl.zeros([N], dtype=tl.float32)
    for s in range(0, S):
        if CONDITIONAL:
            if s % 2 == 0:
                acc += desc.load([s, 0]).reshape(N)
        else:
            acc += desc.load([s, 0]).reshape(N)
    tl.store(out_ptr + pid * N + tl.arange(0, N), acc)


@pytest.mark.parametrize("conditional", [False, True])
def test_split_sum(conditional, device, with_allocator, monkeypatch, tmp_path):
    B, S, N = 8, 6, 128
    x = torch.randn(B, S, N, device=device)
    out = torch.empty(B, N, device=device)
    k = _compile_run(_split_sum_fn, (B, ), (x, out, S), dict(CONDITIONAL=conditional, N=N), monkeypatch, tmp_path)
    assert _block_reads(k) > 0
    rows = x[:, ::2] if conditional else x
    acc = torch.zeros(B, N, device=device)
    for s in range(rows.shape[1]):
        acc += rows[:, s]
    assert torch.equal(out, acc)
