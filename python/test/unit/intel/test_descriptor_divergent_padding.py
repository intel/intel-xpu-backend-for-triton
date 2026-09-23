"""Regression test for issue #8102: divergent descriptor padding must not
silently degrade to PAD_ZERO.

A kernel builds a device-side tensor descriptor in both arms of a runtime `if`,
asking for `padding_option="zero"` in one arm and `"nan"` in the other, then
loads a tile that runs off the end of the surface. Whichever arm ran at runtime,
the out-of-bounds elements must carry that arm's fill.

Before the fix, the descriptor load's provenance had two
`tt.make_tensor_descriptor` candidates whose `padding` disagreed, so every
producer of the padding decision bailed and the LLVM lowering read "no info" as
PAD_ZERO -- the `"nan"` arm silently got zeros.
"""

import math
import re
import struct

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


@triton.jit
def _divergent_padding_load(in_ptr, out_ptr, cond_ptr, IM, IN, YN, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    # CRITICAL: `cond` is loaded from memory, so it is a genuine runtime value.
    #
    # It must NOT be a kernel argument. An `int` argument would be specialized
    # by the JIT (constexpr-folded), the `if` would be resolved at compile time,
    # only one `tl.make_tensor_descriptor` would survive, the descriptor's
    # provenance would have a single candidate, and the test would pass while
    # testing nothing at all. The structural guard in the test body asserts the
    # `scf.if` really is present in the compiled TTIR so that this cannot rot
    # silently.
    cond = tl.load(cond_ptr) != 0

    if cond:
        x_desc = tl.make_tensor_descriptor(in_ptr, shape=[IM, IN], strides=[IN, 1], block_shape=[M_BLOCK, N_BLOCK],
                                           padding_option="zero")
    else:
        x_desc = tl.make_tensor_descriptor(in_ptr, shape=[IM, IN], strides=[IN, 1], block_shape=[M_BLOCK, N_BLOCK],
                                           padding_option="nan")

    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK

    value = x_desc.load([moffset, noffset])

    offs_m = moffset + tl.arange(0, M_BLOCK)
    offs_n = noffset + tl.arange(0, N_BLOCK)
    tl.store(out_ptr + offs_m[:, None] * YN + offs_n[None, :], value)


def _defining_line(ttir, ssa_name):
    """The line defining `ssa_name`, or None. Handles `%0 = ` and `%0:2 = `."""
    pattern = re.compile(rf"^\s*{re.escape(ssa_name)}(:\d+)?\s*=")
    return next((line for line in ttir.splitlines() if pattern.match(line)), None)


def _load_fill_operand(ttir):
    """SSA name of the `other` (out-of-bounds fill) operand of the masked load.

    A masked `tt.load` with a fill prints as `%r = tt.load %ptrs, %mask, %other :`.
    """
    for line in ttir.splitlines():
        match = re.search(r"=\s*tt\.load\s+(%\S+),\s*(%\S+),\s*(%\S+)\s*:", line)
        if match:
            return match.group(3)
    return None


def _splat_value(line):
    """The scalar value of an `arith.constant dense<V>` splat, or None.

    MLIR prints a non-finite float as its raw bit pattern (`0x7FC00000` for a
    f32 NaN), so that form is decoded rather than string-matched -- `0x7FC00000`
    is not self-evidently NaN to a reader, and the exact spelling is the
    printer's business.
    """
    if line is None:
        return None
    match = re.search(r"arith\.constant\s+dense<([^>]+)>", line)
    if not match:
        return None
    text = match.group(1)
    if text.lower().startswith("0x"):
        return struct.unpack(">f", bytes.fromhex(text[2:].rjust(8, "0")))[0]
    try:
        return float(text)
    except ValueError:
        return None


@pytest.mark.skipif(not is_xpu(), reason="Divergent descriptor padding lowering is specific to the XPU backend")
def test_descriptor_divergent_padding(device):

    # Tensor descriptors require a global memory allocation.
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device=device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # Surface is 48x48, tiles are 32x32 and the grid covers 64x64, so every tile
    # except (0, 0) is partially or wholly out of bounds. That is what makes the
    # padding choice observable.
    IM, IN = 48, 48
    OM, ON = 64, 64
    M_BLOCK = N_BLOCK = 32

    inp = torch.arange(IM * IN, device=device, dtype=torch.float32).reshape(IM, IN)
    grid = (triton.cdiv(OM, M_BLOCK), triton.cdiv(ON, N_BLOCK))

    outs = {}
    handles = {}
    for cond_val in (1, 0):
        cond = torch.full((1, ), cond_val, device=device, dtype=torch.int32)
        out = torch.full((OM, ON), -1.0, device=device, dtype=torch.float32)
        handles[cond_val] = _divergent_padding_load[grid](inp, out, cond, IM, IN, ON, M_BLOCK, N_BLOCK)
        outs[cond_val] = out

    # Structural guard, checked BEFORE the value assertions so that it is
    # exercised even while the value assertions are still red.
    #
    # The whole point of this test is that the descriptor's provenance has TWO
    # candidates that disagree on padding. If `cond` ever stops being a runtime
    # value -- or the frontend starts folding the `if` -- the value assertions
    # below would still pass while exercising a single-candidate trace, i.e. not
    # this bug at all.
    #
    # Note what is NOT asserted: `asm["ttir"]` is captured at the END of
    # `make_ttir`, and `add_rewrite_tensor_descriptor_to_pointer` runs inside
    # that stage (third_party/intel/backend/compiler.py), so by then neither
    # `!tt.tensordesc` nor `tt.make_tensor_descriptor` survives anywhere in the
    # module -- the descriptor is already a tuple of pointer/shape/stride/padding
    # values and the load is a plain masked `tt.load`. Asserting on the
    # descriptor type or on the producer count here would be asserting on IR that
    # no longer exists at this stage.
    #
    # What is asserted instead is the actual subject of #8102: the load's
    # out-of-bounds fill is chosen at RUNTIME between a NaN splat and a zero
    # splat. That one line carries everything this test needs -- the `if` was not
    # constant-folded (there is still a condition to select on), the two
    # descriptors really did disagree on padding (agreeing ones leave a single
    # constant fill, as `test_tdesc_load_zero_padding` shows), and the per-branch
    # fill reached the load.
    ttir = handles[1].asm["ttir"]

    assert "scf.if" in ttir, ("expected the runtime `if` to survive into the compiled TTIR; it was probably "
                              "constant-folded, which would make this test vacuous")

    fill = _load_fill_operand(ttir)
    assert fill is not None, f"no masked `tt.load` with an `other` operand in the compiled TTIR:\n{ttir}"

    fill_def = _defining_line(ttir, fill)
    select = re.search(r"arith\.select\s+%\S+,\s*(%\S+),\s*(%\S+)\s*:", fill_def or "")
    assert select, ("expected the load's out-of-bounds fill to be an `arith.select` on the runtime padding "
                    f"flag, got: {fill_def!r}")

    # Not `sorted()`: NaN compares False against everything, so any
    # comparison-based ordering of these two is undefined. Check membership
    # instead, which is also the honest contract -- the arm order depends on how
    # the expansion spells the padding flag, and this test does not care.
    fills = [_splat_value(_defining_line(ttir, ssa)) for ssa in select.groups()]
    assert any(v == 0.0 for v in fills) and any(v is not None and math.isnan(v) for v in fills), (
        f"expected the fill to select between a zero splat and a NaN splat, got {fills} from {fill_def!r}")

    for cond_val, fill in ((1, 0.0), (0, float("nan"))):
        expected = torch.full((OM, ON), fill, device=device, dtype=torch.float32)
        expected[0:IM, 0:IN] = inp
        torch.testing.assert_close(outs[cond_val], expected, equal_nan=True, msg=lambda m: f"cond={cond_val} (padding "
                                   f"{'zero' if cond_val else 'nan'}): {m}")
