"""Native descriptor gathers (#8666) on XPU: a gather with a constant zero column offset and a 16-byte-aligned
row access stays a ttig.descriptor_gather, reads zero out of bounds, and emits the same global loads as the pointer
expansion (TRITON_INTEL_DISABLE_NATIVE_DESCRIPTOR_GATHER=1); any other gather is expanded exactly as with that
switch. The reduction kernel has the vLLM paged-KV layout.
"""

import os
import re
from collections import Counter

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu

ENV = "TRITON_INTEL_DISABLE_NATIVE_DESCRIPTOR_GATHER"
pytestmark = [
    pytest.mark.skipif(not is_xpu(), reason="XPU-specific lowering"),
    pytest.mark.skipif(os.environ.get(ENV) == "1", reason="native descriptor gathers disabled"),
]


@pytest.fixture(autouse=True)
def _allocator(device):
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=device))


@triton.jit
def _gather_store(out_ptr, in_ptr, idx_ptr, y, X, Y, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
                  Y0: tl.constexpr = False):
    idx = tl.load(idx_ptr + tl.arange(0, BLOCK_X))
    desc = tl.make_tensor_descriptor(in_ptr, [X, Y], [Y, 1], [1, BLOCK_Y])
    out = desc.gather(idx, 0 if Y0 else y)
    tl.store(out_ptr + tl.arange(0, BLOCK_X)[:, None] * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :], out)


@triton.jit
def _gather_reduce(out_ptr, in_ptr, idx_ptr, w_ptr, y, X, Y, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
                   Y0: tl.constexpr = False):
    idx = tl.load(idx_ptr + tl.arange(0, BLOCK_X))
    w = tl.load(w_ptr + tl.arange(0, BLOCK_Y))
    desc = tl.make_tensor_descriptor(in_ptr, [X, Y], [Y, 1], [1, BLOCK_Y])
    g = desc.gather(idx, 0 if Y0 else y)
    tl.store(out_ptr + tl.arange(0, BLOCK_X), tl.sum(g.to(tl.float32) * w[None, :], 1))


@triton.jit
def _gather_dot(out_ptr, a_ptr, b_ptr, idx_ptr, y, X, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr, Y0: tl.constexpr = False):
    idx = tl.load(idx_ptr + tl.arange(0, BLOCK_M))
    a_desc = tl.make_tensor_descriptor(a_ptr, [X, K], [K, 1], [1, BLOCK_K])
    a = a_desc.gather(idx, 0 if Y0 else y)
    b = tl.load(b_ptr + tl.arange(0, BLOCK_K)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
    tl.store(out_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :], tl.dot(a, b))


@triton.jit
def _gather_hint(out_ptr, in_ptr, idx_ptr, X, w_ptr, P, HINT: tl.constexpr, BLOCK_X: tl.constexpr,
                 BLOCK_Y: tl.constexpr):
    # multiple_of drops a hint on a kernel argument, so the hinted extent is a loaded scalar.
    idx = tl.load(idx_ptr + tl.arange(0, BLOCK_X))
    desc = tl.make_tensor_descriptor(in_ptr, [X, tl.multiple_of(tl.load(w_ptr), HINT)], [P, 1], [1, BLOCK_Y])
    out = desc.gather(idx, 0)
    tl.store(out_ptr + tl.arange(0, BLOCK_X)[:, None] * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :], out)


def _gather_ref(src, idx, y, block_y):
    rows = idx.long()
    cols = torch.arange(block_y, device=src.device) + y
    valid = ((rows >= 0) & (rows < src.shape[0]))[:, None] & ((cols >= 0) & (cols < src.shape[1]))[None, :]
    out = src[rows.clamp(0, src.shape[0] - 1)][:, cols.clamp(0, src.shape[1] - 1)]
    return torch.where(valid, out, torch.zeros_like(out))


def _make_case(case, n_rows, block_y, dtype, device):
    """Source rows, gather indices and y offset for one out-of-bounds rule."""
    torch.manual_seed(0)
    x, idx_dtype, y, cols = 128, torch.int32, 0, block_y
    if case == "i16_negative":
        x, idx_dtype = 33000, torch.int16  # zero-extending -32768.. would land inside the 33000 rows
    elif case == "cols_partial":
        cols = block_y * 5 // 8  # e.g. 80 columns in a 128-wide block: whole chunks fall past the row
    src = torch.randn(x, cols, device=device).to(dtype)
    idx = torch.randperm(x, device=device)[:n_rows]
    if case == "rows":  # valid, above and negative rows interleaved within one load
        idx[1::4] = x + torch.arange(len(idx[1::4]), device=device)
        idx[3::4] = -1 - torch.arange(len(idx[3::4]), device=device)
    elif case == "i16_negative":
        idx[1::2] = -32768 + torch.arange(len(idx[1::2]), device=device)
    elif case == "cols_straddle":
        y = 6  # unaligned window [6, 6 + BLOCK_Y) runs past the last column
    elif case == "cols_negative":
        y = -3
    elif case == "cols_negative_aligned":
        y = -16  # aligned, so whole chunks fall before the row
    return src, idx.to(idx_dtype), y


_BITS = {"half": 16, "bfloat": 16, "float": 32, "double": 64}


def _load_widths(llir):
    """Byte width of every global load and predicated-load call."""
    tys = re.findall(r"= load (<\d+ x \w+>|\w+), ptr addrspace\(1\)", llir)
    tys += re.findall(r"= (?:tail )?call spir_func (<\d+ x \w+>|\w+) @_Z\d+__spirv_PredicatedLoadINTEL", llir)
    widths = Counter()
    for t in tys:
        n, e = re.fullmatch(r"<(\d+) x (\w+)>", t).groups() if t.startswith("<") else (1, t)
        if e in _BITS or re.fullmatch(r"i\d+", e):
            widths[int(n) * (_BITS.get(e) or int(e[1:])) // 8] += 1
    return widths


def _launch_arms(kernel, monkeypatch, admitted, *args, **kwargs):
    """Launch `kernel` natively and compare it with a separate JITFunction compiled, not launched, with the pointer
    expansion: an admitted gather stays native with the same multiset of global load widths, and a rejected one
    compiles to the pointer arm's TTGIR."""
    monkeypatch.setenv(ENV, "1")
    pointer = triton.jit(kernel.fn).warmup(*args, grid=(1, ), **kwargs)
    monkeypatch.delenv(ENV)
    native = kernel[(1, )](*args, **kwargs)
    assert "ttig.descriptor_gather" not in pointer.asm["ttgir"]
    if admitted:
        _check_native(native)
        assert _load_widths(native.asm["llir"]) == _load_widths(pointer.asm["llir"])
    else:
        assert native.asm["ttgir"] == pointer.asm["ttgir"]
    return native


def _aligned(y, src):
    """Launch specialization proves alignment only for a y and a row pitch that are multiples of 16."""
    return y % 16 == 0 and src.shape[1] % 16 == 0


def _admitted(y0, src):
    """The admission rule for these 16-bit kernels: a constexpr y = 0 and a pitch and shape (both src.shape[1])
    that launch specialization proves are multiples of 16 elements."""
    return y0 and src.shape[1] % 16 == 0


def _check_native(kernel):
    ttgir = kernel.asm["ttgir"]
    assert "tt.descriptor_gather" not in ttgir
    gathers = re.findall(
        r"ttig\.descriptor_gather .*?, tensor<\d+xi(?:16|32), (.+?)>, i32\) "
        r"-> tensor<\d+x\d+x\w+, (.+?)>(?= loc\(|\s*$)", ttgir, re.M)
    assert len(gathers) == 1
    offsets_enc, result_enc = gathers[0]
    assert offsets_enc == f"#ttg.slice<{{dim = 1, parent = {result_enc}}}>"


CASES = ["in_bounds", "rows", "i16_negative", "cols_straddle", "cols_negative", "cols_negative_aligned", "cols_partial"]
# Every case with a runtime y, and the y = 0 cases again with a constexpr y.
ARMS = [(c, False) for c in CASES] + [(c, True) for c in ["in_bounds", "rows", "i16_negative", "cols_partial"]]


@pytest.mark.parametrize("case, y0", ARMS)
def test_gather_store(case, y0, device, monkeypatch):
    src, idx, y = _make_case(case, 32, 128, torch.float16, device)
    out = torch.full((32, 128), 7.0, device=device, dtype=src.dtype)
    k = _launch_arms(_gather_store, monkeypatch, _admitted(y0, src), out, src, idx, y, src.shape[0], src.shape[1], 32,
                     128, Y0=y0, num_warps=4)
    assert (16 in _load_widths(k.asm["llir"])) == _aligned(y, src)
    assert torch.equal(out.view(torch.int16), _gather_ref(src, idx, y, 128).view(torch.int16))


@pytest.mark.parametrize("case, y0", ARMS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gather_reduce_vllm_layout(case, y0, dtype, device, monkeypatch):
    src, idx, y = _make_case(case, 64, 128, dtype, device)
    w = torch.randn(128, device=device)
    out = torch.empty(64, device=device)
    _launch_arms(_gather_reduce, monkeypatch, _admitted(y0, src), out, src, idx, w, y, src.shape[0], src.shape[1], 64,
                 128, Y0=y0, num_warps=4)
    ref = (_gather_ref(src, idx, y, 128).float() * w[None, :]).sum(1)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("case, y0", ARMS)
def test_gather_dot(case, y0, device, monkeypatch):
    a, idx, y = _make_case(case, 64, 32, torch.float16, device)
    b = torch.randn(32, 64, device=device, dtype=torch.float16)
    out = torch.empty(64, 64, device=device)
    k = _launch_arms(_gather_dot, monkeypatch, _admitted(y0, a), out, a, b, idx, y, a.shape[0], a.shape[1], 64, 64, 32,
                     Y0=y0, num_warps=4)
    assert "lsc_load.ugm" not in k.asm["llir"]
    ref = _gather_ref(a, idx, y, 32).float() @ b.float()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("width, hint, block_y", [(24, 24, 64), (20, 20, 16), (0, 16, 64)])
def test_gather_shape_hint(width, hint, block_y, device):
    """Admitted shapes whose divisor is not a power of two (24), is clamped by the block (20 in 16), or is maximal
    (an empty row): rows load as 16-byte chunks, none of which straddles the extent."""
    torch.manual_seed(0)
    src = torch.randn(128, 64, device=device, dtype=torch.bfloat16)
    idx = torch.randperm(128, device=device)[:64].to(torch.int32)
    out = torch.full((64, block_y), 7.0, device=device, dtype=src.dtype)
    w = torch.tensor([width], device=device, dtype=torch.int32)
    k = _gather_hint[(1, )](out, src, idx, 128, w, 64, hint, 64, block_y, num_warps=4)
    _check_native(k)
    assert 16 in _load_widths(k.asm["llir"])
    ref = _gather_ref(src[:, :width], idx, 0, block_y) if width else torch.zeros_like(out)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


# Hand-written TTGIR pins a layout the frontend does not reliably produce: vectorized lanes spanning rows.
_TTGIR = """
#blocked = #ttg.blocked<{{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [1, 4], order = [1, 0]}}>
#sx = #ttg.slice<{{dim = 1, parent = #blocked}}>
#sy = #ttg.slice<{{dim = 0, parent = #blocked}}>
module attributes {{{attrs}}} {{
  tt.func public @gather_rows(%in: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %idxp: !tt.ptr<i32> {{tt.divisibility = 16 : i32}},
                              %out: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}{yarg}) {{
    %c0 = arith.constant 0 : i32
    %c256 = arith.constant 256 : i32
    %c64 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64t = arith.constant dense<64> : tensor<32x1xi32, #blocked>
    %r = tt.make_range {{end = 32 : i32, start = 0 : i32}} : tensor<32xi32, #sx>
    %cr = tt.make_range {{end = 64 : i32, start = 0 : i32}} : tensor<64xi32, #sy>
    %desc = tt.make_tensor_descriptor %in, [%c256, %c64], [%c64_i64, %c1_i64] : <f16>, <1x64xf16>
    %ip = tt.splat %idxp : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #sx>
    %ipp = tt.addptr %ip, %r : tensor<32x!tt.ptr<i32>, #sx>, tensor<32xi32, #sx>
    %idx = tt.load %ipp : tensor<32x!tt.ptr<i32>, #sx>
    %g = ttig.descriptor_gather %desc[%idx, {y}] : (!tt.tensordesc<1x64xf16>, tensor<32xi32, #sx>, i32) -> tensor<32x64xf16, #blocked>
    %rows = tt.expand_dims %r {{axis = 1 : i32}} : tensor<32xi32, #sx> -> tensor<32x1xi32, #blocked>
    %rowoff = arith.muli %rows, %c64t : tensor<32x1xi32, #blocked>
    %cols = tt.expand_dims %cr {{axis = 0 : i32}} : tensor<64xi32, #sy> -> tensor<1x64xi32, #blocked>
    %rb = tt.broadcast %rowoff : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %cb = tt.broadcast %cols : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %off = arith.addi %rb, %cb : tensor<32x64xi32, #blocked>
    %op = tt.splat %out : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %opp = tt.addptr %op, %off : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
    tt.store %opp, %g : tensor<32x64x!tt.ptr<f16>, #blocked>
    tt.return
  }}
}}
"""


@triton.jit
def _copy16(in_ptr, out_ptr):
    tl.store(out_ptr + tl.arange(0, 16), tl.load(in_ptr + tl.arange(0, 16)))


@pytest.mark.parametrize("y", [None, 6, -3])
def test_gather_ttgir_lanes_on_rows(y, device, tmp_path):
    """A constant y = 0 loads 4-element chunks; a runtime y, here straddling the row end or starting before it,
    loads one element at a time."""
    x = torch.zeros(16, device=device)
    attrs = re.search(r"^module attributes \{(.*)\} \{$", _copy16[(1, )](x, x).asm["ttgir"], re.M).group(1)
    attrs = re.sub(r'"ttg.threads-per-warp" = \d+', '"ttg.threads-per-warp" = 16', attrs)
    path = tmp_path / "gather.ttgir"
    path.write_text(_TTGIR.format(attrs=attrs, yarg="" if y is None else ", %y: i32", y="%c0" if y is None else "%y"))
    k = triton.compile(str(path), options={"warp_size": 16, "num_warps": 4})
    widths = _load_widths(k.asm["llir"])
    assert "lsc_load.ugm" not in k.asm["llir"] and (8 if y is None else 2) in widths and (8 in widths) == (y is None)
    torch.manual_seed(0)
    src = torch.randn(256, 64, device=device, dtype=torch.float16)
    idx = torch.randperm(256, device=device)[:32].to(torch.int32)
    idx[1::4] = 256 + torch.arange(8, device=device, dtype=torch.int32)  # valid, above and negative rows
    idx[3::4] = -1 - torch.arange(8, device=device, dtype=torch.int32)
    out = torch.full((32, 64), 7.0, device=device, dtype=torch.float16)
    k[(1, 1, 1)](src, idx, out, *([] if y is None else [y]))
    assert torch.equal(out.view(torch.int16), _gather_ref(src, idx, y or 0, 64).view(torch.int16))
