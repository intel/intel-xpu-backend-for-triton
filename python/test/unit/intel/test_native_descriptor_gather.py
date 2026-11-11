"""Native descriptor gathers (#8666) on XPU: a gather on a zero-padded device-side descriptor stays a
ttig.descriptor_gather, and every lowering regime must read zero out of bounds, as the pointer expansion does.

The regimes are per-element loads (R1), per-lane loads (R2) and the sub-group gather load (R3). Each kernel pins its
regime from the LLIR, so a layout change cannot silently move a case to another regime. The reduction kernel has the
vLLM paged-KV layout at sub-group size 32, where packed per-lane loads used to read the wrong columns.
"""

import os
import re

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu

pytestmark = [
    pytest.mark.skipif(not is_xpu(), reason="XPU-specific lowering"),
    pytest.mark.skipif(
        os.environ.get("TRITON_INTEL_DISABLE_NATIVE_DESCRIPTOR_GATHER") == "1",
        reason="native descriptor gathers disabled"),
]


@pytest.fixture(autouse=True)
def _allocator(device):
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=device))


@triton.jit
def _gather_store(out_ptr, in_ptr, idx_ptr, y, X, Y, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
    idx = tl.load(idx_ptr + tl.arange(0, BLOCK_X))
    desc = tl.make_tensor_descriptor(in_ptr, [X, Y], [Y, 1], [1, BLOCK_Y])
    out = desc.gather(idx, y)
    tl.store(out_ptr + tl.arange(0, BLOCK_X)[:, None] * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :], out)


@triton.jit
def _gather_reduce(out_ptr, in_ptr, idx_ptr, w_ptr, y, X, Y, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
    idx = tl.load(idx_ptr + tl.arange(0, BLOCK_X))
    w = tl.load(w_ptr + tl.arange(0, BLOCK_Y))
    desc = tl.make_tensor_descriptor(in_ptr, [X, Y], [Y, 1], [1, BLOCK_Y])
    g = desc.gather(idx, y)
    tl.store(out_ptr + tl.arange(0, BLOCK_X), tl.sum(g.to(tl.float32) * w[None, :], 1))


@triton.jit
def _gather_dot(out_ptr, a_ptr, b_ptr, idx_ptr, y, X, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
    idx = tl.load(idx_ptr + tl.arange(0, BLOCK_M))
    a_desc = tl.make_tensor_descriptor(a_ptr, [X, K], [K, 1], [1, BLOCK_K])
    a = a_desc.gather(idx, y)
    b = tl.load(b_ptr + tl.arange(0, BLOCK_K)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
    tl.store(out_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :], tl.dot(a, b))


def _gather_ref(src, idx, y, block_y):
    rows = idx.long()
    cols = torch.arange(block_y, device=src.device) + y
    valid = ((rows >= 0) & (rows < src.shape[0]))[:, None] & ((cols >= 0) & (cols < src.shape[1]))[None, :]
    out = src[rows.clamp(0, src.shape[0] - 1)][:, cols.clamp(0, src.shape[1] - 1)]
    return torch.where(valid, out, torch.zeros_like(out))


def _make_case(case, n_rows, block_y, dtype, device):
    """Source rows, gather indices and y offset for one out-of-bounds rule."""
    torch.manual_seed(0)
    x, idx_dtype, y = 128, torch.int32, 0
    if case == "i16_negative":
        x, idx_dtype = 33000, torch.int16  # zero-extending -32768.. would land inside the 33000 rows
    src = torch.randn(x, block_y, device=device).to(dtype)
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
    return src, idx.to(idx_dtype), y


def _data_loads(llir, dtype):
    ty = {torch.float16: "half", torch.bfloat16: "bfloat"}[dtype]
    return [int(n) for n in re.findall(rf"load <(\d+) x {ty}>, ptr addrspace\(1\)", llir)]


def _check_native(kernel):
    ttgir = kernel.asm["ttgir"]
    assert "tt.descriptor_gather" not in ttgir
    gathers = re.findall(
        r"ttig\.descriptor_gather .*?, tensor<\d+xi(?:16|32), (.+?)>, i32\) "
        r"-> tensor<\d+x\d+x\w+, (.+?)>(?= loc\(|\s*$)", ttgir, re.M)
    assert len(gathers) == 1
    offsets_enc, result_enc = gathers[0]
    assert offsets_enc == f"#ttg.slice<{{dim = 1, parent = {result_enc}}}>"


CASES = ["in_bounds", "rows", "i16_negative", "cols_straddle", "cols_negative"]


@pytest.mark.parametrize("case", CASES)
def test_gather_store_per_element(case, device):
    src, idx, y = _make_case(case, 32, 64, torch.float16, device)
    out = torch.full((32, 64), 7.0, device=device, dtype=src.dtype)
    k = _gather_store[(1, )](out, src, idx, y, src.shape[0], src.shape[1], 32, 64, num_warps=4)
    _check_native(k)
    assert "lsc_load.ugm" not in k.asm["llir"] and max(_data_loads(k.asm["llir"], src.dtype), default=1) == 1
    torch.testing.assert_close(out, _gather_ref(src, idx, y, 64), atol=0, rtol=0)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gather_reduce_vllm_layout(case, dtype, device):
    src, idx, y = _make_case(case, 64, 128, dtype, device)
    w = torch.randn(128, device=device)
    out = torch.empty(64, device=device)
    k = _gather_reduce[(1, )](out, src, idx, w, y, src.shape[0], src.shape[1], 64, 128, num_warps=4)
    _check_native(k)
    assert "lsc_load.ugm" not in k.asm["llir"] and max(_data_loads(k.asm["llir"], dtype), default=1) == 1
    ref = (_gather_ref(src, idx, y, 128).float() * w[None, :]).sum(1)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("case", CASES)
def test_gather_dot_sub_group(case, device):
    a, idx, y = _make_case(case, 64, 32, torch.float16, device)
    b = torch.randn(32, 64, device=device, dtype=torch.float16)
    out = torch.empty(64, 64, device=device)
    k = _gather_dot[(1, )](out, a, b, idx, y, a.shape[0], a.shape[1], 64, 64, 32, num_warps=4)
    _check_native(k)
    assert "lsc_load.ugm" in k.asm["llir"]
    ref = _gather_ref(a, idx, y, 32).float() @ b.float()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


# Hand-written TTGIR pins layouts the frontend does not reliably produce: lanes spanning rows must leave the fast
# paths, and R3's predicated-off channels must read zero even when the registers hold an earlier iteration's rows.
_TTGIR = """
#blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [{tr}, {tc}], warpsPerCTA = [1, 4], order = [1, 0]}}>
#sx = #ttg.slice<{{dim = 1, parent = #blocked}}>
#sy = #ttg.slice<{{dim = 0, parent = #blocked}}>
module attributes {{{attrs}}} {{
  tt.func public @gather_iters(%in: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}, %idxp: !tt.ptr<i32> {{tt.divisibility = 16 : i32}},
                               %out: !tt.ptr<f16> {{tt.divisibility = 16 : i32}}) {{
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant 32 : i32
    %c2048 = arith.constant 2048 : i32
    %c256 = arith.constant 256 : i32
    %c64 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %citers = arith.constant {iters} : i32
    %c64t = arith.constant dense<64> : tensor<32x1xi32, #blocked>
    %r = tt.make_range {{end = 32 : i32, start = 0 : i32}} : tensor<32xi32, #sx>
    %cr = tt.make_range {{end = 64 : i32, start = 0 : i32}} : tensor<64xi32, #sy>
    %desc = tt.make_tensor_descriptor %in, [%c256, %c64], [%c64_i64, %c1_i64] : <f16>, <1x64xf16>
    scf.for %it = %c0 to %citers step %c1 : i32 {{
      %ib = arith.muli %it, %c32 : i32
      %idxb = tt.addptr %idxp, %ib : !tt.ptr<i32>, i32
      %ip = tt.splat %idxb : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #sx>
      %ipp = tt.addptr %ip, %r : tensor<32x!tt.ptr<i32>, #sx>, tensor<32xi32, #sx>
      %idx = tt.load %ipp : tensor<32x!tt.ptr<i32>, #sx>
      %g = ttig.descriptor_gather %desc[%idx, %c0] : (!tt.tensordesc<1x64xf16>, tensor<32xi32, #sx>, i32) -> tensor<32x64xf16, #blocked>
      %ob = arith.muli %it, %c2048 : i32
      %outb = tt.addptr %out, %ob : !tt.ptr<f16>, i32
      %rows = tt.expand_dims %r {{axis = 1 : i32}} : tensor<32xi32, #sx> -> tensor<32x1xi32, #blocked>
      %rowoff = arith.muli %rows, %c64t : tensor<32x1xi32, #blocked>
      %cols = tt.expand_dims %cr {{axis = 0 : i32}} : tensor<64xi32, #sy> -> tensor<1x64xi32, #blocked>
      %rb = tt.broadcast %rowoff : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked>
      %cb = tt.broadcast %cols : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
      %off = arith.addi %rb, %cb : tensor<32x64xi32, #blocked>
      %op = tt.splat %outb : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
      %opp = tt.addptr %op, %off : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
      tt.store %opp, %g : tensor<32x64x!tt.ptr<f16>, #blocked>
    }}
    tt.return
  }}
}}
"""


@triton.jit
def _copy16(in_ptr, out_ptr):
    tl.store(out_ptr + tl.arange(0, 16), tl.load(in_ptr + tl.arange(0, 16)))


@pytest.mark.parametrize("lanes_on_rows, iters", [(True, 1), (False, 2)])
def test_gather_ttgir_layouts(lanes_on_rows, iters, device, tmp_path):
    x = torch.zeros(16, device=device)
    attrs = re.search(r"^module attributes \{(.*)\} \{$", _copy16[(1, )](x, x).asm["ttgir"], re.M).group(1)
    attrs = re.sub(r'"ttg.threads-per-warp" = \d+', '"ttg.threads-per-warp" = 16', attrs)
    path = tmp_path / "gather.ttgir"
    path.write_text(_TTGIR.format(tr=2 if lanes_on_rows else 1, tc=8 if lanes_on_rows else 16, iters=iters,
                                  attrs=attrs))
    k = triton.compile(str(path), options={"warp_size": 16, "num_warps": 4})
    assert ("lsc_load.ugm" in k.asm["llir"]) == (not lanes_on_rows)
    torch.manual_seed(0)
    src = torch.randn(256, 64, device=device, dtype=torch.float16)
    idx = torch.randperm(256, device=device)[:32 * iters].to(torch.int32)
    last = idx[32 * (iters - 1):]  # the last iteration mixes valid, above and negative rows
    last[1::4] = 256 + torch.arange(8, device=device, dtype=torch.int32)
    last[3::4] = -1 - torch.arange(8, device=device, dtype=torch.int32)
    out = torch.full((32 * iters, 64), 7.0, device=device, dtype=torch.float16)
    k[(1, 1, 1)](src, idx, out)
    torch.testing.assert_close(out, _gather_ref(src, idx, 0, 64), atol=0, rtol=0)
