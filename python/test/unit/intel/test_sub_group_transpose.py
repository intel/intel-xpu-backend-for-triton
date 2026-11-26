"""Regression tests for issue #8749: sub-group transposes of sub-dword elements through local memory.

A `ttg.convert_layout` whose source holds one 16-byte row per lane (coalesced loads of
16-byte-aligned `uint8` rows) and whose destination holds one column per lane is lowered
as a sub-group transpose through SLM. With 1-byte elements the padded SLM rows were 17
bytes long, so the per-lane 16-byte vector loads were not dword aligned while carrying
LLVM's implied alignment, and lane `l` got its rows shifted by `l % 4`.

The same held for 16-bit elements with their 34-byte rows (odd lanes at 2 mod 4): f16, bf16 and
i16 transposes came back wrong on CRI too, while 4-byte elements (68-byte rows) were fine.

`test_sub_group_transpose_through_slm` runs the conversion itself, written in TTGIR, for each
element type. The dot kernel below (the reporter's) hits that conversion on its unpacked 8-bit A operand whenever a
warp owns 16 rows of A and the dot runs at sub-group size 16; the other configurations take the
generic shared-memory conversion and act as controls. Whether the dot takes that path depends on
the operand type: an fp8 dot only on targets with fp8 DPAS (the fp8 operands are widened before the
dot elsewhere), an int8 dot on every DPAS target, so the int8 variant is the portable one and the
test checks the compiled IR for the 8-bit transpose and skips when a target does not take it.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu

# fp4 (E2M1) code -> value; every product of two of these is exact in fp32.
E2M1 = [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6]

# A pure 16x16 sub-group transpose per warp: each lane holds a 16-element row of the loaded tile and
# must end up holding a column, the layout the (`lane = K column`) dot operands are built from.
_TRANSPOSE_TTGIR = """
#blocked = #ttg.blocked<{{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [{warps}, 1], order = [1, 0]}}>
#blocked1 = #ttg.blocked<{{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [{warps}, 1], order = [0, 1]}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = {warps} : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64"}} {{
  tt.func public @transpose(%A: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %C: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) attributes {{noinline = false}} {{
    %c16 = arith.constant dense<16> : tensor<{rows}x1xi32, #blocked>
    %c16t = arith.constant dense<16> : tensor<{rows}x1xi32, #blocked1>
    %rm = tt.make_range {{end = {rows} : i32, start = 0 : i32}} : tensor<{rows}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
    %rk = tt.make_range {{end = 16 : i32, start = 0 : i32}} : tensor<16xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
    %rm2 = tt.expand_dims %rm {{axis = 1 : i32}} : tensor<{rows}xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>> -> tensor<{rows}x1xi32, #blocked>
    %rk2 = tt.expand_dims %rk {{axis = 0 : i32}} : tensor<16xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>> -> tensor<1x16xi32, #blocked>
    %rm16 = arith.muli %rm2, %c16 : tensor<{rows}x1xi32, #blocked>
    %rmb = tt.broadcast %rm16 : tensor<{rows}x1xi32, #blocked> -> tensor<{rows}x16xi32, #blocked>
    %rkb = tt.broadcast %rk2 : tensor<1x16xi32, #blocked> -> tensor<{rows}x16xi32, #blocked>
    %off = arith.addi %rmb, %rkb : tensor<{rows}x16xi32, #blocked>
    %Ab = tt.splat %A : !tt.ptr<{ty}> -> tensor<{rows}x16x!tt.ptr<{ty}>, #blocked>
    %Ap = tt.addptr %Ab, %off : tensor<{rows}x16x!tt.ptr<{ty}>, #blocked>, tensor<{rows}x16xi32, #blocked>
    %a = tt.load %Ap : tensor<{rows}x16x!tt.ptr<{ty}>, #blocked>
    %at = ttg.convert_layout %a : tensor<{rows}x16x{ty}, #blocked> -> tensor<{rows}x16x{ty}, #blocked1>
    %tm = tt.make_range {{end = {rows} : i32, start = 0 : i32}} : tensor<{rows}xi32, #ttg.slice<{{dim = 1, parent = #blocked1}}>>
    %tk = tt.make_range {{end = 16 : i32, start = 0 : i32}} : tensor<16xi32, #ttg.slice<{{dim = 0, parent = #blocked1}}>>
    %tm2 = tt.expand_dims %tm {{axis = 1 : i32}} : tensor<{rows}xi32, #ttg.slice<{{dim = 1, parent = #blocked1}}>> -> tensor<{rows}x1xi32, #blocked1>
    %tk2 = tt.expand_dims %tk {{axis = 0 : i32}} : tensor<16xi32, #ttg.slice<{{dim = 0, parent = #blocked1}}>> -> tensor<1x16xi32, #blocked1>
    %tm16 = arith.muli %tm2, %c16t : tensor<{rows}x1xi32, #blocked1>
    %tmb = tt.broadcast %tm16 : tensor<{rows}x1xi32, #blocked1> -> tensor<{rows}x16xi32, #blocked1>
    %tkb = tt.broadcast %tk2 : tensor<1x16xi32, #blocked1> -> tensor<{rows}x16xi32, #blocked1>
    %toff = arith.addi %tmb, %tkb : tensor<{rows}x16xi32, #blocked1>
    %Cb = tt.splat %C : !tt.ptr<{ty}> -> tensor<{rows}x16x!tt.ptr<{ty}>, #blocked1>
    %Cp = tt.addptr %Cb, %toff : tensor<{rows}x16x!tt.ptr<{ty}>, #blocked1>, tensor<{rows}x16xi32, #blocked1>
    tt.store %Cp, %at : tensor<{rows}x16x!tt.ptr<{ty}>, #blocked1>
    tt.return
  }}
}}
"""

_TRANSPOSE_DTYPES = {
    "i8": torch.uint8, "i16": torch.int16, "f16": torch.float16, "bf16": torch.bfloat16, "i32": torch.int32, "f32":
    torch.float32
}


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.parametrize("ty", list(_TRANSPOSE_DTYPES))
@pytest.mark.parametrize("num_warps", [1, 8])
def test_sub_group_transpose_through_slm(ty, num_warps, device, tmp_path):
    """Each lane's row of a 16x16 tile comes back as a column, for every element size.

    The copy through local memory must be exact; before the fix the 1- and 2-byte rows were not
    dword-aligned and lane `l` read its rows shifted by `l % 4` (zero-filled) or `l % 2`.
    """
    rows = 16 * num_warps
    path = tmp_path / f"transpose_{ty}.ttgir"
    path.write_text(_TRANSPOSE_TTGIR.format(ty=ty, rows=rows, warps=num_warps))
    kernel = triton.compile(str(path))
    assert "intel_sub_group_block_write" in kernel.asm["llir"], "the conversion did not take the sub-group transpose"

    torch.manual_seed(0)
    dtype = _TRANSPOSE_DTYPES[ty]
    values = torch.randint(-100, 100, (rows, 16)) if ty != "i8" else torch.randint(0, 256, (rows, 16))
    a = values.to(dtype).to(device)
    c = torch.zeros_like(a)
    kernel[(1, 1, 1)](a, c)
    torch.testing.assert_close(c, a, atol=0, rtol=0)


@triton.jit
def _e2m1_to_e4m3_bits(code):
    code = code.to(tl.int32)
    mag = code & 7
    bits = tl.where(mag == 0, 0, tl.where(mag == 1, 0x30, (mag + 12) << 2))
    return bits | ((code >> 3) << 7)


@triton.jit
def _packed_dot_kernel(A, B, C, M, N, K, stride_a_m, stride_a_k, stride_b_n, stride_b_k, BM: tl.constexpr,
                       BN: tl.constexpr, BK: tl.constexpr, DTYPE: tl.constexpr):
    # The kernel of issue #8749: A [M, K/2] and B [N, K/2] hold packed nibbles that are unpacked
    # to 8-bit operands in registers right before the dot. The unpacked A operand reaches the dot
    # through an 8-bit sub-group transpose. DTYPE: "fp8" (E2M1 codes to e4m3, the reporter's),
    # "bf16" (the same, then widened: a control), or "int8" (nibbles as signed int8).
    num_n = tl.cdiv(N, BN)
    pid_m = tl.program_id(0) // num_n
    pid_n = tl.program_id(0) % num_n
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK // 2)
    a_ptrs = A + rm[:, None] * stride_a_m + rk[None, :] * stride_a_k
    b_ptrs = B + rn[None, :] * stride_b_n + rk[:, None] * stride_b_k
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in tl.range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs)
        w = tl.load(b_ptrs)
        if DTYPE == "int8":
            aq = tl.interleave((a & 0xF).to(tl.int32) - 8, (a >> 4).to(tl.int32) - 8).to(tl.int8)
            wq = tl.trans(tl.interleave(tl.trans((w & 0xF).to(tl.int32) - 8), tl.trans((w >> 4).to(tl.int32) - 8)))
            acc += tl.dot(aq, wq.to(tl.int8)).to(tl.float32)
        else:
            aq = tl.interleave(_e2m1_to_e4m3_bits(a & 0xF), _e2m1_to_e4m3_bits(a >> 4))
            aq = aq.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
            wq = tl.trans(tl.interleave(tl.trans(_e2m1_to_e4m3_bits(w & 0xF)), tl.trans(_e2m1_to_e4m3_bits(w >> 4))))
            wq = wq.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
            if DTYPE == "bf16":
                acc += tl.dot(aq.to(tl.bfloat16), wq.to(tl.bfloat16))
            else:
                acc += tl.dot(aq, wq)
        a_ptrs += BK // 2 * stride_a_k
        b_ptrs += BK // 2 * stride_b_k
    tl.store(C + rm[:, None] * N + rn[None, :], acc)


def _unpack(p, dtype):
    """[R, C] packed uint8 -> [R, 2C] fp32 values, low nibble first: E2M1 values, or the nibble minus 8 for int8."""
    table = torch.arange(16, dtype=torch.float32) - 8 if dtype == "int8" else torch.tensor(E2M1)
    p = p.cpu()
    return torch.stack([table[(p & 0xF).long()], table[(p >> 4).long()]], -1).flatten(-2)


def _takes_8bit_transpose(kernel) -> bool:
    """Whether the compiled kernel moves its 8-bit A operand through the SLM sub-group transpose."""
    return "intel_sub_group_block_write_uc" in kernel.asm["llir"]


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.parametrize("BM, num_warps, transposes", [(128, 8, True), (128, 4, False), (64, 8, False), (64, 4, True),
                                                       (256, 16, True)])
@pytest.mark.parametrize("dtype", ["int8", "fp8", "bf16"])
def test_packed_nibble_dot(BM, num_warps, transposes, dtype, device):
    M, N, K = 256, 128, 256
    BN, BK = 32, 32
    torch.manual_seed(0)
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
    c = torch.zeros(M, N, device=device)
    grid = (triton.cdiv(M, BM) * (N // BN), )
    kernel = _packed_dot_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), BM, BN, BK,
                                      dtype, num_warps=num_warps, num_stages=2)
    ref = _unpack(a, dtype) @ _unpack(b, dtype).T
    torch.testing.assert_close(c.cpu(), ref, atol=0, rtol=0)
    if dtype == "bf16":
        # The control: widened before the dot, so never through the 8-bit transpose.
        assert not _takes_8bit_transpose(kernel)
    elif transposes and not _takes_8bit_transpose(kernel):
        # A target without 8-bit DPAS for this type (fp8 on PVC) widens the operands first.
        pytest.skip(f"the {dtype} dot does not take the 8-bit sub-group transpose on this target")
    elif not transposes:
        assert not _takes_8bit_transpose(kernel), "a control configuration took the 8-bit transpose"
