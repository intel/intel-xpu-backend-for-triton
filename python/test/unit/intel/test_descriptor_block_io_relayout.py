"""End-to-end tests for the opt-in descriptor-load block-IO relayout (issue #8458).

With TRITON_INTEL_DESCRIPTOR_BLOCK_IO_RELAYOUT=1, MaterializeBlockPointer gives a row-major
`tt.descriptor_load` a layout LowerTo2DBlockLoad accepts, and marks it `ttig.block_io_anchor` so
RemoveLayoutConversions keeps that layout. The load then lowers to 2D block reads instead of gathers.
The gate is function-wide: only loads in a function with no `tt.dot` or `tt.dot_scaled` are relaid
out, not merely loads that feed no dot. A load whose tile reaches a store is left alone: the store
keeps its own layout, so the new one would cost an SLM convert.

The correctness gates of MaterializeBlockPointer.cpp are function-wide as well: a function is relaid
out only if every descriptor load in it has a provably non-negative column index (G1, #8479) and zero
padding (G3) and, with 16 lanes, every value computed from a reduction is f32 or i32 (G2).

`JITFunction.run` keys its in-memory cache on specialization and options, not on the environment, so
every test wraps a plain module-level function in a fresh `triton.jit` object and compiles it into its
own TRITON_CACHE_DIR.
"""
import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


def _has_2d_block_io():
    """Check if current device supports 2D block I/O."""
    if not is_xpu():
        return False
    return triton.runtime.driver.active.get_current_target().arch.get('has_2d_block_io', False)


def _count_2d_block_loads(llir):
    """Occurrences of the 2D block load symbol in the LLVM IR, declaration included.

    SPIR-V builtin or GenISA fallback. Used only for > 0 / == 0, so the exact count
    does not matter; it is not a call count.
    """
    return llir.count('spirv_Subgroup2DBlockLoad') + llir.count('GenISA.LSC2DBlockRead')


pytestmark = [
    pytest.mark.skipif(not is_xpu(), reason="Descriptor-load block-IO relayout is specific to the XPU backend"),
    pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False),
]

RELAYOUT_ENV = "TRITON_INTEL_DESCRIPTOR_BLOCK_IO_RELAYOUT"
ANCHOR = "ttig.block_io_anchor"
# Fills the memory around the regression tests' tensors: a read that should have been padding shows up in a sum.
POISON = 7.0
# The default launch is 4 warps of 32 lanes; "tuned" is the issue's hand-tuned launch.
TUNED = dict(num_warps=1, warp_size=16, grf_mode="256")
# (launch, output dtype, relaid out with the knob on) for the issue kernel. With 16 lanes G2 refuses a function in
# which a value computed from a reduction is not f32 or i32, since IGC miscompiles f32->f16 conversions of sub-group
# reduction results there. So the issue's bf16 output keeps the gather path at the tuned launch; fp32 is the control.
LAUNCHES = [
    pytest.param({}, torch.bfloat16, True, id="default"),
    pytest.param(TUNED, torch.bfloat16, False, id="tuned-bf16-refused"),
    pytest.param(TUNED, torch.float32, True, id="tuned-fp32"),
]


@pytest.fixture(autouse=True)
def triton_block_io(monkeypatch):
    # TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=0 (e.g. on the CRI simulator) disables the relayout.
    monkeypatch.setenv("TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS", "1")
    yield


def _set_compile_env(monkeypatch, tmp_path, knob):
    """Compile into an empty on-disk cache, with the relayout knob set (on) or unset (off)."""
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    if knob:
        monkeypatch.setenv(RELAYOUT_ENV, "1")
    else:
        monkeypatch.delenv(RELAYOUT_ENV, raising=False)


def _misaligned_tensor(rows, cols, pitch, device, misalign=32):
    """An fp32 [rows, cols] tensor of integers in [-2, 2] with row pitch `pitch`, based `misalign` bytes past a
    64-byte boundary inside an allocation that holds POISON everywhere else, between the rows too.

    Returns the base view, for the kernel, and the values on the CPU, for the reference.
    """
    lead = (4096 + misalign) // 4  # fp32 elements before the base
    buf = torch.full((lead + rows * pitch + 1024, ), POISON, device=device)
    assert buf.data_ptr() % 64 == 0
    vals = torch.randint(-2, 3, (rows, cols), generator=torch.Generator().manual_seed(17)).float()
    base = buf[lead:]
    base[:rows * pitch].view(rows, pitch)[:, :cols] = vals.to(device)
    return base, vals


def _tile_column_sums(vals, c0, width):
    """Column sums of the tile of `vals` that starts at column c0 and is `width` wide, with the columns outside
    `vals` read as zero."""
    cols = torch.arange(width) + c0
    inside = (cols >= 0) & (cols < vals.shape[1])
    return torch.where(inside, vals[:, cols.clamp(0, vals.shape[1] - 1)].sum(0), 0.0)


# The kernels below are plain functions, not @triton.jit: each test wraps one in a fresh JITFunction.


def _reduce_kernel_fn(out_ptr, x_ptr, m_ptr, nseg, H, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    """The issue kernel's descriptor path: weight the rows of an [S, DP] fp32 tile and sum them, in the output's
    dtype (bf16 in the issue)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    seg = tl.arange(0, S)
    dim = tl.arange(0, DP)
    smask = seg < nseg
    m = tl.load(m_ptr + (t * H + h) * S + seg, mask=smask, other=float("-inf"))
    w = tl.exp(m - tl.max(m))
    base = x_ptr + (t * H + h) * (S * DP)
    x = tl.make_tensor_descriptor(base=base, shape=(nseg, D), strides=(DP, 1), block_shape=(S, DP)).load([0, 0])
    tl.store(out_ptr + (t * H + h) * D + dim, tl.sum(x * w[:, None], axis=0).to(out_ptr.dtype.element_ty), mask=dim < D)


def _copy_padded_kernel_fn(out_ptr, x_ptr, nseg, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    """Store the whole [S, DP] descriptor tile with no mask, so its padding is observable."""
    rows = tl.arange(0, S)
    cols = tl.arange(0, DP)
    desc = tl.make_tensor_descriptor(x_ptr, shape=(nseg, D), strides=(DP, 1), block_shape=(S, DP))
    tl.store(out_ptr + rows[:, None] * DP + cols[None, :], desc.load([0, 0]))


def _reduce_padded_kernel_fn(out_ptr, x_ptr, nseg, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr,
                             PAD: tl.constexpr = "zero"):
    """Sum the whole [S, DP] descriptor tile over its rows and store all DP sums with no mask, so its padding
    is observable."""
    desc = tl.make_tensor_descriptor(x_ptr, shape=(nseg, D), strides=(DP, 1), block_shape=(S, DP), padding_option=PAD)
    tl.store(out_ptr + tl.arange(0, DP), tl.sum(desc.load([0, 0]), axis=0))


def _loop_accumulate_kernel_fn(out_ptr, x_ptr, M, N: tl.constexpr, BLOCK_M: tl.constexpr):
    """Sum an [M, N] fp32 tensor over its rows, one [BLOCK_M, N] descriptor load per loop iteration."""
    desc = tl.make_tensor_descriptor(x_ptr, shape=(M, N), strides=(N, 1), block_shape=(BLOCK_M, N))
    acc = tl.zeros((N, ), dtype=tl.float32)
    for i in range(M // BLOCK_M):
        x = desc.load([i * BLOCK_M, 0])
        acc += tl.sum(x, axis=0)
    tl.store(out_ptr + tl.arange(0, N), acc)


def _column_sum_kernel_fn(out_ptr, x_ptr, NCOLS, PITCH, C0: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    """Sum the [BR, BC] fp32 descriptor tile at column C0 over its rows."""
    desc = tl.make_tensor_descriptor(x_ptr, shape=(BR, NCOLS), strides=(PITCH, 1), block_shape=(BR, BC))
    tl.store(out_ptr + tl.arange(0, BC), tl.sum(desc.load([0, C0]), axis=0))


def _column_sum_runtime_offset_kernel_fn(out_ptr, x_ptr, NCOLS, PITCH, c0, BR: tl.constexpr, BC: tl.constexpr):
    """_column_sum_kernel_fn with the column offset a runtime argument."""
    desc = tl.make_tensor_descriptor(x_ptr, shape=(BR, NCOLS), strides=(PITCH, 1), block_shape=(BR, BC))
    tl.store(out_ptr + tl.arange(0, BC), tl.sum(desc.load([0, c0]), axis=0))


def _stencil_kernel_fn(out_ptr, x_ptr, NCOLS, PITCH, BR: tl.constexpr, BC: tl.constexpr):
    """For the BC columns c from program_id * BC, sum the 3-tap stencil x[:, c - 1] + x[:, c] + x[:, c + 1] over
    the BR rows."""
    col = tl.program_id(0) * BC
    desc = tl.make_tensor_descriptor(x_ptr, shape=(BR, NCOLS), strides=(PITCH, 1), block_shape=(BR, BC))
    taps = desc.load([0, col - 1]) + desc.load([0, col]) + desc.load([0, col + 1])
    tl.store(out_ptr + col + tl.arange(0, BC), tl.sum(taps, axis=0))


def _row_sum_kernel_fn(out_ptr, x_ptr, R: tl.constexpr, C: tl.constexpr, ACC: tl.constexpr):
    """Sum each row of an [R, C] descriptor tile in ACC and store the sums in the output's dtype."""
    desc = tl.make_tensor_descriptor(x_ptr, shape=(R, C), strides=(C, 1), block_shape=(R, C))
    sums = tl.sum(desc.load([0, 0]).to(ACC), axis=1)
    tl.store(out_ptr + tl.arange(0, R), sums.to(out_ptr.dtype.element_ty))


@pytest.mark.parametrize("knob", [True, False], ids=["on", "off"])
@pytest.mark.parametrize("launch, out_dtype, relaid_out", LAUNCHES)
@pytest.mark.parametrize("D", [64, 96, 128, 256], ids=lambda d: f"D{d}")
def test_relayout_emits_2d_block_load(D, launch, out_dtype, relaid_out, knob, device, monkeypatch, tmp_path):
    _set_compile_env(monkeypatch, tmp_path, knob)
    kernel = triton.jit(_reduce_kernel_fn)
    S, H, T = 16, 32, 1
    DP = triton.next_power_of_2(D)
    x = torch.randn(T, H, S, DP, device=device)
    m = torch.randn(T, H, S, device=device)
    out = torch.empty(T, H, D, device=device, dtype=out_dtype)
    k = kernel[(T, H)](out, x, m, S, H, D=D, DP=DP, S=S, **launch)
    torch.xpu.synchronize()

    # The TTGIR is captured before LowerTo2DBlockLoad runs (in make_llir), so the anchor is still there.
    ttgir = k.asm["ttgir"]
    block_loads = _count_2d_block_loads(k.asm["llir"])
    if knob and relaid_out:
        assert ANCHOR in ttgir, f"knob on: no {ANCHOR} in the TTGIR, the descriptor load was not relaid out"
        assert block_loads > 0, "knob on: the relaid-out descriptor load did not lower to 2D block reads"
    else:
        # Today's gather path for this kernel, which a refused relayout keeps.
        state = "on, refused" if knob else "off"
        assert ANCHOR not in ttgir, f"knob {state}: {ANCHOR} in the TTGIR"
        assert block_loads == 0, f"knob {state}: {block_loads} 2D block load(s), expected the gather path"


@pytest.mark.parametrize("launch, out_dtype, relaid_out", LAUNCHES)
@pytest.mark.parametrize("D", [64, 96, 128, 256], ids=lambda d: f"D{d}")
@pytest.mark.parametrize("nseg", [1, 5, 15, 16], ids=lambda n: f"nseg{n}")
def test_relayout_correctness(nseg, D, launch, out_dtype, relaid_out, device, monkeypatch, tmp_path):
    _set_compile_env(monkeypatch, tmp_path, knob=True)
    kernel = triton.jit(_reduce_kernel_fn)
    S, H, T = 16, 32, 2
    DP = triton.next_power_of_2(D)
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(T, H, S, DP, generator=generator).to(device)
    # Rows >= nseg are outside the descriptor: they must be zero-filled, not read from memory.
    x[:, :, nseg:, :] = float("nan")
    m = torch.randn(T, H, S, generator=generator).to(device)
    out = torch.empty(T, H, D, device=device, dtype=out_dtype)
    k = kernel[(T, H)](out, x, m, nseg, H, D=D, DP=DP, S=S, **launch)

    ttgir = k.asm["ttgir"]
    block_loads = _count_2d_block_loads(k.asm["llir"])
    if relaid_out:
        assert ANCHOR in ttgir, f"no {ANCHOR} in the TTGIR, the descriptor load was not relaid out"
        assert block_loads > 0, "the relaid-out descriptor load did not lower to 2D block reads"
    else:
        assert ANCHOR not in ttgir, f"{ANCHOR} in the TTGIR, the relayout was not refused"
        assert block_loads == 0, f"{block_loads} 2D block load(s) in a refused relayout, expected the gather path"
    # float64 reference on the CPU: not every XPU supports fp64.
    x64 = x.cpu().to(torch.float64)
    m64 = m.cpu().to(torch.float64)[..., :nseg]
    w = torch.exp(m64 - m64.amax(-1, keepdim=True))
    ref = (x64[:, :, :nseg, :D] * w[..., None]).sum(-2).to(torch.float32).to(device)
    # The bf16 output measured <= 2.65e-2 absolute error at D=256, the fp32 output <= 9.2e-7.
    tol = dict(atol=4e-2, rtol=1.6e-2) if out_dtype == torch.bfloat16 else dict(atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out.float(), ref.float(), **tol)


@pytest.mark.parametrize("D", [96, 128], ids=lambda d: f"D{d}")
@pytest.mark.parametrize("nseg", [1, 15, 16], ids=lambda n: f"nseg{n}")
def test_relayout_reduce_padded(nseg, D, device, monkeypatch, tmp_path):
    _set_compile_env(monkeypatch, tmp_path, knob=True)
    kernel = triton.jit(_reduce_padded_kernel_fn)
    S = 16
    DP = triton.next_power_of_2(D)
    generator = torch.Generator().manual_seed(17)
    # NaN outside the descriptor's [nseg, D] shape: any padding read from memory turns its sum into NaN.
    x = torch.full((S, DP), float("nan"), device=device)
    x[:nseg, :D] = torch.randn(nseg, D, generator=generator).to(device)
    # NaN in the output as well, so the zero check proves the kernel wrote the padded columns.
    out = torch.full((DP, ), float("nan"), device=device)
    k = kernel[(1, )](out, x, nseg, D=D, DP=DP, S=S)

    ref = x[:nseg, :D].cpu().to(torch.float64).sum(0).to(torch.float32).to(device)
    torch.testing.assert_close(out[:D], ref, atol=1e-4, rtol=1e-4)
    assert (out[D:] == 0).all()
    # The padding checked above must come from the 2D block read.
    assert ANCHOR in k.asm["ttgir"], f"no {ANCHOR} in the TTGIR, the descriptor load was not relaid out"
    assert _count_2d_block_loads(k.asm["llir"]) > 0, "the descriptor load did not lower to 2D block reads"


@pytest.mark.parametrize("D", [96, 128], ids=lambda d: f"D{d}")
@pytest.mark.parametrize("nseg", [1, 15, 16], ids=lambda n: f"nseg{n}")
def test_relayout_skips_store_ending_copy(nseg, D, device, monkeypatch, tmp_path):
    # The tile reaches a store, which keeps its own layout, so a relaid-out tile would pay an SLM convert.
    _set_compile_env(monkeypatch, tmp_path, knob=True)
    kernel = triton.jit(_copy_padded_kernel_fn)
    S = 16
    DP = triton.next_power_of_2(D)
    generator = torch.Generator().manual_seed(17)
    x = torch.full((S, DP), float("nan"), device=device)
    x[:nseg, :D] = torch.randn(nseg, D, generator=generator).to(device)
    out = torch.full((S, DP), float("nan"), device=device)
    k = kernel[(1, )](out, x, nseg, D=D, DP=DP, S=S)

    assert ANCHOR not in k.asm["ttgir"], f"knob on: {ANCHOR} in the TTGIR of a copy that ends in a store"
    block_loads = _count_2d_block_loads(k.asm["llir"])
    assert block_loads == 0, f"knob on: {block_loads} 2D block load(s) in a copy that ends in a store"
    assert torch.equal(out[:nseg, :D], x[:nseg, :D])
    assert (out[nseg:, :] == 0).all() and (out[:, D:] == 0).all()


def test_relayout_loop_accumulate(device, monkeypatch, tmp_path):
    _set_compile_env(monkeypatch, tmp_path, knob=True)
    kernel = triton.jit(_loop_accumulate_kernel_fn)
    M, N, BLOCK_M = 64, 128, 16  # 4 row-blocks
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(M, N, generator=generator).to(device)
    out = torch.empty(N, device=device)
    k = kernel[(1, )](out, x, M, N=N, BLOCK_M=BLOCK_M)

    ttgir = k.asm["ttgir"]
    assert "scf.for" in ttgir, "the row-block loop is gone, so the load is no longer in a loop"
    ref = x.cpu().to(torch.float64).sum(0).to(torch.float32).to(device)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    assert ANCHOR in ttgir, f"no {ANCHOR} in the TTGIR, the descriptor load in the loop was not relaid out"
    assert _count_2d_block_loads(k.asm["llir"]) > 0, "the descriptor load in the loop did not lower to 2D block reads"


@pytest.mark.parametrize("c0, runtime_c0, relaid_out", [
    pytest.param(-4, False, False, id="const-4"),
    pytest.param(-8, False, False, id="const-8"),
    pytest.param(0, False, True, id="const0"),
    pytest.param(-8, True, False, id="arg-8"),
    pytest.param(8, True, False, id="arg8"),
])
def test_relayout_column_offset_sign(c0, runtime_c0, relaid_out, device, monkeypatch, tmp_path):
    # Issue #8479: from a base that is not 64-byte aligned, a 2D block read at a negative column offset returns the
    # memory left of the base instead of zeros. So G1 needs a provably non-negative column index, which a runtime
    # argument never is, and the refused load stays a gather. Offset 0 from the same base is the control.
    _set_compile_env(monkeypatch, tmp_path, knob=True)
    BR, BC, PITCH = 16, 64, 96
    x, vals = _misaligned_tensor(BR, BC, PITCH, device)
    out = torch.full((BC, ), float("nan"), device=device)
    if runtime_c0:
        k = triton.jit(_column_sum_runtime_offset_kernel_fn)[(1, )](out, x, BC, PITCH, c0, BR=BR, BC=BC)
    else:
        k = triton.jit(_column_sum_kernel_fn)[(1, )](out, x, BC, PITCH, C0=c0, BR=BR, BC=BC)

    # Exact: the sums are of small integers. A column read from memory instead of padding sums to BR * POISON.
    torch.testing.assert_close(out.cpu(), _tile_column_sums(vals, c0, BC), rtol=0, atol=0)
    ttgir = k.asm["ttgir"]
    block_loads = _count_2d_block_loads(k.asm["llir"])
    if relaid_out:
        assert ANCHOR in ttgir, f"no {ANCHOR} in the TTGIR, the descriptor load was not relaid out"
        assert block_loads > 0, "the relaid-out descriptor load did not lower to 2D block reads"
    else:
        assert ANCHOR not in ttgir, f"{ANCHOR} in the TTGIR of a load at column {c0}"
        assert block_loads == 0, f"{block_loads} 2D block load(s) at column {c0}, expected a gather"


def test_relayout_stencil_negative_tap(device, monkeypatch, tmp_path):
    # G1 cannot prove the left tap's column, program_id * BC - 1, non-negative, and it is -1 in program 0. With
    # per-load gates that tap kept its 2D-block tag, so RemoveLayoutConversions could move it into the relaid-out
    # layout of the other taps, making it a 2D read at column -1 (#8479). Function-wide gates relay out no tap.
    BR, BC, GRID = 16, 64, 2
    NCOLS, PITCH = GRID * BC, GRID * BC + 32
    x, vals = _misaligned_tensor(BR, NCOLS, PITCH, device)
    padded = torch.nn.functional.pad(vals, (1, 1))  # columns -1 and NCOLS are outside the tensor: zero
    ref = (padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]).sum(0)
    for knob in (True, False):
        state = "on" if knob else "off"
        (tmp_path / state).mkdir()
        _set_compile_env(monkeypatch, tmp_path / state, knob)
        out = torch.full((NCOLS, ), float("nan"), device=device)
        k = triton.jit(_stencil_kernel_fn)[(GRID, )](out, x, NCOLS, PITCH, BR=BR, BC=BC)

        # Knob on and off must both match the CPU reference exactly.
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0, msg=lambda m: f"knob {state}: {m}")
        assert ANCHOR not in k.asm["ttgir"], f"knob {state}: {ANCHOR} in the TTGIR"
        block_loads = _count_2d_block_loads(k.asm["llir"])
        assert block_loads == 0, f"knob {state}: {block_loads} 2D block load(s), a stencil tap became a 2D read"


L1X16 = dict(num_warps=1, warp_size=16)
# (input dtype, output dtype, launch, relaid out with the knob on) for _row_sum_kernel_fn.
ROW_SUM_CASES = [
    # Refused: a narrow output at 16 lanes.
    pytest.param(torch.float16, torch.float16, L1X16, False, id="f16-f16-16lanes"),
    pytest.param(torch.int8, torch.int8, L1X16, False, id="i8-i8-16lanes"),
    pytest.param(torch.float32, torch.bfloat16, L1X16, False, id="f32-bf16-16lanes"),
    # Controls: the same inputs with a 32-bit output at 16 lanes, and the narrow outputs at 32 lanes.
    pytest.param(torch.float16, torch.float32, L1X16, True, id="f16-f32-16lanes"),
    pytest.param(torch.int8, torch.int32, L1X16, True, id="i8-i32-16lanes"),
    pytest.param(torch.float32, torch.float32, L1X16, True, id="f32-f32-16lanes"),
    pytest.param(torch.float16, torch.float16, {}, True, id="f16-f16-default"),
    pytest.param(torch.int8, torch.int8, {}, True, id="i8-i8-default"),
    pytest.param(torch.float32, torch.bfloat16, {}, True, id="f32-bf16-default"),
]


@pytest.mark.parametrize("x_dtype, out_dtype, launch, relaid_out", ROW_SUM_CASES)
def test_relayout_reduce_output_dtype(x_dtype, out_dtype, launch, relaid_out, monkeypatch, tmp_path):
    # With 16 lanes, every value computed from a reduction must be f32 or i32 (G2). Compile only: at 16 lanes, f16
    # sums come out wrong or lose the device even with the knob off (an IGC miscompile), and relaid-out int8 sums
    # came out wrong.
    _set_compile_env(monkeypatch, tmp_path, knob=True)
    acc = tl.float32 if x_dtype.is_floating_point else tl.int32
    k = triton.jit(_row_sum_kernel_fn).warmup(out_dtype, x_dtype, R=16, C=128, ACC=acc, grid=(1, ), **launch)

    ttgir = k.asm["ttgir"]
    block_loads = _count_2d_block_loads(k.asm["llir"])
    if relaid_out:
        assert ANCHOR in ttgir, f"no {ANCHOR} in the TTGIR, the descriptor load was not relaid out"
        assert block_loads > 0, "the relaid-out descriptor load did not lower to 2D block reads"
    else:
        assert ANCHOR not in ttgir, f"{ANCHOR} in the TTGIR of a 16-lane kernel with a {out_dtype} reduction"
        assert block_loads == 0, f"{block_loads} 2D block load(s) in a refused relayout, expected the gather path"


@pytest.mark.parametrize("pad", ["zero", "nan"])
@pytest.mark.parametrize("nseg", [15, 16], ids=lambda n: f"nseg{n}")
def test_relayout_padding_option(nseg, pad, device, monkeypatch, tmp_path):
    # Relaid out, a NaN-padded descriptor one element wide read 0.0 instead of NaN at 16 lanes, so only zero padding
    # is relaid out (G3), whatever the lane count. This runs at the default launch.
    _set_compile_env(monkeypatch, tmp_path, knob=True)
    kernel = triton.jit(_reduce_padded_kernel_fn)
    S, D, DP = 16, 96, 128
    vals = torch.randint(-2, 3, (nseg, D), generator=torch.Generator().manual_seed(17)).float()
    # POISON outside the descriptor's [nseg, D] shape, so padding read from memory shows up in the sums.
    x = torch.full((S, DP), POISON, device=device)
    x[:nseg, :D] = vals.to(device)
    out = torch.full((DP, ), -99.0, device=device)
    k = kernel[(1, )](out, x, nseg, D=D, DP=DP, S=S, PAD=pad)

    tile = torch.full((S, DP), float("nan") if pad == "nan" else 0.0)
    tile[:nseg, :D] = vals
    torch.testing.assert_close(out.cpu(), tile.sum(0), rtol=0, atol=0, equal_nan=True)
    ttgir = k.asm["ttgir"]
    block_loads = _count_2d_block_loads(k.asm["llir"])
    if pad == "zero":
        assert ANCHOR in ttgir, f"no {ANCHOR} in the TTGIR, the zero-padded descriptor load was not relaid out"
        assert block_loads > 0, "the relaid-out descriptor load did not lower to 2D block reads"
    else:
        assert ANCHOR not in ttgir, f"{ANCHOR} in the TTGIR of a NaN-padded descriptor load"
        assert block_loads == 0, f"{block_loads} 2D block load(s) for a NaN-padded descriptor, expected a gather"
