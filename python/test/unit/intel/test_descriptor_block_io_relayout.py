"""End-to-end tests for the opt-in descriptor-load block-IO relayout (issue #8458).

With TRITON_INTEL_DESCRIPTOR_BLOCK_IO_RELAYOUT=1, MaterializeBlockPointer gives a row-major
`tt.descriptor_load` that feeds no dot a layout LowerTo2DBlockLoad accepts, and marks it
`ttig.block_io_anchor` so RemoveLayoutConversions keeps that layout. The load then lowers to 2D block
reads instead of gathers.

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
# The default launch is 4 warps of 32 lanes; "tuned" is the issue's hand-tuned launch.
LAUNCHES = [
    pytest.param({}, id="default"),
    pytest.param(dict(num_warps=1, warp_size=16, grf_mode="256"), id="tuned"),
]


def _set_compile_env(monkeypatch, tmp_path, knob):
    """Compile into an empty on-disk cache, with the relayout knob set (on) or unset (off)."""
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    if knob:
        monkeypatch.setenv(RELAYOUT_ENV, "1")
    else:
        monkeypatch.delenv(RELAYOUT_ENV, raising=False)


# The kernels below are plain functions, not @triton.jit: each test wraps one in a fresh JITFunction.


def _reduce_kernel_fn(out_ptr, x_ptr, m_ptr, nseg, H, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    """The issue kernel's descriptor path: weight the rows of an [S, DP] fp32 tile and sum them."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    seg = tl.arange(0, S)
    dim = tl.arange(0, DP)
    smask = seg < nseg
    m = tl.load(m_ptr + (t * H + h) * S + seg, mask=smask, other=float("-inf"))
    w = tl.exp(m - tl.max(m))
    base = x_ptr + (t * H + h) * (S * DP)
    x = tl.make_tensor_descriptor(base=base, shape=(nseg, D), strides=(DP, 1), block_shape=(S, DP)).load([0, 0])
    tl.store(out_ptr + (t * H + h) * D + dim, tl.sum(x * w[:, None], axis=0).to(tl.bfloat16), mask=dim < D)


def _copy_padded_kernel_fn(out_ptr, x_ptr, nseg, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr):
    """Store the whole [S, DP] descriptor tile with no mask, so its padding is observable."""
    rows = tl.arange(0, S)
    cols = tl.arange(0, DP)
    desc = tl.make_tensor_descriptor(x_ptr, shape=(nseg, D), strides=(DP, 1), block_shape=(S, DP))
    tl.store(out_ptr + rows[:, None] * DP + cols[None, :], desc.load([0, 0]))


def _loop_accumulate_kernel_fn(out_ptr, x_ptr, M, N: tl.constexpr, BLOCK_M: tl.constexpr):
    """Sum an [M, N] fp32 tensor over its rows, one [BLOCK_M, N] descriptor load per loop iteration."""
    desc = tl.make_tensor_descriptor(x_ptr, shape=(M, N), strides=(N, 1), block_shape=(BLOCK_M, N))
    acc = tl.zeros((N, ), dtype=tl.float32)
    for i in range(M // BLOCK_M):
        x = desc.load([i * BLOCK_M, 0])
        acc += tl.sum(x, axis=0)
    tl.store(out_ptr + tl.arange(0, N), acc)


@pytest.mark.parametrize("knob", [True, False], ids=["on", "off"])
@pytest.mark.parametrize("launch", LAUNCHES)
@pytest.mark.parametrize("D", [64, 96, 128, 256], ids=lambda d: f"D{d}")
def test_relayout_emits_2d_block_load(D, launch, knob, device, monkeypatch, tmp_path):
    _set_compile_env(monkeypatch, tmp_path, knob)
    kernel = triton.jit(_reduce_kernel_fn)
    S, H, T = 16, 32, 1
    DP = triton.next_power_of_2(D)
    x = torch.randn(T, H, S, DP, device=device)
    m = torch.randn(T, H, S, device=device)
    out = torch.empty(T, H, D, device=device, dtype=torch.bfloat16)
    k = kernel[(T, H)](out, x, m, S, H, D=D, DP=DP, S=S, **launch)
    torch.xpu.synchronize()

    # The TTGIR is captured before LowerTo2DBlockLoad runs (in make_llir), so the anchor is still there.
    ttgir = k.asm["ttgir"]
    block_loads = _count_2d_block_loads(k.asm["llir"])
    if knob:
        assert ANCHOR in ttgir, f"knob on: no {ANCHOR} in the TTGIR, the descriptor load was not relaid out"
        assert block_loads > 0, "knob on: the relaid-out descriptor load did not lower to 2D block reads"
    else:
        # Today's gather path for this kernel.
        assert ANCHOR not in ttgir, f"knob off: {ANCHOR} in the TTGIR"
        assert block_loads == 0, f"knob off: {block_loads} 2D block load(s), expected the gather path"


@pytest.mark.parametrize("launch", LAUNCHES)
@pytest.mark.parametrize("D", [64, 96, 128, 256], ids=lambda d: f"D{d}")
@pytest.mark.parametrize("nseg", [1, 5, 15, 16], ids=lambda n: f"nseg{n}")
def test_relayout_correctness(nseg, D, launch, device, monkeypatch, tmp_path):
    _set_compile_env(monkeypatch, tmp_path, knob=True)
    kernel = triton.jit(_reduce_kernel_fn)
    S, H, T = 16, 32, 2
    DP = triton.next_power_of_2(D)
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(T, H, S, DP, generator=generator).to(device)
    # Rows >= nseg are outside the descriptor: they must be zero-filled, not read from memory.
    x[:, :, nseg:, :] = float("nan")
    m = torch.randn(T, H, S, generator=generator).to(device)
    out = torch.empty(T, H, D, device=device, dtype=torch.bfloat16)
    kernel[(T, H)](out, x, m, nseg, H, D=D, DP=DP, S=S, **launch)

    # float64 reference on the CPU: not every XPU supports fp64.
    x64 = x.cpu().to(torch.float64)
    m64 = m.cpu().to(torch.float64)[..., :nseg]
    w = torch.exp(m64 - m64.amax(-1, keepdim=True))
    ref = (x64[:, :, :nseg, :D] * w[..., None]).sum(-2).to(torch.float32).to(device)
    # The bf16 output measured <= 2.65e-2 absolute error at D=256.
    torch.testing.assert_close(out.float(), ref.float(), atol=4e-2, rtol=1.6e-2)


@pytest.mark.parametrize("D", [96, 128], ids=lambda d: f"D{d}")
@pytest.mark.parametrize("nseg", [1, 15, 16], ids=lambda n: f"nseg{n}")
def test_relayout_copy_padded(nseg, D, device, monkeypatch, tmp_path):
    _set_compile_env(monkeypatch, tmp_path, knob=True)
    kernel = triton.jit(_copy_padded_kernel_fn)
    S = 16
    DP = triton.next_power_of_2(D)
    generator = torch.Generator().manual_seed(17)
    # NaN outside the descriptor's [nseg, D] shape: any padding read from memory shows up as NaN.
    x = torch.full((S, DP), float("nan"), device=device)
    x[:nseg, :D] = torch.randn(nseg, D, generator=generator).to(device)
    # NaN in the output as well, so the zero checks prove the kernel wrote the padding.
    out = torch.full((S, DP), float("nan"), device=device)
    k = kernel[(1, )](out, x, nseg, D=D, DP=DP, S=S)

    assert torch.equal(out[:nseg, :D], x[:nseg, :D])
    assert (out[nseg:, :] == 0).all() and (out[:, D:] == 0).all()
    # The padding checked above must come from the 2D block read.
    assert ANCHOR in k.asm["ttgir"], f"no {ANCHOR} in the TTGIR, the descriptor load was not relaid out"
    assert _count_2d_block_loads(k.asm["llir"]) > 0, "the descriptor load did not lower to 2D block reads"


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
    torch.testing.assert_close(out, ref, atol=4e-2, rtol=1.6e-2)
    assert ANCHOR in ttgir, f"no {ANCHOR} in the TTGIR, the descriptor load in the loop was not relaid out"
    assert _count_2d_block_loads(k.asm["llir"]) > 0, "the descriptor load in the loop did not lower to 2D block reads"
