"""Gate G1 of the store boundary clamp, and its rows: a masked pointer store whose column bound is not a multiple of
the 16-column message, or whose row bound leaves a row constancy of 1, is lowered to 2D block stores whose surface
ends at the bounds, shifted left when narrower than 64 bytes."""

import os
import re
import subprocess
import sys

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import default_alloc_fn, is_xpu

BM, BN, BK, K = 64, 128, 32, 64
GUARD = 256  # sentinel bytes before the output: more than a 64-byte shift plus 63 bytes of alignment compensation
SENTINEL = -1000.0  # a GEMM of {-1, 0, 1} entries over K = 64 cannot produce it
ARMS = {"clamp": 0, "kill": 1, "assert": 2}
KILL_ENV, ASSERT_ENV = "TRITON_INTEL_DISABLE_STORE_BOUNDARY_CLAMP", "TRITON_INTEL_2DBLOCK_ASSERT"


@triton.jit
def _gemm_store(a_ptr, b_ptr, c_ptr, M_RT, M: tl.constexpr, N: tl.constexpr, NB: tl.constexpr, K: tl.constexpr,
                STRIDE: tl.constexpr, OFF: tl.constexpr, ARM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr):
    # ARM only makes each arm its own specialization: the in-process kernel cache ignores the environment.
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, NB], strides=[NB, 1], block_shape=[BK, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc = tl.dot(a_desc.load([pid_m * BM, k]), b_desc.load([k, pid_n * BN]), acc)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    # OFF is added here, not by an offset view, which would lose the base's 16-byte alignment hint. A non-None M_RT
    # replaces M as the row bound at runtime, chosen inline: an assignment would turn a constexpr M into a tensor.
    tl.store(c_ptr + OFF + rm[:, None] * STRIDE + rn[None, :], acc.to(c_ptr.dtype.element_ty),
             mask=(rm[:, None] < (M if M_RT is None else M_RT)) & (rn[None, :] < N))


# (store dtype, N, M, output start residue mod 64 in bytes, STRIDE, extra programs per grid dim, sub-group size).
# The last column tile has N % 128 valid columns; the origin shifts when they span fewer than 64 bytes. C6's minimum
# STRIDE is 160 for 16-bit and 144 for f32 stores; only N % 16 == 0 (N = 160) gets block stores today.
CASES = [
    ("float16", 158, 128, 0, 160, 0, 16),  # 30 valid columns: 60 bytes, shifted
    ("float16", 136, 128, 0, 160, 0, 16),  # 8 columns, shifted
    ("float16", 130, 128, 0, 160, 0, 16),  # 2 columns, shifted
    ("float16", 30, 128, 0, 160, 0, 16),  # first tile: row 0's shifted origin lies in the guard before the output
    ("float32", 136, 128, 0, 144, 0, 16),  # 8 columns: 32 bytes, shifted
    ("float32", 130, 128, 0, 144, 0, 16),  # 2 columns, shifted
    ("float16", 248, 128, 0, 248, 0, 16),  # 120 columns: 240 bytes, no shift; STRIDE >= N rules out C6's minimum
    ("float16", 160, 128, 4, 160, 0, 16),  # 32 columns: works today
    ("float16", 8, 72, 4, 264, 1, 16),
    ("float16", 2, 128, 60, 160, 0, 32),
    ("float16", 136, 72, 60, 264, 1, 16),
    ("float16", 130, 128, 32, 160, 0, 32),
    ("bfloat16", 248, 72, 60, 264, 1, 32),
    ("bfloat16", 160, 128, 0, 264, 0, 32),
    ("bfloat16", 158, 72, 4, 264, 0, 16),
    ("bfloat16", 136, 128, 32, 160, 0, 32),
    ("bfloat16", 130, 128, 60, 264, 1, 16),
    ("bfloat16", 30, 128, 32, 264, 0, 32),
    ("bfloat16", 8, 128, 0, 160, 0, 16),
    ("bfloat16", 2, 72, 4, 160, 1, 16),
    ("float32", 248, 128, 4, 260, 0, 16),
    ("float32", 160, 72, 32, 260, 1, 16),
    ("float32", 158, 128, 60, 260, 0, 32),  # 30 columns of f32: 120 bytes, no shift
    ("float32", 8, 128, 60, 144, 0, 16),
    ("float32", 2, 128, 4, 260, 0, 32),
    ("float32", 130, 72, 32, 144, 1, 32),
    # Rows: an odd M leaves M % 64 valid rows in the last row tile at a row constancy of 1, with no block store today.
    # "runtime_m" passes M as M_RT, whose divisibility Triton specializes for 16 only.
    ("float16", 128, 129, 0, 160, 0, 16),
    ("bfloat16", 128, 135, 4, 264, 0, 32),
    ("float32", 128, 137, 60, 144, 0, 16),
    ("bfloat16", 128, 135, 32, 160, 1, 16),  # H = 0 in the extra row of programs
    ("float16", 136, 137, 0, 160, 0, 16),  # a ragged row and column edge together
    ("float16", 128, 135, 4, 160, 0, 16, "runtime_m"),
]


def _lowering(kernel):
    """Whether the only store is stamped `ttig.block_io`, emits a 2D block store, and emits a scalar global store."""
    ttgir, llir = kernel.asm["ttgir"], kernel.asm["llir"]
    stamped = re.search(r'tt\.store [^\n]*ttig\.block_io = "row_major"', ttgir) is not None
    block = "spirv_Subgroup2DBlockStoreINTEL" in llir or "GenISA.LSC2DBlockWrite" in llir
    # Without predicated I/O (LTS) the scalar fallback is a branch-guarded plain store.
    scalar = "PredicatedStoreINTEL" in llir or re.search(r"^\s*store [^\n]*ptr addrspace\(1\)", llir, re.M) is not None
    return stamped, block, scalar


def _stamped_today(case):
    """Whether MaterializeBlockPointer stamps `case` without the clamp: a mask constancy of 2 or more along both dims."""
    return case[1] % 2 == 0 and case[2] % (16 if "runtime_m" in case else 2) == 0


def _launch(device, case, arm):
    """Run `case` into a sentinel-filled buffer, check every element of it, and return the compiled kernel."""
    dtype, N, M, res, stride, extra, sg = case[:7]
    triton.set_allocator(default_alloc_fn)  # the assertion arm's child process has no fixture
    out_dtype = getattr(torch, dtype)
    in_dtype = torch.float16 if out_dtype == torch.float32 else out_dtype
    eb = out_dtype.itemsize
    off = (GUARD + res) // eb
    gm, gn = triton.cdiv(M, BM) + extra, triton.cdiv(N, BN) + extra
    nb = triton.cdiv(N, BN) * BN  # B is zero-padded to whole tiles, so its descriptor is never narrow
    gen = torch.Generator().manual_seed(17)
    a = torch.randint(-1, 2, (M, K), generator=gen).to(in_dtype)
    b = torch.zeros((K, nb), dtype=in_dtype)
    b[:, :N] = torch.randint(-1, 2, (K, N), generator=gen).to(in_dtype)
    # Every address a program can form lies inside `total`: a stray write fails the sentinel check, it cannot fault.
    total = off + (gm * BM + 1) * stride + gn * BN
    raw = torch.full((total + 64 // eb, ), SENTINEL, dtype=out_dtype, device=device)
    c = raw[(-raw.data_ptr() % 64) // eb:][:total]
    assert c.data_ptr() % 64 == 0
    args = (a.to(device), b.to(device), c)
    kw = dict(M_RT=M if "runtime_m" in case else None, M=M, N=N, NB=nb, K=K, STRIDE=stride, OFF=off, ARM=ARMS[arm],
              BM=BM, BN=BN, BK=BK, num_warps=16, warp_size=sg, sub_32_dpas=sg == 32)
    kernel = _gemm_store.warmup(*args, grid=(gm, gn), **kw)
    assert kernel.metadata.threads_per_warp == sg
    if arm == "assert" and (N % 16 or not _stamped_today(case)):
        llir = kernel.asm["llir"]
        assert re.search(r"call [^\n]*@__devicelib_assert_fail\(", llir), "no 2D block assertion in the LLVM IR"
        assert "(base width) should be" in llir
        # The A/B block loads emit the same assertions; the store's own exist only if it is a block store.
        assert _lowering(kernel)[1], "no 2D block store: the assertions found belong to the loads alone"
    _gemm_store[(gm, gn)](*args, **kw)
    torch.xpu.current_stream().synchronize()  # torch.xpu.synchronize() does not report a device assert

    expect = torch.full((total, ), SENTINEL, dtype=out_dtype)
    inside = torch.zeros(total, dtype=torch.bool)
    expect[off:off + M * stride].view(M, stride)[:, :N] = (a.float() @ b[:, :N].float()).to(out_dtype)
    inside[off:off + M * stride].view(M, stride)[:, :N] = True
    bad = c.cpu() != expect
    assert not bad.any(), (f"{int((bad & ~inside).sum())} guard elements written, "
                           f"{int((bad & inside).sum())} valid elements wrong")
    return kernel


def _run_case(device, case, arm, triton_file=None):
    here = os.path.realpath(triton.__file__)
    assert triton_file in (None, here), f"the child imported a different triton ({here}); it would test another build"
    kernel = _launch(device, case, arm)
    stamped, block, scalar = _lowering(kernel)
    # The kill switch restores today's lowering: stamped as `_stamped_today` says, with block stores only if the column
    # constancy is also 16 or more. Unstamped, a clamp or assert arm tests nothing.
    today = _stamped_today(case)
    assert stamped == (arm != "kill" or today), f"ttig.block_io stamped {stamped} in the {arm} arm"
    expect_block = arm != "kill" or (today and case[1] % 16 == 0)
    assert (block, scalar) == (expect_block, not expect_block), f"block store {block}, scalar store {scalar}"


# Puts the parent's triton first on sys.path and loads this file by path, so `_run_case` tests the parent's build.
_CHILD = """import ast, importlib.util, os, sys
test_file, triton_file, device, case, arm = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.dirname(triton_file)))
spec = importlib.util.spec_from_file_location("store_boundary_clamp_child", test_file)
test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test)
test._run_case(device, ast.literal_eval(case), arm, triton_file)
"""


def _run_in_child(device, case, arm, env):
    """Run `_run_case` in a fresh interpreter started with `env`, failing with its output unless it exits cleanly."""
    paths = (os.path.realpath(__file__), os.path.realpath(triton.__file__))
    cmd = [sys.executable, "-c", _CHILD, *paths, device, repr(case), arm]
    try:
        proc = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as exc:  # its output stays bytes despite text=True
        out = [s.decode(errors="replace") if isinstance(s, bytes) else s or "" for s in (exc.stdout, exc.stderr)]
        pytest.fail("the child did not exit within 60 s\n" + "\n".join(out))
    assert proc.returncode == 0, f"the child exited with {proc.returncode}\n{proc.stdout}\n{proc.stderr}"


def _set_env(monkeypatch, tmp_path, arm):
    env = {"TRITON_CACHE_DIR": str(tmp_path), KILL_ENV: str(int(arm == "kill")), ASSERT_ENV: str(int(arm == "assert"))}
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return env


def _skip_unsupported_sub_group(sg):
    sizes = triton.runtime.driver.active.utils.get_device_properties(torch.xpu.current_device())["sub_group_sizes"]
    if sg not in sizes:
        pytest.skip(f"sub-group size {sg} is not in the device's sub-group sizes {sizes}")


@pytest.mark.xfail(not torch.xpu.get_device_capability()['has_subgroup_2d_block_io'],
                   reason="Block stores not supported on this architecture", run=False)
@pytest.mark.skipif(not is_xpu(), reason="Block store tests are specific to the XPU backend")
@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(map(str, c)))
def test_store_boundary_clamp(case, arm, device, tmp_path, monkeypatch, with_allocator):
    # UNSURE: that a 32-lane DPAS store gets 2D block stores at all; no lit test covers it.
    _skip_unsupported_sub_group(case[6])
    env = _set_env(monkeypatch, tmp_path, arm)
    if arm != "assert":
        _run_case(device, case, arm)
        return
    # A failed device assert aborts the process, so this arm runs in a fresh interpreter started with the arm's
    # environment.
    _run_in_child(device, case, arm, env)


# Not contract validation: each keeps the scalar path. Pitches 304 and 256 (rows only) fail C6 only with alignment
# compensation.
@pytest.mark.xfail(not torch.xpu.get_device_capability()['has_subgroup_2d_block_io'],
                   reason="Block stores not supported on this architecture", run=False)
@pytest.mark.skipif(not is_xpu(), reason="Block store tests are specific to the XPU backend")
@pytest.mark.parametrize("N, M, stride, res", [(136, 128, 152, 0), (120, 128, 120, 4), (131, 128, 160, 0),
                                               (128, 135, 128, 4)])
def test_store_boundary_clamp_declines(N, M, stride, res, device, tmp_path, monkeypatch, with_allocator):
    _set_env(monkeypatch, tmp_path, "clamp")
    kernel = _launch(device, ("float16", N, M, res, stride, 0, 16), "clamp")
    stamped, block, scalar = _lowering(kernel)
    align = re.search(r"ttig\.2d_block_io_base_alignment = (\d+)", kernel.asm["ttgir"])
    if align and int(align.group(1)) <= 4 and N % 2 == 0 and stride * 2 >= 256:
        pytest.skip("no alignment compensation on this device, so C6 accepts this pitch")
    # An odd N or M has a constancy of 1, which MaterializeBlockPointer stamps only for a clamp.
    assert (stamped, block, scalar) == (N % 2 == 0 and M % 2 == 0, False, True)
