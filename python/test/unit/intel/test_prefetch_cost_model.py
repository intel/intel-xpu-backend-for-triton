"""Tests for the opt-in `prefetch_cost_model` XPUOptions field (TRITON_INTEL_ENABLE_PREFETCH_COST_MODEL),
which lets the pipeliner drop the prefetch of a gapped dot operand."""

import re

import pytest
import torch

import triton
import triton.language as tl
from triton.backends.intel.compiler import XPUBackend
from triton.compiler.compiler import make_backend


@triton.jit
def _split_k_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(A, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(B, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N])
    pid_m = tl.program_id(0) // tl.cdiv(N, BLOCK_N)
    pid_n = tl.program_id(0) % tl.cdiv(N, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    off_k = tl.program_id(1) * BLOCK_K
    for _ in range(0, K, BLOCK_K * SPLIT_K):  # A's walk skips SPLIT_K - 1 chunks per iteration
        acc = tl.dot(a_desc.load([pid_m * BLOCK_M, off_k]), b_desc.load([off_k, pid_n * BLOCK_N]), acc)
        off_k += BLOCK_K * SPLIT_K
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.atomic_add(C + rm[:, None] * N + rn[None, :], acc, sem="relaxed")


def _prefetch_count(ttgir, arg):
    desc = re.search(rf"(%\w+) = tt\.make_tensor_descriptor %{arg}\b", ttgir)
    assert desc, f"{arg}'s make_tensor_descriptor not found in the TTGIR"
    return len(re.findall(rf"ttig\.descriptor_prefetch {re.escape(desc.group(1))}\[", ttgir))


@pytest.mark.skipif(triton.runtime.driver.active.get_current_target().arch.get("arch") != "bmg",
                    reason="the prefetch cost model has constants for BMG only")
@pytest.mark.parametrize("enabled", [False, True])
def test_prefetch_cost_model_drops_a(enabled, device):
    M, N, K = 256, 256, 28672  # 224 iterations at SPLIT_K=4, long enough for the model to drop A
    a = torch.empty((M, K), dtype=torch.bfloat16, device=device)
    b = torch.empty((K, N), dtype=torch.bfloat16, device=device)
    c = torch.empty((M, N), dtype=torch.float32, device=device)
    kernel = _split_k_kernel.warmup(a, b, c, M=M, N=N, K=K, BLOCK_M=256, BLOCK_N=256, BLOCK_K=32, SPLIT_K=4,
                                    num_warps=32, num_stages=4, grf_mode="256", prefetch_cost_model=enabled,
                                    grid=(1, 4))
    ttgir = kernel.asm["ttgir"]
    assert _prefetch_count(ttgir, "B") > 0
    assert (_prefetch_count(ttgir, "A") == 0) == enabled


@pytest.mark.parametrize("env, option, expected", [
    (None, None, False),
    ("1", None, True),
    ("0", None, False),
    ("1", False, False),
    ("0", True, True),
])
def test_prefetch_cost_model_precedence(env, option, expected, monkeypatch):
    # The option wins over TRITON_INTEL_ENABLE_PREFETCH_COST_MODEL, which defaults to off.
    if env is None:
        monkeypatch.delenv("TRITON_INTEL_ENABLE_PREFETCH_COST_MODEL", raising=False)
    else:
        monkeypatch.setenv("TRITON_INTEL_ENABLE_PREFETCH_COST_MODEL", env)
    backend = make_backend(triton.runtime.driver.active.get_current_target())
    assert isinstance(backend, XPUBackend)
    opts = {} if option is None else {"prefetch_cost_model": option}
    assert backend.parse_options(opts).prefetch_cost_model is expected
