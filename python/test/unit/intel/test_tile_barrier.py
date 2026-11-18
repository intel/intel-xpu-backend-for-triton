"""Tests for the `tile_barrier` XPUOptions field (TRITON_INTEL_PIPELINE_TILE_BARRIER), which re-aligns the warps of a
persistent kernel at every tile."""

import pytest
import torch

import triton
import triton.language as tl
from triton.backends.intel.compiler import XPUBackend
from triton.compiler.compiler import make_backend


@triton.jit
def _persistent_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr,
                       BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(A, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(B, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(C, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    n_tiles_n = tl.cdiv(N, BLOCK_N)
    for tile in range(tl.program_id(0), tl.cdiv(M, BLOCK_M) * n_tiles_n, tl.num_programs(0)):
        off_m, off_n = (tile // n_tiles_n) * BLOCK_M, (tile % n_tiles_n) * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            acc = tl.dot(a_desc.load([off_m, k]), b_desc.load([k, off_n]), acc)
        c_desc.store([off_m, off_n], acc.to(tl.bfloat16))


@pytest.mark.parametrize("tile_barrier", [False, True])
def test_tile_barrier_reaches_pipeliner(tile_barrier, device):
    # `use_barrier` and `tile_barrier` are adjacent bools in the add_pipeline call; a swap fails the True case.
    M = N = K = 256
    a = torch.empty((M, K), dtype=torch.bfloat16, device=device)
    b = torch.empty((K, N), dtype=torch.bfloat16, device=device)
    c = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    kernel = _persistent_kernel.warmup(a, b, c, M=M, N=N, K=K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8,
                                       num_stages=3, tile_barrier=tile_barrier, grid=(2, ))
    assert ("ttg.barrier none" in kernel.asm["ttgir"]) == tile_barrier


@pytest.mark.parametrize("env, option, expected", [
    (None, None, True),
    ("0", None, False),
    ("1", False, False),
    ("0", True, True),
])
def test_tile_barrier_precedence(env, option, expected, monkeypatch):
    # The option wins over TRITON_INTEL_PIPELINE_TILE_BARRIER, which defaults to on.
    if env is None:
        monkeypatch.delenv("TRITON_INTEL_PIPELINE_TILE_BARRIER", raising=False)
    else:
        monkeypatch.setenv("TRITON_INTEL_PIPELINE_TILE_BARRIER", env)
    backend = make_backend(triton.runtime.driver.active.get_current_target())
    assert isinstance(backend, XPUBackend)
    opts = {} if option is None else {"tile_barrier": option}
    assert backend.parse_options(opts).tile_barrier is expected
