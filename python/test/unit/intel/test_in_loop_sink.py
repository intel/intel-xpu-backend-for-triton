"""Tests for the `in_loop_sink` XPUOptions field (TRITON_INTEL_IN_LOOP_SINK), which
controls ReduceVariableLiveness's sinking of in-loop dot operand loads."""

import re

import pytest
import torch

import triton
import triton.language as tl
from triton.backends.intel.compiler import XPUBackend
from triton.compiler.compiler import make_backend


@triton.jit
def _attn_like_kernel(Q, K, V, O, N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr):
    desc_q = tl.make_tensor_descriptor(Q, shape=[N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])
    desc_k = tl.make_tensor_descriptor(K, shape=[N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_N, HEAD_DIM])
    desc_v = tl.make_tensor_descriptor(V, shape=[N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = tl.make_tensor_descriptor(O, shape=[N_CTX, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])
    start_m = tl.program_id(0) * BLOCK_M
    q = desc_q.load([start_m, 0])
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    for start_n in tl.range(0, N_CTX, BLOCK_N):
        k = desc_k.load([start_n, 0]).T
        v = desc_v.load([start_n, 0])
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        acc = acc * tl.math.exp2(m_i - m_ij)[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij
    desc_o.store([start_m, 0], acc.to(tl.float16))


@pytest.mark.skipif(not triton.runtime.driver.active.get_current_target().arch.get('has_2d_block_io', False),
                    reason="RVL only runs with 2D block IO")
@pytest.mark.parametrize("in_loop_sink", [True, False])
def test_in_loop_sink_reaches_rvl(in_loop_sink, device):
    x = torch.empty((1024, 64), dtype=torch.float16, device=device)
    kernel = _attn_like_kernel.warmup(x, x, x, x, N_CTX=1024, HEAD_DIM=64, BLOCK_M=128, BLOCK_N=64, num_warps=8,
                                      grid=(1, ), in_loop_sink=in_loop_sink)
    body = kernel.asm["ttgir"].split("scf.for", 1)[1]
    v_load = re.search(r"tt\.descriptor_load %desc_v\b", body).start()
    first_dot = body.index("tt.dot")
    # The sink moves V's load from the top of the body to just before its dot.
    assert (v_load > first_dot) == in_loop_sink


@pytest.mark.parametrize("env, option, arch_default, expected", [
    (None, None, True, True),
    (None, None, False, False),
    ("0", None, True, False),
    ("1", None, False, True),
    ("0", True, False, True),
    ("1", False, True, False),
])
def test_in_loop_sink_precedence(env, option, arch_default, expected, monkeypatch):
    # The option wins over TRITON_INTEL_IN_LOOP_SINK, which wins over the arch default.
    if env is None:
        monkeypatch.delenv("TRITON_INTEL_IN_LOOP_SINK", raising=False)
    else:
        monkeypatch.setenv("TRITON_INTEL_IN_LOOP_SINK", env)
    backend = make_backend(triton.runtime.driver.active.get_current_target())
    assert isinstance(backend, XPUBackend)
    monkeypatch.setattr(type(backend), "default_in_loop_sink", lambda self: arch_default)
    opts = {} if option is None else {"in_loop_sink": option}
    assert backend.parse_options(opts).in_loop_sink is expected
