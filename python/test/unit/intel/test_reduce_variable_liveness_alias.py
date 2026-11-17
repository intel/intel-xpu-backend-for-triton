"""ReduceVariableLiveness must not move a dot operand loaded before a loop past a write to
the memory it reads: the dot has to see the tile as it was when the program loaded it.

Each kernel loads one operand tile once, before the loop, then writes that tile's memory
(in the loop, or between the load and the loop) through a second pointer argument the
caller binds to the same buffer. The `no_write` variants are the same
kernels without the write, where the pass may still move the load.

With num_warps > 1 the warp that writes an element need not be one that read it before
the loop, and Triton orders global accesses of different warps only at a barrier, so the
num_warps > 1 write kernels put `tl.debug_barrier()` after the pre-loop load and after each write: the
program is then race free and the torch reference is exact. The pass treats that barrier
as a write too, so only the num_warps == 1 variants, which need no barrier, test that the
store alone keeps the load in place.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu

# The writes store a [BM, BN] tile over the operand they target, so BN == BK (A tile)
# and K == BM (the whole [K, N] B buffer).
M, N, K = 128, 64, 128
BM, BN, BK = 128, 64, 64

PATTERNS = [
    "in_loop_other_arg",  # A, written in the loop through the aliasing second argument
    "before_loop",  # A, written between the load and the loop
    "in_loop_use_after",  # A, written in the loop, also read by a dot after the loop
    "operand_b",  # B, written in the loop through the aliasing second argument
]


@triton.jit
def _pre_loop_operand_kernel(a_ptr, b_ptr, w_ptr, out_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                             BK: tl.constexpr, PATTERN: tl.constexpr, WRITE: tl.constexpr, SYNC: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BK, BN])
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    if PATTERN == "operand_b":
        w_desc = tl.make_tensor_descriptor(w_ptr, shape=[K, N], strides=[N, 1], block_shape=[BM, BN])
        b = b_desc.load([0, 0])
        if SYNC:
            tl.debug_barrier()
        for k in range(0, K, BK):
            acc = tl.dot(a_desc.load([0, k]), b, acc)
            if WRITE:
                w_desc.store([0, 0], acc.to(w_ptr.dtype.element_ty))
                if SYNC:
                    tl.debug_barrier()
    else:
        w_desc = tl.make_tensor_descriptor(w_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BN])
        a = a_desc.load([0, 0])
        if SYNC:
            tl.debug_barrier()
        if WRITE and PATTERN == "before_loop":
            w_desc.store([0, 0], tl.full([BM, BN], 1.0, w_ptr.dtype.element_ty))
            if SYNC:
                tl.debug_barrier()
        for k in range(0, K, BK):
            acc = tl.dot(a, b_desc.load([k, 0]), acc)
            if WRITE and PATTERN != "before_loop":
                w_desc.store([0, 0], acc.to(w_ptr.dtype.element_ty))
                if SYNC:
                    tl.debug_barrier()
        if PATTERN == "in_loop_use_after":
            acc = tl.dot(a, b_desc.load([0, 0]), acc)
    out_desc.store([0, 0], acc)


def _reference(pattern: str, write: bool, a: torch.Tensor,
               b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The kernel in program order: the pre-loop tile is read once, before any write."""
    a, b = a.clone(), b.clone()
    acc = torch.zeros(BM, BN, dtype=torch.float32)
    if pattern == "operand_b":
        b_tile = b[:BK, :BN].float()
        for k in range(0, K, BK):
            acc = a[:BM, k:k + BK].float() @ b_tile + acc
            if write:
                b[:BM, :BN] = acc.to(b.dtype)
        return acc, a, b
    a_tile = a[:BM, :BK].float()
    if write and pattern == "before_loop":
        a[:BM, :BN] = 1.0
    for k in range(0, K, BK):
        acc = a_tile @ b[k:k + BK, :BN].float() + acc
        if write and pattern != "before_loop":
            a[:BM, :BN] = acc.to(a.dtype)
    if pattern == "in_loop_use_after":
        acc = a_tile @ b[:BK, :BN].float() + acc
    return acc, a, b


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific pass")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("grf_mode", ["default", "256"])
@pytest.mark.parametrize("num_warps", [1, 4, 8])
@pytest.mark.parametrize("write", [True, False], ids=["write", "no_write"])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_pre_loop_operand_not_moved_past_write(pattern, write, num_warps, grf_mode, dtype, device, with_allocator):
    torch.manual_seed(0)
    a = (torch.randn(M, K, device=device) / 8).to(dtype)
    b = (torch.randn(K, N, device=device) / 8).to(dtype)
    ref_out, ref_a, ref_b = _reference(pattern, write, a.cpu(), b.cpu())

    out = torch.empty(M, N, dtype=torch.float32, device=device)
    w = b if pattern == "operand_b" else a
    _pre_loop_operand_kernel[(1, )](a, b, w, out, M=M, N=N, K=K, BM=BM, BN=BN, BK=BK, PATTERN=pattern, WRITE=write,
                                    SYNC=write and num_warps > 1, num_warps=num_warps, grf_mode=grf_mode)

    torch.testing.assert_close(out.cpu(), ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(a.cpu().float(), ref_a.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(b.cpu().float(), ref_b.float(), atol=1e-2, rtol=1e-2)
