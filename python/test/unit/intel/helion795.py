from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from helion.runtime import default_launcher as _default_launcher

DEVICE = 'xpu'


@triton.jit
def _helion_matmul(x, y, epilogue_closure_0, out, _BLOCK_SIZE_0: tl.constexpr, _BLOCK_SIZE_1: tl.constexpr,
                   _BLOCK_SIZE_2: tl.constexpr):
    num_pid_m = tl.cdiv(1024, _BLOCK_SIZE_0)
    num_pid_n = tl.cdiv(1024, _BLOCK_SIZE_1)
    inner_2d_pid = tl.program_id(0)
    num_pid_in_group = 64 * num_pid_n
    group_id = inner_2d_pid // num_pid_in_group
    first_pid_m = group_id * 64
    group_size_m = min(num_pid_m - first_pid_m, 64)
    pid_0 = first_pid_m + inner_2d_pid % num_pid_in_group % group_size_m
    pid_1 = inner_2d_pid % num_pid_in_group // group_size_m
    offset_0 = pid_0 * _BLOCK_SIZE_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    for offset_2 in tl.range(0, 1024, _BLOCK_SIZE_2):
        acc_copy = acc
        load = tl.load(
            tl.make_block_ptr(x, [1024, 1024], [1024, 1], [offset_0, offset_2], [_BLOCK_SIZE_0, _BLOCK_SIZE_2], [1, 0]),
            boundary_check=[0, 1], padding_option='zero')
        load_1 = tl.load(
            tl.make_block_ptr(y, [1024, 1024], [1024, 1], [offset_2, offset_1], [_BLOCK_SIZE_2, _BLOCK_SIZE_1], [1, 0]),
            boundary_check=[0, 1], padding_option='zero')
        acc = tl.dot(tl.cast(load, tl.float16), tl.cast(load_1, tl.float16), acc=acc_copy, input_precision='tf32',
                     out_dtype=tl.float32)
    load_2 = tl.load(
        tl.make_block_ptr(epilogue_closure_0, [1, 1024], [1024, 1], [0, offset_1], [1, _BLOCK_SIZE_1], [1, 0]))
    v_0 = tl.cast(load_2, tl.float32)
    v_1 = acc + v_0
    v_2 = tl.full([], 0, tl.int32)
    v_3 = triton_helpers.maximum(v_2, v_1)
    v_4 = tl.cast(v_3, tl.float16)
    tl.store(
        tl.make_block_ptr(out, [1024, 1024], [1024, 1], [offset_0, offset_1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1], [1, 0]),
        v_4, boundary_check=[0, 1])


def matmul(x, y, epilogue: Callable[[Tensor, tuple[Tensor, ...]], Tensor] = lambda acc, tile: acc, *,
           _launcher=_default_launcher):
    """
    Performs matrix multiplication of x and y with an optional epilogue function.
    Args:
        x (Tensor): Left matrix of shape [m, k].
        y (Tensor): Right matrix of shape [k, n].
        epilogue (Callable, optional): Function applied to the accumulator and tile indices
            after the matmul. Defaults to identity (no change).
    Returns:
        Tensor: Resulting matrix of shape [m, n].
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f'size mismatch {k} != {k2}'
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    _BLOCK_SIZE_0 = 64
    _BLOCK_SIZE_1 = 64
    _BLOCK_SIZE_2 = 16
    _launcher(_helion_matmul, (triton.cdiv(1024, _BLOCK_SIZE_0) * triton.cdiv(1024, _BLOCK_SIZE_1), ), x, y,
              epilogue.__closure__[0].cell_contents, out, _BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_2, num_warps=2,
              num_stages=4)
    return out


bias = torch.ones([1, 1024], device=DEVICE, dtype=torch.float16)
args = (
    torch.ones([1024, 1024], device=DEVICE, dtype=torch.float16),
    torch.ones([1024, 1024], device=DEVICE, dtype=torch.float16),
    lambda acc, tile: torch.relu(acc + bias[tile]),
)

bias.fill_(0.7)
args[0].fill_(0.1)
args[1].fill_(0.2)


def make_epilogue(bias):

    def epilogue(acc, tile):
        return acc + bias[tile[0], tile[1]]

    return epilogue


epilogue = make_epilogue(bias)

out = matmul(args[0], args[1], epilogue)
torch.xpu.synchronize()
torch.testing.assert_close(out, torch.relu(args[0] @ args[1] + bias))
