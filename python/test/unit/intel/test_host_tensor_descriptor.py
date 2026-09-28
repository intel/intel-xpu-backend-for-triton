"""End-to-end tests for host-side TensorDescriptor on Intel XPU backend.

Verifies that host-side TensorDescriptor objects (created on the host and passed
as kernel arguments) reach the efficient 2D block I/O hardware path, producing
the same results and codegen as device-side tl.make_tensor_descriptor.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu
from triton.tools.tensor_descriptor import TensorDescriptor


def _has_2d_block_io():
    """Check if current device supports 2D block I/O."""
    return triton.runtime.driver.active.get_current_target().arch.get('has_2d_block_io', False)


@triton.jit
def _matmul_kernel(a_desc_or_ptr, b_desc_or_ptr, c_ptr, M, N, K, stride_am: tl.constexpr, stride_ak: tl.constexpr,
                   stride_bk: tl.constexpr, stride_bn: tl.constexpr, stride_cm, stride_cn, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Simple matmul kernel that works with both host and device descriptors."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if isinstance(a_desc_or_ptr, tl.tensor_descriptor):
        a_desc = a_desc_or_ptr
    else:
        a_desc = tl.make_tensor_descriptor(a_desc_or_ptr, shape=[M, K], strides=[stride_am, stride_ak],
                                           block_shape=[BLOCK_M, BLOCK_K])
    if isinstance(b_desc_or_ptr, tl.tensor_descriptor):
        b_desc = b_desc_or_ptr
    else:
        b_desc = tl.make_tensor_descriptor(b_desc_or_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
                                           block_shape=[BLOCK_K, BLOCK_N])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = a_desc.load([pid_m * BLOCK_M, k])
        b = b_desc.load([k, pid_n * BLOCK_N])
        acc = tl.dot(a, b, acc)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@pytest.mark.parametrize("M, N, K", [[128, 128, 64], [64, 64, 32]])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_host_descriptor_matmul_2d_block_io(M, N, K, dtype, device):
    """Host-side TensorDescriptor matmul produces correct results and 2D block loads."""
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32

    torch.manual_seed(42)
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((K, N), dtype=dtype, device=device)

    # Device-side path.
    c_device = torch.empty((M, N), dtype=torch.float32, device=device)
    grid = (M // BLOCK_M, N // BLOCK_N)
    kernel_device = _matmul_kernel[grid](
        a,
        b,
        c_device,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c_device.stride(0),
        c_device.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Host-side path.
    c_host = torch.empty((M, N), dtype=torch.float32, device=device)
    a_desc = TensorDescriptor(a, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor(b, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N])
    kernel_host = _matmul_kernel[grid](
        a_desc,
        b_desc,
        c_host,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c_host.stride(0),
        c_host.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Correctness: both paths match reference.
    ref = (a.to(torch.float32) @ b.to(torch.float32))
    torch.testing.assert_close(c_device, ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(c_host, ref, rtol=1e-2, atol=1e-2)

    # Codegen: both paths must generate the same number of 2D block loads.
    device_llir = kernel_device.asm["llir"]
    host_llir = kernel_host.asm["llir"]
    device_loads = device_llir.count('spirv_Subgroup2DBlockLoad') + device_llir.count('GenISA.LSC2DBlockRead')
    host_loads = host_llir.count('spirv_Subgroup2DBlockLoad') + host_llir.count('GenISA.LSC2DBlockRead')
    assert device_loads > 0, "device-side path: no 2D block loads found"
    assert host_loads == device_loads, \
        f"host has {host_loads} 2D block loads, expected {device_loads} (same as device)"


@triton.jit
def _split_barrier_matmul_kernel(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                STRIDE_A: tl.constexpr, STRIDE_B: tl.constexpr):
    _matmul_kernel(a, b, c, M, N, K, STRIDE_A, 1, STRIDE_B, 1, N, 1, 256, 256, 32)


@pytest.mark.parametrize("M, N, K", [(256, 256, 256), (512, 768, 384), (64, 1024, 512), (1024, 64, 512),
                                     (257, 264, 72), (65, 72, 8), (65, 72, 32), (65, 72, 40)])
@pytest.mark.parametrize("num_stages", [2, 3, 4])
@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
def test_workgroup_split_barrier_matmul_shapes(M, N, K, num_stages, device):
    properties = triton.runtime.driver.active.get_current_target().arch
    if not properties.get('has_split_work_group_barrier', False) or not _has_2d_block_io():
        pytest.skip("Workgroup split barriers and 2D block I/O are required")
    torch.manual_seed(42)
    a = torch.randn((M, triton.cdiv(K, 64) * 64), dtype=torch.bfloat16, device=device)[:, :K]
    b = torch.randn((K, triton.cdiv(N, 64) * 64), dtype=torch.bfloat16, device=device)[:, :N]
    c = torch.empty((M, N), dtype=torch.float32, device=device)
    grid = (triton.cdiv(M, 256), triton.cdiv(N, 256))
    if K < 32:
        kernel = _matmul_kernel[grid](
            a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride(),
            BLOCK_M=256, BLOCK_N=256, BLOCK_K=32, num_warps=32, num_stages=num_stages,
            grf_mode='256', use_barrier=True)
    else:
        kernel = _split_barrier_matmul_kernel[grid](
            a, b, c, M, N, K, a.stride(0), b.stride(0), num_warps=32, num_stages=num_stages,
            grf_mode='256', use_barrier=True)
    torch.testing.assert_close(c, a.float() @ b.float(), atol=2e-3, rtol=1e-4)
    if K > 32:
        assert '__spirv_ControlBarrierArriveINTEL' in kernel.asm['llir']
        assert '__spirv_ControlBarrierWaitINTEL' in kernel.asm['llir']
        assert 'intel_manageable_barrier' not in kernel.asm['llir']


@triton.jit
def _split_barrier_attention_kernel(q, k, v, out, blocks, qlens, klens, NUM_SEQS: tl.constexpr,
                                    MAX_Q: tl.constexpr, MAX_K: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    left, right = 0, NUM_SEQS
    while left < right:
        mid = (left + right) // 2
        offset = tl.load(blocks + mid)
        if offset <= pid:
            left = mid + 1
        else:
            right = mid
    seq = left - 1
    row = (pid - tl.load(blocks + seq)) * 16
    qlen, klen = tl.load(qlens + seq), tl.load(klens + seq)
    if row >= qlen:
        return
    qdesc = tl.make_tensor_descriptor(q + seq * MAX_Q * D, [qlen, D], [D, 1], [16, D])
    kdesc = tl.make_tensor_descriptor(k + seq * MAX_K * D, [klen, D], [D, 1], [32, D])
    vdesc = tl.make_tensor_descriptor(v + seq * MAX_K * D, [klen, D], [D, 1], [32, D])
    query = qdesc.load([row, 0])
    m = tl.full((16,), float('-inf'), tl.float32)
    l = tl.full((16,), 1., tl.float32)
    acc = tl.zeros((16, D), tl.float32)
    rows = row + tl.arange(0, 16)
    for start in range(0, klen, 32):
        key, value = kdesc.load([start, 0]), vdesc.load([start, 0])
        scores = tl.dot(query, key.T) * D**-.5
        cols = start + tl.arange(0, 32)
        scores = tl.where((cols[None, :] < klen) & (cols[None, :] <= rows[:, None] + klen - qlen),
                          scores, float('-inf'))
        new_m = tl.maximum(m, tl.max(scores, 1))
        p = tl.exp(scores - new_m[:, None])
        alpha = tl.exp(m - new_m)
        l = l * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(query.dtype), value, acc)
        m = new_m
    cols = tl.arange(0, D)
    tl.store(out + seq * MAX_Q * D + rows[:, None] * D + cols[None, :],
             acc / l[:, None], rows[:, None] < qlen)


@pytest.mark.parametrize('num_stages', [2, 3])
@pytest.mark.parametrize('num_warps', [4, 8])
@pytest.mark.skipif(not is_xpu(), reason='XPU-specific test')
def test_workgroup_split_barrier_attention_dynamic_lengths(num_stages, num_warps, device):
    properties = triton.runtime.driver.active.get_current_target().arch
    if not properties.get('has_split_work_group_barrier', False) or not _has_2d_block_io():
        pytest.skip('Workgroup split barriers and 2D block I/O are required')
    torch.manual_seed(42)
    query_lengths, kv_lengths = [0, 1, 17, 64], [0, 129, 193, 512]
    q = torch.randn((4, 64, 128), dtype=torch.bfloat16, device=device)
    k = torch.randn((4, 512, 128), dtype=q.dtype, device=device)
    v = torch.randn_like(k)
    out = torch.full_like(q, float('nan'))
    offsets = [0]
    for length in query_lengths:
        offsets.append(offsets[-1] + triton.cdiv(length, 16))
    blocks = torch.tensor(offsets, dtype=torch.int32, device=device)
    qlens = torch.tensor(query_lengths, dtype=torch.int32, device=device)
    klens = torch.tensor(kv_lengths, dtype=torch.int32, device=device)
    kernel = _split_barrier_attention_kernel[(offsets[-1] + 1,)](
        q, k, v, out, blocks, qlens, klens, 4, 64, 512, 128,
        num_warps=num_warps, num_stages=num_stages, use_barrier=True)
    for i, (ql, kl) in enumerate(zip(query_lengths, kv_lengths)):
        if ql == 0:
            continue
        scores = (q[i, :ql].float() @ k[i, :kl].float().T) * 128**-.5
        mask = torch.arange(kl, device=device)[None, :] > torch.arange(ql, device=device)[:, None] + kl - ql
        scores.masked_fill_(mask, float('-inf'))
        ref = scores.softmax(-1) @ v[i, :kl].float()
        torch.testing.assert_close(out[i, :ql].float(), ref, atol=.015, rtol=.015)
        assert out[i, ql:].isnan().all()
    assert out[0].isnan().all()
    assert '__spirv_ControlBarrierArriveINTEL' in kernel.asm['llir']
    assert '__spirv_ControlBarrierWaitINTEL' in kernel.asm['llir']
    assert 'intel_manageable_barrier' not in kernel.asm['llir']
