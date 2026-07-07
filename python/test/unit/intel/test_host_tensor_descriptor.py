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
def _matmul_kernel(a_desc_or_ptr, b_desc_or_ptr, c_ptr, M, N, K,
                   stride_am: tl.constexpr, stride_ak: tl.constexpr,
                   stride_bk: tl.constexpr, stride_bn: tl.constexpr,
                   stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Simple matmul kernel that works with both host and device descriptors."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if isinstance(a_desc_or_ptr, tl.tensor_descriptor):
        a_desc = a_desc_or_ptr
    else:
        a_desc = tl.make_tensor_descriptor(a_desc_or_ptr, shape=[M, K],
                                           strides=[stride_am, stride_ak],
                                           block_shape=[BLOCK_M, BLOCK_K])
    if isinstance(b_desc_or_ptr, tl.tensor_descriptor):
        b_desc = b_desc_or_ptr
    else:
        b_desc = tl.make_tensor_descriptor(b_desc_or_ptr, shape=[K, N],
                                           strides=[stride_bk, stride_bn],
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
        a, b, c_device, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c_device.stride(0), c_device.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # Host-side path.
    c_host = torch.empty((M, N), dtype=torch.float32, device=device)
    a_desc = TensorDescriptor(a, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor(b, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N])
    kernel_host = _matmul_kernel[grid](
        a_desc, b_desc, c_host, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c_host.stride(0), c_host.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # Correctness: both paths match reference.
    ref = (a.to(torch.float32) @ b.to(torch.float32))
    torch.testing.assert_close(c_device, ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(c_host, ref, rtol=1e-2, atol=1e-2)

    # Codegen: both paths should generate 2D block loads.
    for label, kernel in [("device", kernel_device), ("host", kernel_host)]:
        llir = kernel.asm["llir"]
        load_count = (llir.count('spirv_Subgroup2DBlockLoad') +
                      llir.count('GenISA.LSC2DBlockRead'))
        assert load_count > 0, \
            f"{label}-side path: expected 2D block load in LLIR but found none"
