import torch
import triton
import triton.language as tl
import pytest
from triton._internal_testing import is_xpu


@pytest.mark.skipif(not is_xpu(), reason="Test requires XPU")
def test_pinned_memory_access(device):
    # 1. Create CPU pinned tensor
    N = 1024
    cpu_tensor = torch.arange(N, dtype=torch.float32).pin_memory()

    # 2. Triton kernel: y = x + 1
    @triton.jit
    def add_one_kernel(X, Y, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        y = x + 1.0
        tl.store(Y + offs, y, mask=mask)

    # 3. Output tensor on device
    out_tensor = torch.empty_like(cpu_tensor, device=device)

    # 4. Launch kernel
    # Passing cpu_tensor directly - driver should handle the pinned memory pointer
    BLOCK = 1024
    add_one_kernel[(1, )](cpu_tensor, out_tensor, N, BLOCK=BLOCK)

    # 5. Verify correctness
    assert torch.allclose(cpu_tensor + 1, out_tensor.cpu())
