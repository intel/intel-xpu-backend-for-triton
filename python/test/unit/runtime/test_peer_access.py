import pytest
import torch

import triton
import triton.language as tl


def test_peer_access(device):
    if not hasattr(torch, device):
       pytest.skip(f"{device} does not support peer access")
    if getattr(torch, device).device_count() < 2:
       pytest.skip("need at least 2 devices to test peer access")

    @triton.jit
    def device_accumulate(my_ptr, peer_ptr):
        tl.store(my_ptr, tl.load(my_ptr) + tl.load(peer_ptr))

    my_tensor = torch.randn(1, device=f"{device}:0")
    peer_tensor = torch.randn(1, device=f"{device}:1")
    expected = my_tensor + peer_tensor.to(device=f"{device}:0")

    device_accumulate[(1,1,1)](my_tensor, peer_tensor)

    torch.testing.assert_close(my_tensor, expected)
