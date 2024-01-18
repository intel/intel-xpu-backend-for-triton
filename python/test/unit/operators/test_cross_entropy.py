import pytest
import torch
import intel_extension_for_pytorch  # type: ignore # noqa: F401

import triton
import triton.ops

# FIXME remove this once Triton L0 queue and IPEX SYCL queue can be synchronized through events
torch.xpu.enable_sync_mode()


@pytest.mark.parametrize("M, N, dtype, mode", [  #
    (M, N, dtype, mode)
    for M in [1024, 821]
    for N in [512, 857, 1871, 2089, 8573, 31000]
    for dtype in ['float16', 'float32']
    for mode in ['forward', 'backward']
])
def test_op(M, N, dtype, mode):
    pytest.skip("FIXME: Port get_device_capability to XPU")
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8 and dtype == "bfloat16":
        pytest.skip("Only test bfloat16 on devices with sm >= 80")
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[dtype]
    # create inputs
    x = torch.randn(M, N, dtype=dtype, device='cuda', requires_grad=True)
    idx = 4 + torch.ones(M, dtype=torch.int64, device='cuda')
    # forward pass
    tt_y = triton.ops.cross_entropy(x, idx)
    th_y = torch.nn.CrossEntropyLoss(reduction="none")(x, idx)
    if mode == 'forward':
        torch.testing.assert_close(th_y, tt_y)
    # backward pass
    elif mode == 'backward':
        dy = torch.randn_like(tt_y)
        # triton backward
        tt_y.backward(dy)
        tt_dx = x.grad.clone()
        # torch backward
        x.grad = None
        th_y.backward(dy)
        th_dx = x.grad.clone()
        if dtype == torch.float16:
            torch.testing.assert_close(th_dx, tt_dx, rtol=0.001, atol=0.001)
        else:
            torch.testing.assert_close(th_dx, tt_dx)
