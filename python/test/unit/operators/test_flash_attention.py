import pytest
import torch
import intel_extension_for_pytorch  # type: ignore # noqa: F401

import triton
import triton.ops

# FIXME remove this once Triton L0 queue and IPEX SYCL queue can be synchronized through events
torch.xpu.enable_sync_mode()


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [  #
    (2, 4, 512, 16),
    (2, 4, 512, 32),
    (2, 4, 512, 64),
    (2, 4, 512, 128),
])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('seq_par', [True, False])
def test_op(Z, H, N_CTX, D_HEAD, dtype, causal, seq_par, device):
    if D_HEAD != 16:
        pytest.skip("FIXME: Enable larger problem sizes when tl.dot uses DPAS")

    import os
    enable_tma = os.environ.get('ENABLE_TMA', 'not found').lower()
    if enable_tma in ["on", "true", "1"]:
        if dtype == torch.bfloat16:
            pytest.xfail('bfloat16 tma not support currently')

    capability = 0
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()

    interpreter = os.environ.get("TRITON_INTERPRET", 'not found') in ["on", "true", "1"]
    if torch.cuda.is_available():
        if not interpreter and capability[0] < 8:
            pytest.xfail("Flash attention only supported for compute capability >= 80")
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device).normal_(mean=0., std=0.5).requires_grad_()
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=device))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # # triton implementation
    tri_out = triton.ops.attention(q, k, v, causal, sm_scale, seq_par)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    atol = 1e-1 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(torch.nn.functional.normalize(torch.flatten(ref_out), dim=0),
                               torch.nn.functional.normalize(torch.flatten(tri_out), dim=0), atol=atol, rtol=0)
    torch.testing.assert_close(torch.nn.functional.normalize(torch.flatten(ref_dv), dim=0),
                               torch.nn.functional.normalize(torch.flatten(tri_dv), dim=0), atol=atol, rtol=0)
    torch.testing.assert_close(torch.nn.functional.normalize(torch.flatten(ref_dk), dim=0),
                               torch.nn.functional.normalize(torch.flatten(tri_dk), dim=0), atol=atol, rtol=0)
    torch.testing.assert_close(torch.nn.functional.normalize(torch.flatten(ref_dq), dim=0),
                               torch.nn.functional.normalize(torch.flatten(tri_dq), dim=0), atol=atol, rtol=0)


try:
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
# vary seq length for fixed head and batch=4
configs = [
    triton.testing.Benchmark(
        x_names=['N_CTX'], x_vals=[2**i for i in range(10, 14)], line_arg='provider',
        line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
        line_names=['Triton'] + (['Flash'] if HAS_FLASH else []), styles=[('red', '-'), ('blue', '-')], ylabel='ms',
        plot_name=f'fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-{casual}-{seq_par}', args={
            'H': N_HEADS,
            'BATCH': BATCH,
            'D_HEAD': D_HEAD,
            'dtype': torch.float16,
            'mode': mode,
            'casual': casual,
            'seq_par': seq_par,
        }) for mode in ['fwd', 'bwd'] for casual in [True, False] for seq_par in [True, False]
]


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, mode, casual, seq_par, provider, dtype=torch.float16, device="xpu"):
    assert mode in ['fwd', 'bwd']
    warmup = 25
    rep = 100
    sm_scale = 1.3
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    if provider == "triton":
        fn = lambda: triton.ops.attention(q, k, v, casual, sm_scale, seq_par)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    if provider == "flash":
        lengths = torch.full((BATCH, ), fill_value=N_CTX, device=device)
        cu_seqlens = torch.zeros((BATCH + 1, ), device=device, dtype=torch.int32)
        cu_seqlens[1:] = lengths.cumsum(0)
        fn = lambda: flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=sm_scale, causal=casual)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms


# only works on post-Ampere GPUs right now
# bench_flash_attention.run(save_path='.', print_data=True)
