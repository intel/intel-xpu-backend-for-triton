import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suit
import xetla_kernel

if benchmark_suit.USE_IPEX_OPTION:
    import intel_extension_for_pytorch  # type: ignore # noqa: F401


# pylint: disable=unused-argument
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        acc += tl.dot(p.to(tl.float16), v)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,  #
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,  #
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vk: tl.constexpr, stride_vn: tl.constexpr,  #
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,  #
              Z: tl.constexpr, H: tl.constexpr,  #
              N_CTX: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):  # pylint: disable=unused-argument

    start_m = tl.program_id(2)
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        tl.debug_barrier()
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    # m_ptrs = M + off_hz * N_CTX + offs_m
    # tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def forward(q, k, v, causal, sm_scale):
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q, dtype=torch.float32)
    BLOCK_M = 128
    BLOCK_N = 64 if Lk <= 64 else 32
    num_stages = 3
    num_warps = 8 if Lq == 64 else 16
    stage = 3 if causal else 1
    grid = (q.shape[0], q.shape[1], triton.cdiv(q.shape[2], BLOCK_M))
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    _attn_fwd[grid](
        q, k, v, sm_scale, M, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        q.shape[0], q.shape[1],  #
        N_CTX=q.shape[2],  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_N=BLOCK_N,  #
        BLOCK_DMODEL=Lk,  #
        STAGE=stage,  #
        num_warps=num_warps,  #
        num_stages=num_stages  #
    )
    return o


@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['Z', 'H', 'N_CTX', 'D_HEAD', 'CAUSAL'],
        x_vals=[  #
            [1, 16, 16384, 128, False],  #
            [1, 16, 16384, 128, True],  #
            [1, 32, 16384, 64, False],  #
            [1, 32, 16384, 64, True],  #
            [2, 16, 8192, 128, False],  #
            [2, 16, 8192, 128, True],  #
            [2, 32, 8192, 64, False],  #
            [2, 32, 8192, 64, True],  #
            [4, 16, 4096, 128, False],  #
            [4, 16, 4096, 128, True],  #
            [4, 32, 4096, 64, False],  #
            [4, 32, 4096, 64, True],  #
            [4, 48, 1024, 64, False],  #
            [4, 48, 1024, 64, True],  #
            [8, 16, 2048, 128, False],  #
            [8, 16, 2048, 128, True],  #
            [8, 32, 2048, 64, False],  #
            [8, 32, 2048, 64, True],  #
            [16, 16, 1024, 128, False],  #
            [16, 16, 1024, 128, True],  #
            [16, 32, 1024, 64, False],  #
            [16, 32, 1024, 64, True],  #
            [32, 16, 512, 128, False],  #
            [32, 16, 512, 128, True],  #
            [32, 32, 512, 64, False],  #
            [32, 32, 512, 64, True],  #
        ],
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['triton', 'xetla'],
        # label name for the lines
        line_names=['Triton', 'XeTLA'],
        # line styles
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel=['GB/s', 'TFlops'],  # label name for the y-axis
        plot_name='attn-performance',
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(Z, H, N_CTX, D_HEAD, CAUSAL, provider):
    dtype = torch.float16
    q = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype)
    k = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype)
    v = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype)
    sm_scale = 0.125
    quantiles = [0.5, 0.0, 1.0]
    if provider == 'onednn':
        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(
            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=
                                                                     CAUSAL, scale=sm_scale), warmup=10, rep=10,
            quantiles=quantiles)

    elif provider == 'triton':
        # FIXME: remove below if condition when extend attention support for Causal = True done
        # https://github.com/intel/intel-xpu-backend-for-triton/issues/1102
        import os
        if os.environ.get('TRITON_INTEL_ADVANCED_PATH', '0') == '1' and CAUSAL:
            min_ms, max_ms, mean, cv = (float('inf'), ) * 4
        else:
            triton_fn = lambda: forward(q, k, v, CAUSAL, sm_scale)
            if benchmark_suit.USE_IPEX_OPTION:
                torch_fn = lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=CAUSAL, scale=sm_scale).to(torch.float32)
            else:
                # FIXME: use torch sdpa for result check after https://github.com/intel/intel-xpu-backend-for-triton/issues/2042 fixed
                torch_fn = lambda: torch.nn.functional.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(
                ), attn_mask=None, dropout_p=0.0, is_causal=CAUSAL, scale=sm_scale).to(torch.float32)
            atol = 1e-1 if N_CTX == 16384 else 1e-2
            benchmark_suit.assert_close(triton_fn(), torch_fn(), atol=atol, rtol=1e-3, err_msg='triton to torch')
            _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(triton_fn, warmup=10, rep=10, quantiles=quantiles,
                                                                  kernel_name='_attn_fwd')

    elif provider == 'xetla':
        module_name = f'flash_attn_causal_{CAUSAL}'.lower()
        func = getattr(xetla_kernel, module_name)
        out = torch.empty_like(q, device='xpu', dtype=dtype)
        size_score = Z * H * N_CTX * N_CTX
        size_attn_mask = Z * N_CTX * N_CTX
        dropout_mask = torch.empty((size_score, ), device='xpu', dtype=torch.uint8)
        bias = torch.empty((size_attn_mask, ), device='xpu', dtype=dtype)
        size_ml = Z * H * N_CTX
        m = torch.empty((size_ml, ), device='xpu', dtype=torch.float)
        l = torch.empty((size_ml, ), device='xpu', dtype=torch.float)

        xetla_fn = lambda: func(q, k, v, out, dropout_mask, bias, m, l, Z, H, D_HEAD, N_CTX, N_CTX, sm_scale)
        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(xetla_fn, warmup=10, rep=10, quantiles=quantiles,
                                                              kernel_name='gpu::xetla::fmha::FmhaForwardKernel<')

    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    tflops = lambda mean: 2 * 2 * Z * H * N_CTX * N_CTX * D_HEAD * (1e-12) / (mean * 1e-3)
    gbps = lambda mean: Z * H * (N_CTX * D_HEAD + N_CTX * D_HEAD) * 2 * 2 * (1e-9) / (mean * 1e-3)

    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
