import torch
import intel_extension_for_pytorch

import triton
import triton.language as tl



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
              ):

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
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
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
    num_stages = 4 if Lk <= 64 else 3
    num_warps = 8 if Lq == 64 else 16
    stage = 3 if causal else 1
    grid = (q.shape[0],  q.shape[1],triton.cdiv(q.shape[2], BLOCK_M))
    print("Q stride =", q.stride(0), q.stride(1), q.stride(2), q.stride(3))
    print("K stride =", k.stride(0), k.stride(1), k.stride(2), k.stride(3))
    print("V stride =", v.stride(0), v.stride(1), v.stride(2), v.stride(3))
    print(q.shape[0], q.shape[1])
    print("causal = ", causal)
    print(stage)
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

# torch.manual_seed(0)
# Z = 1
# H = 2
# N_CTX = 1024
# D_HEAD = 64
# causal = False
# dtype=torch.float16
# q = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype)
# k = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype)
# v = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype)
# sm_scale = 0.125
# dout = torch.randn_like(q)
# #torch_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
# #torch.save(torch_output, "./torch_output.pt")
# torch_output = torch.load("./torch_output.pt")
# triton_output = forward(q, k, v, causal, sm_scale)

# torch_outputf32 = torch_output.to(torch.float32)
# if torch.allclose(triton_output, torch_outputf32, atol=1e-3, rtol=1e-3):
#     print("✅ Triton and Torch match")
# else:
#     print("❌ Triton and Torch differ")



@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['Z', 'H', 'N_CTX', 'D_HEAD'],
        x_vals=[[4, 48, 1024, 64]],
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        #line_vals=['triton', 'onednn'],
        line_vals=['triton'],
        # label name for the lines
        #line_names=["Triton", "oneDNN"],
        line_names=["Triton"],
        # line styles
        #styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel="TFLOPS",  # label name for the y-axis
        plot_name="attn-performance",
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(Z, H, N_CTX, D_HEAD, provider):
    causal = True
    dtype=torch.float16
    q = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype)
    k = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype)
    v = torch.randn((Z, H, N_CTX, D_HEAD), device='xpu', dtype=dtype)
    sm_scale = 0.125
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'onednn':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal, scale = sm_scale), rep=1000, quantiles=quantiles,
                                                     fast_flush=False)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: forward(q, k, v, causal, sm_scale), rep=1000, quantiles=quantiles,
                                                     fast_flush=False)

    def perf(ms):
        return  2 * 2 * Z * H * N_CTX * N_CTX * D_HEAD * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=False, print_data=True)


