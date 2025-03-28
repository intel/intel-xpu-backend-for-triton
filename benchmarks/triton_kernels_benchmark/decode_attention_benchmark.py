import torch

from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd

import triton_kernels_benchmark as benchmark_suit


# pylint: disable=unused-argument
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['BATCH', 'SEQ_LENS', 'Q_HEAD_NUM', 'KV_HEAD_NUM', 'HEAD_DIM', 'MODE', 'VALIDATE'],
        x_vals=[  #
            [bs, [1024, 64], 32, 8, 128, 'fwd', False] for bs in [1, 16, 32, 64, 128]
        ] + [  #
            [bs, [1024, 64], 32, 32, 96, 'fwd', False] for bs in [1, 16, 32, 64, 128]
        ] + [  #
            [bs, [1024, 64], 28, 4, 128, 'fwd', False] for bs in [1, 16, 32, 64, 128]
        ],
        line_arg='provider',
        # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=[
            'triton',
        ],
        # label name for the lines
        line_names=[
            'Triton',
        ],
        # line styles
        styles=[('green', '-'), ('green', '--'), ('blue', '-'), ('blue', '--')],
        ylabel=['GB/s', 'TFlops'],  # label name for the y-axis
        plot_name='decode-attn-performance',
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(BATCH, SEQ_LENS, Q_HEAD_NUM, KV_HEAD_NUM, HEAD_DIM, MODE, VALIDATE, provider):
    dtype = torch.bfloat16
    N_CTX = sum(SEQ_LENS)
    total_tokens = BATCH * N_CTX
    sm_scale = 1.0 / (HEAD_DIM**0.5)
    num_kv_splits = 8

    # q represents the new token being generated, one per batch
    q = torch.randn(BATCH, Q_HEAD_NUM, HEAD_DIM, dtype=dtype, device='xpu')

    # k_buffer and v_buffer represent all previous tokens
    k_buffer = torch.randn(total_tokens, KV_HEAD_NUM, HEAD_DIM, dtype=dtype, device='xpu')
    v_buffer = torch.randn(total_tokens, KV_HEAD_NUM, HEAD_DIM, dtype=dtype, device='xpu')

    # o will have the same shape as q
    o = torch.zeros(BATCH, Q_HEAD_NUM, HEAD_DIM, dtype=dtype, device='xpu')

    b_seq_len = torch.full((BATCH, ), N_CTX, device='xpu')

    kv_indptr = torch.zeros((BATCH + 1, ), dtype=torch.int32, device='xpu')
    kv_indptr[1:BATCH + 1] = torch.cumsum(b_seq_len[:BATCH], dim=0)
    kv_indices = torch.arange(total_tokens, device='xpu')

    attn_logits = torch.empty(
        (BATCH, Q_HEAD_NUM, num_kv_splits, HEAD_DIM + 1),
        dtype=torch.float32,
        device='xpu',
    )

    quantiles = [0.5, 0.0, 1.0]

    if provider == 'triton':
        triton_fn = lambda: decode_attention_fwd(q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits,
                                                 num_kv_splits, sm_scale, logit_cap=0.0)
        # TODO: decode attention do not have validation function
        if VALIDATE:
            raise NotImplementedError('Validation is not implemented for decode stage')

        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(triton_fn, n_warmup=10, n_repeat=10, quantiles=quantiles)

    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    tflops = lambda ms: 2 * BATCH * (Q_HEAD_NUM + KV_HEAD_NUM * N_CTX) * N_CTX * HEAD_DIM * (1e-12) / (ms * 1e-3)

    gbps = lambda ms: 2 * BATCH * (Q_HEAD_NUM + KV_HEAD_NUM * N_CTX) * HEAD_DIM * 2 * (1e-9) / (ms * 1e-3)

    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
