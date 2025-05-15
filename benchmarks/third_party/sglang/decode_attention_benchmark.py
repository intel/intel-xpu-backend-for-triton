import torch

from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd

import triton_kernels_benchmark as benchmark_suit


def gen_args(B, N_CTX, H_Q, H_KV, D, dtype, device):

    total_tokens = B * N_CTX
    sm_scale = 1.0 / (D**0.5)
    max_kv_splits = 8
    num_kv_splits = torch.full((B, ), 4, dtype=torch.int32, device=device)

    # q represents the new token being generated, one per B
    q = torch.randn(B, H_Q, D, dtype=dtype, device=device)

    # k_buffer and v_buffer represent all previous tokens
    k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)
    v_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)

    # o will have the same shape as q
    o = torch.zeros(B, H_Q, D, dtype=dtype, device=device)

    b_seq_len = torch.full((B, ), N_CTX, device=device)

    kv_indptr = torch.zeros((B + 1, ), dtype=torch.int32, device=device)
    kv_indptr[1:B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
    kv_indices = torch.arange(total_tokens, device=device)

    attn_logits = torch.empty(
        (B, H_Q, max_kv_splits, D),
        dtype=torch.float32,
        device=device,
    )
    attn_lse = torch.empty(
        (B, H_Q, max_kv_splits),
        dtype=torch.float32,
        device=device,
    )

    return (q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits, max_kv_splits,
            sm_scale)


# pylint: disable=unused-argument
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['B', 'SEQ_LENS', 'H_Q', 'H_KV', 'D', 'MODE', 'VALIDATE'],
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
def benchmark(B, SEQ_LENS, H_Q, H_KV, D, MODE, VALIDATE, provider):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    quantiles = [0.5, 0.0, 1.0]
    N_CTX = sum(SEQ_LENS)

    q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits, max_kv_splits, sm_scale = gen_args(
        B, N_CTX, H_Q, H_KV, D, dtype, 'xpu')

    if provider == 'triton':
        triton_fn = lambda: decode_attention_fwd(q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse,
                                                 num_kv_splits, max_kv_splits, sm_scale)

        # TODO: decode attention should have the validation function
        if VALIDATE:
            raise NotImplementedError('Validation is not implemented for decode stage')

        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(triton_fn, n_warmup=10, n_repeat=10, quantiles=quantiles)

    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    tflops = lambda ms: 2 * B * (H_Q + H_KV) * N_CTX * D * (1e-12) / (ms * 1e-3)
    gbps = lambda ms: 2 * B * (H_Q + H_KV * N_CTX) * D * 2 * (1e-9) / (ms * 1e-3)

    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
