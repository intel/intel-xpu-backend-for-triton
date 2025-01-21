import torch

from sglang.srt.layers.attention.triton_ops.prefill_attention import context_attention_fwd

import triton_kernels_benchmark as benchmark_suit


def gen_args(B, SEQ_LENS, H_Q, H_KV, D, dtype, device):
    max_seq_len = max(SEQ_LENS)
    N_CTX = sum(SEQ_LENS)

    # Create random input tensors
    q = torch.randn((B * N_CTX, H_Q, D), device=device, dtype=dtype)
    k = torch.randn((B * N_CTX, H_KV, D), device=device, dtype=dtype)
    v = torch.randn((B * N_CTX, H_KV, D), device=device, dtype=dtype)
    o = torch.zeros((B * N_CTX, H_Q, D), device=device, dtype=dtype)

    # Create b_start_loc and b_seq_len tensors
    b_start_loc = torch.tensor([0, SEQ_LENS[0]], device=device)
    b_seq_len = torch.tensor(SEQ_LENS, device=device)

    return (q, k, v, o, b_start_loc, b_seq_len, max_seq_len)


# pylint: disable=unused-argument
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['B', 'SEQ_LENS', 'H_Q', 'H_KV', 'D', 'CAUSAL', 'MODE', 'VALIDATE'],
        x_vals=[  #
            [bs, [1024], 32, 8, 128, causal, 'fwd', False] for causal in [True, False] for bs in [1, 16, 32, 64, 128]
        ] + [  #
            [bs, [1024], 32, 32, 96, causal, 'fwd', False] for causal in [True, False] for bs in [1, 16, 32, 64, 128]
        ] + [  #
            [bs, [1024], 28, 4, 128, causal, 'fwd', False] for causal in [True, False] for bs in [1, 16, 32, 64, 128]
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
        plot_name='prefill-attn-performance',
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(B, SEQ_LENS, H_Q, H_KV, D, CAUSAL, MODE, VALIDATE, provider):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    N_CTX = sum(SEQ_LENS)

    q, k, v, o, b_start_loc, b_seq_len, max_seq_len = gen_args(B, SEQ_LENS, H_Q, H_KV, D, dtype, 'xpu')

    quantiles = [0.5, 0.0, 1.0]
    if provider == 'triton':

        triton_fn = lambda: context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=CAUSAL)

        if VALIDATE:
            # FIXME: torch sdpa does not support different H_Q and H_KV
            cu_seq_lens = [0] * (len(SEQ_LENS) + 1)
            for i, seq_len in enumerate(SEQ_LENS):
                cu_seq_lens[i + 1] = cu_seq_lens[i] + seq_len

            for i in range(len(SEQ_LENS)):
                start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
                o_torch = torch.nn.functional.scaled_dot_product_attention(
                    q[start:end].permute(1, 0, 2),
                    k[start:end].permute(1, 0, 2),
                    v[start:end].permute(1, 0, 2),
                    is_causal=CAUSAL,
                ).permute(1, 0, 2)

                cos_sim = torch.nn.functional.cosine_similarity(o[start:end].flatten(), o_torch.flatten(), dim=0)
                assert cos_sim.item() > 1 - (1e-5)
                assert torch.allclose(o[start:end], o_torch, atol=1e-2)

        _, min_ms, max_ms, mean_ms, cv = benchmark_suit.do_bench(triton_fn, n_warmup=10, n_repeat=10,
                                                                 quantiles=quantiles)

    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    tflops = lambda ms: 2 * B * (H_Q + H_KV) * N_CTX * N_CTX * D * (1e-12) / (ms * 1e-3)
    gbps = lambda ms: 2 * B * (H_Q + H_KV) * N_CTX * D * 2 * (1e-9) / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
