import torch

from sglang.srt.layers.attention.triton_ops.prefill_attention import context_attention_fwd

import triton_kernels_benchmark as benchmark_suit


def gen_args(B, SEQ_LENS, H_Q, H_KV, D, dtype, device):
    max_seq_len = SEQ_LENS
    N_CTX = SEQ_LENS

    # Create random input tensors
    q = torch.randn((B * N_CTX, H_Q, D), device=device, dtype=dtype)
    k = torch.randn((B * N_CTX, H_KV, D), device=device, dtype=dtype)
    v = torch.randn((B * N_CTX, H_KV, D), device=device, dtype=dtype)
    o = torch.zeros((B * N_CTX, H_Q, D), device=device, dtype=dtype)

    # Create b_start_loc and b_seq_len tensors
    b_start_loc = torch.tensor([0, SEQ_LENS], device=device)
    b_seq_len = torch.tensor([SEQ_LENS], device=device)

    return (q, k, v, o, b_start_loc, b_seq_len, max_seq_len)


def get_dtype(dtype_str: str):
    if dtype_str == 'bfloat16':
        return torch.bfloat16
    if dtype_str == 'float16':
        return torch.float16
    if dtype_str == 'float32':
        return torch.float32
    raise ValueError(f'Unsupported dtype: {dtype_str}')


X_VALS = [[bs, *sizes, causal, mode, dtype]
          for bs in [1, 16, 32, 64, 128]
          for sizes in [(1024, 32, 8, 128), (1024, 32, 32, 96), (1024, 28, 4, 128)]
          for causal in [True, False]
          for mode in ['fwd']
          for dtype in ['bfloat16']]


# pylint: disable=unused-argument
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['B', 'SEQ_LENS', 'H_Q', 'H_KV', 'D', 'CAUSAL', 'MODE', 'DTYPE'],
        x_vals=X_VALS,
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
        plot_name='sglang-prefill-attn-performance',
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(B, SEQ_LENS, H_Q, H_KV, D, CAUSAL, MODE, DTYPE, provider):
    torch.manual_seed(0)
    dtype = get_dtype(DTYPE)

    q, k, v, o, b_start_loc, b_seq_len, max_seq_len = gen_args(B, SEQ_LENS, H_Q, H_KV, D, dtype, 'xpu')

    quantiles = [0.5, 0.0, 1.0]
    if provider == 'triton' and MODE == 'fwd':
        triton_fn = lambda: context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=CAUSAL)
        _, min_ms, max_ms, mean_ms, cv = benchmark_suit.do_bench(triton_fn, n_warmup=10, n_repeat=10,
                                                                 quantiles=quantiles)
    else:
        raise NotImplementedError(f'Unsupported provider {provider} and mode {MODE}')

    N_CTX = SEQ_LENS
    tflops = lambda ms: B * N_CTX * H_Q * D * H_KV * 2 * 2 * (1e-12) / (ms * 1e-3)
    gbps = lambda ms: B * N_CTX * (H_Q + 2 * H_KV) * D * 2 * (1e-9) / (ms * 1e-3)

    return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
