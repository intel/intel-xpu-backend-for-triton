import torch
from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd, )
import triton_kernels_benchmark as benchmark_suit


# pylint: disable=unused-argument
def gen_args(B, EXTEND_LEN, PREFIX_LEN, H_Q, H_KV, D, dtype, device):

    b_seq_len_prefix = torch.full((B, ), PREFIX_LEN, dtype=torch.int32, device=device)
    b_seq_len_extend = torch.full((B, ), EXTEND_LEN, dtype=torch.int32, device=device)
    b_seq_len = b_seq_len_prefix + b_seq_len_extend

    b_start_loc = torch.zeros((B, ), dtype=torch.int32, device=device)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B, ), dtype=torch.int32, device=device)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    kv_indptr = torch.zeros((B + 1, ), dtype=torch.int32, device=device)
    kv_indptr[1:B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
    kv_indices = torch.zeros((b_seq_len_prefix.sum().item(), ), dtype=torch.int32, device=device)

    for i in range(B):
        kv_indices[kv_indptr[i]:kv_indptr[i + 1]] = torch.arange(b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i])

    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()
    k_buffer = torch.empty((total_token_num, H_KV, D), dtype=dtype, device=device).normal_(mean=0.1, std=0.2)
    v_buffer = torch.empty((total_token_num, H_KV, D), dtype=dtype, device=device).normal_(mean=0.1, std=0.2)

    k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
    v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
    q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)
    for i in range(B):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
        k_extend[extend_start:extend_end] = k_buffer[extend_start_in_buffer:extend_end_in_buffer]
        v_extend[extend_start:extend_end] = v_buffer[extend_start_in_buffer:extend_end_in_buffer]
        q_extend[extend_start:extend_end] = torch.empty((b_seq_len_extend[i], H_Q, D), dtype=dtype,
                                                        device=device).normal_(mean=0.1, std=0.2)

    o_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)

    b_seq_len_extend = b_seq_len - b_seq_len_prefix
    max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
    qo_indptr = torch.zeros((B + 1, ), dtype=torch.int32, device=device)
    qo_indptr[1:B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)

    params = []
    params.append((q_extend, k_extend, v_extend, o_extend))
    params.append((k_buffer, v_buffer))
    params.append((qo_indptr, kv_indptr, kv_indices, max_len_extend))
    return params


def get_dtype(dtype_str: str):
    if dtype_str == 'bfloat16':
        return torch.bfloat16
    if dtype_str == 'float16':
        return torch.float16
    if dtype_str == 'float32':
        return torch.float32
    raise ValueError(f'Unsupported dtype: {dtype_str}')


X_VALS = [[bs, *sizes, mode, dtype]
          for sizes in [(512, 1024 + 128, 32, 8, 128),  #
                        (512, 1024 + 128, 32, 32, 96),  #
                        (512, 1024 + 128, 28, 4, 128)]
          for bs in [1, 16, 32, 64, 128]
          for mode in ['fwd']
          for dtype in ['bfloat16']]


# pylint: disable=unused-argument
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['B', 'EXTEND_LEN', 'PREFIX_LEN', 'H_Q', 'H_KV', 'D', 'MODE', 'DTYPE'],
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
        plot_name='sglang-extended-attn-performance',
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(B, EXTEND_LEN, PREFIX_LEN, H_Q, H_KV, D, MODE, DTYPE, provider):
    torch.manual_seed(0)
    dtype = get_dtype(DTYPE)

    params = gen_args(B, EXTEND_LEN, PREFIX_LEN, H_Q, H_KV, D, dtype, 'xpu')
    q_extend, k_extend, v_extend, o_extend = params[0]
    k_buffer, v_buffer = params[1]
    qo_indptr, kv_indptr, kv_indices, max_len_extend = params[2]
    custom_mask = None
    mask_indptr = None

    quantiles = [0.5, 0.0, 1.0]
    if provider == 'triton' and MODE == 'fwd':
        triton_fn = lambda: extend_attention_fwd(q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, qo_indptr,
                                                 kv_indptr, kv_indices, custom_mask, True, mask_indptr, max_len_extend)
        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(triton_fn, n_warmup=10, n_repeat=10, quantiles=quantiles)

    else:
        raise NotImplementedError(f'Unsupported provider {provider} and mode {MODE}')

    N_CTX_TOTAL = PREFIX_LEN + EXTEND_LEN
    N_CTX_EXTEND = EXTEND_LEN

    tflops = lambda ms: B * (N_CTX_EXTEND + N_CTX_TOTAL) * H_Q * D * H_KV * 2 * 2 * (1e-12) / (ms * 1e-3)
    gbps = lambda ms: B * ((H_Q * N_CTX_EXTEND) + H_KV *
                           (N_CTX_EXTEND + N_CTX_TOTAL) * 2) * D * 2 * (1e-9) / (ms * 1e-3)

    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
