import torch
from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd,
    redundant_attention,
)
import triton_kernels_benchmark as benchmark_suit


def gen_args(BATCH, N_CTX, Q_HEAD_NUM, KV_HEAD_NUM, HEAD_DIM, dtype, device):

    b_seq_len_prefix = torch.randint(1, N_CTX // 2, (BATCH, ), dtype=torch.int32, device=device)
    b_seq_len_extend = torch.randint(1, N_CTX // 2, (BATCH, ), dtype=torch.int32, device=device)
    b_seq_len = b_seq_len_prefix + b_seq_len_extend
    max_len_in_batch = torch.max(b_seq_len, 0)[0].item()

    b_req_idx = torch.arange(BATCH, dtype=torch.int32, device=device)
    b_start_loc = torch.zeros((BATCH, ), dtype=torch.int32, device=device)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((BATCH, ), dtype=torch.int32, device=device)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    kv_indptr = torch.zeros((BATCH + 1, ), dtype=torch.int32, device=device)
    kv_indptr[1:BATCH + 1] = torch.cumsum(b_seq_len_prefix[:BATCH], dim=0)
    kv_indices = torch.zeros((b_seq_len_prefix.sum().item(), ), dtype=torch.int32, device=device)

    for i in range(BATCH):
        kv_indices[kv_indptr[i]:kv_indptr[i + 1]] = torch.arange(b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i])

    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()
    k_buffer = torch.empty((total_token_num, KV_HEAD_NUM, HEAD_DIM), dtype=dtype,
                           device=device).normal_(mean=0.1, std=0.2)
    v_buffer = torch.empty((total_token_num, KV_HEAD_NUM, HEAD_DIM), dtype=dtype,
                           device=device).normal_(mean=0.1, std=0.2)

    k_extend = torch.empty((extend_token_num, KV_HEAD_NUM, HEAD_DIM), dtype=dtype, device=device)
    v_extend = torch.empty((extend_token_num, KV_HEAD_NUM, HEAD_DIM), dtype=dtype, device=device)
    q_extend = torch.empty((extend_token_num, Q_HEAD_NUM, HEAD_DIM), dtype=dtype, device=device)
    for i in range(BATCH):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
        k_extend[extend_start:extend_end] = k_buffer[extend_start_in_buffer:extend_end_in_buffer]
        v_extend[extend_start:extend_end] = v_buffer[extend_start_in_buffer:extend_end_in_buffer]
        q_extend[extend_start:extend_end] = torch.empty((b_seq_len_extend[i], Q_HEAD_NUM, HEAD_DIM), dtype=dtype,
                                                        device=device).normal_(mean=0.1, std=0.2)

    o_extend = torch.empty((extend_token_num, Q_HEAD_NUM, HEAD_DIM), dtype=dtype, device=device)
    o_redundant = torch.empty((extend_token_num, Q_HEAD_NUM, HEAD_DIM), dtype=dtype, device=device)

    b_seq_len_extend = b_seq_len - b_seq_len_prefix
    max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
    qo_indptr = torch.zeros((BATCH + 1, ), dtype=torch.int32, device=device)
    qo_indptr[1:BATCH + 1] = torch.cumsum(b_seq_len_extend[:BATCH], dim=0)

    params = []
    params.append((q_extend, k_extend, v_extend, o_extend, o_redundant))
    params.append((k_buffer, v_buffer))
    params.append((qo_indptr, kv_indptr, kv_indices, max_len_extend))
    params.append((b_req_idx, b_start_loc, b_seq_len, b_seq_len_prefix, max_len_in_batch))
    return params


# pylint: disable=unused-argument
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['B', 'SEQ_LENS', 'H_Q', 'H_KV', 'D', 'MODE', 'VALIDATE'],
        x_vals=[  #
            [bs, [1024, 128, 512], 32, 8, 128, 'fwd', True] for bs in [1, 16, 32, 64, 128]
        ] + [  #
            [bs, [1024, 128, 512], 32, 32, 96, 'fwd', True] for bs in [1, 16, 32, 64, 128]
        ] + [  #
            [bs, [1024, 128, 512], 28, 4, 128, 'fwd', True] for bs in [1, 16, 32, 64, 128]
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
        plot_name='extended-attn-performance',
        # name for the plot. Used also as a file name for saving the plot.
        args={},
    ))
def benchmark(B, SEQ_LENS, H_Q, H_KV, D, MODE, VALIDATE, provider):
    torch.manual_seed(0)

    dtype = torch.bfloat16
    N_CTX = sum(SEQ_LENS)

    params = gen_args(B, N_CTX, H_Q, H_KV, D, dtype, 'xpu')
    q_extend, k_extend, v_extend, o_extend, o_redundant = params[0]
    k_buffer, v_buffer = params[1]
    qo_indptr, kv_indptr, kv_indices, max_len_extend = params[2]
    b_req_idx, b_start_loc, b_seq_len, b_seq_len_prefix, max_len_in_batch = params[3]
    custom_mask = None
    mask_indptr = None

    quantiles = [0.5, 0.0, 1.0]
    if provider == 'triton':

        def triton_fn():
            extend_attention_fwd(q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer, qo_indptr, kv_indptr,
                                 kv_indices, custom_mask, mask_indptr, max_len_extend)
            return o_extend

        if VALIDATE:

            def refer_fn():
                redundant_attention(q_extend, o_redundant, k_buffer, v_buffer, b_req_idx, b_start_loc, b_seq_len,
                                    b_seq_len_prefix, max_len_in_batch)
                return o_redundant

            benchmark_suit.assert_close(triton_fn, refer_fn, atol=1e-3, rtol=1e-2, err_msg='extend to refer')

        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(triton_fn, n_warmup=10, n_repeat=10, quantiles=quantiles)

    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    tflops = lambda ms: 2 * B * (H_Q + H_KV * N_CTX) * N_CTX * D * (1e-12) / (ms * 1e-3)
    gbps = lambda ms: 2 * B * (H_Q + H_KV * N_CTX) * D * 2 * (1e-9) / (ms * 1e-3)

    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    benchmark.run(show_plots=False, print_data=True)
