import torch
from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd, )
import triton_kernels_benchmark as benchmark_suit

VALIDATION_ATOL = 3e-2
VALIDATION_RTOL = 3e-2
VALIDATION_MAX_B = 2
VALIDATION_MAX_EXTEND_LEN = 32
VALIDATION_MAX_PREFIX_LEN = 128


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


def _repeat_kv_heads(x, num_q_heads):
    if x.shape[1] == num_q_heads:
        return x
    return torch.repeat_interleave(x, num_q_heads // x.shape[1], dim=1)


def _extended_attention_torch_ref(q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices):
    output = torch.empty_like(q_extend)
    num_batches = qo_indptr.numel() - 1

    for b in range(num_batches):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        q_len = q_end - q_start
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())
        prefix_indices = kv_indices[kv_start:kv_end].long()
        prefix_len = int(prefix_indices.numel())
        total_len = prefix_len + q_len

        q = q_extend[q_start:q_end].float()
        k = torch.cat([k_buffer[prefix_indices], k_extend[q_start:q_end]], dim=0)
        v = torch.cat([v_buffer[prefix_indices], v_extend[q_start:q_end]], dim=0)
        k = _repeat_kv_heads(k, q.shape[1]).float()
        v = _repeat_kv_heads(v, q.shape[1]).float()

        attn = torch.einsum("thd,shd->hts", q, k) * (q.shape[-1]**-0.5)
        q_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
        k_idx = torch.arange(total_len, device=q.device).unsqueeze(0)
        causal_mask = k_idx > (prefix_len + q_idx)
        attn = attn.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        output[q_start:q_end] = torch.einsum("hts,shd->thd", attn, v).to(output.dtype)

    return output


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
        B_val = min(B, VALIDATION_MAX_B)
        EXTEND_LEN_val = min(EXTEND_LEN, VALIDATION_MAX_EXTEND_LEN)
        PREFIX_LEN_val = min(PREFIX_LEN, VALIDATION_MAX_PREFIX_LEN)
        params_ref = gen_args(B_val, EXTEND_LEN_val, PREFIX_LEN_val, H_Q, H_KV, D, dtype, 'xpu')
        q_extend_ref, k_extend_ref, v_extend_ref, o_extend_ref = params_ref[0]
        k_buffer_ref, v_buffer_ref = params_ref[1]
        qo_indptr_ref, kv_indptr_ref, kv_indices_ref, max_len_extend_ref = params_ref[2]
        triton_ref_fn = lambda: extend_attention_fwd(q_extend_ref, k_extend_ref, v_extend_ref, o_extend_ref,
                                                     k_buffer_ref, v_buffer_ref, qo_indptr_ref, kv_indptr_ref,
                                                     kv_indices_ref, custom_mask, True, mask_indptr, max_len_extend_ref)
        torch_ref_fn = lambda: _extended_attention_torch_ref(q_extend_ref, k_extend_ref, v_extend_ref, k_buffer_ref,
                                                              v_buffer_ref, qo_indptr_ref, kv_indptr_ref,
                                                              kv_indices_ref)
        benchmark_suit.assert_close(
            triton_ref_fn, torch_ref_fn, atol=VALIDATION_ATOL, rtol=VALIDATION_RTOL, err_msg='extended_attention')
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
