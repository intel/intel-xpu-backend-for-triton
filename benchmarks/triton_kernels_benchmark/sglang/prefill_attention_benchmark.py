import torch

from sglang.srt.layers.attention.triton_ops.prefill_attention import context_attention_fwd

import triton_kernels_benchmark as benchmark_suit

VALIDATION_ATOL = 3e-2
VALIDATION_RTOL = 3e-2
VALIDATION_MAX_B = 2
VALIDATION_MAX_SEQ_LENS = 64


def gen_args(B, SEQ_LENS, H_Q, H_KV, D, dtype, device):
    max_seq_len = SEQ_LENS
    N_CTX = SEQ_LENS

    # Create random input tensors
    q = torch.randn((B * N_CTX, H_Q, D), device=device, dtype=dtype)
    k = torch.randn((B * N_CTX, H_KV, D), device=device, dtype=dtype)
    v = torch.randn((B * N_CTX, H_KV, D), device=device, dtype=dtype)
    o = torch.zeros((B * N_CTX, H_Q, D), device=device, dtype=dtype)

    # Create per-batch metadata for flattened [B * N_CTX, ...] layout
    b_start_loc = torch.arange(0, (B + 1) * SEQ_LENS, SEQ_LENS, dtype=torch.int32, device=device)
    b_seq_len = torch.full((B, ), SEQ_LENS, dtype=torch.int32, device=device)

    return (q, k, v, o, b_start_loc, b_seq_len, max_seq_len)


def _repeat_kv_heads(x, num_q_heads):
    if x.shape[1] == num_q_heads:
        return x
    return torch.repeat_interleave(x, num_q_heads // x.shape[1], dim=1)


def _prefill_attention_torch_ref(q, k, v, b_start_loc, b_seq_len, is_causal):
    output = torch.empty_like(q)
    for b in range(b_seq_len.numel()):
        start = int(b_start_loc[b].item())
        seq_len = int(b_seq_len[b].item())
        q_slice = q[start:start + seq_len].float()
        k_slice = _repeat_kv_heads(k[start:start + seq_len], q.shape[1]).float()
        v_slice = _repeat_kv_heads(v[start:start + seq_len], q.shape[1]).float()
        attn = torch.einsum('thd,shd->hts', q_slice, k_slice) * (q_slice.shape[-1]**-0.5)
        if is_causal:
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device), diagonal=1)
            attn = attn.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        output[start:start + seq_len] = torch.einsum('hts,shd->thd', attn, v_slice).to(output.dtype)
    return output


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
        B_val = min(B, VALIDATION_MAX_B)
        SEQ_LENS_val = min(SEQ_LENS, VALIDATION_MAX_SEQ_LENS)
        q_ref, k_ref, v_ref, o_ref, b_start_loc_ref, b_seq_len_ref, max_seq_len_ref = gen_args(
            B_val, SEQ_LENS_val, H_Q, H_KV, D, dtype, 'xpu')
        triton_ref_fn = lambda: context_attention_fwd(q_ref, k_ref, v_ref, o_ref, b_start_loc_ref, b_seq_len_ref,
                                                      max_seq_len_ref, is_causal=CAUSAL)
        torch_ref_fn = lambda: _prefill_attention_torch_ref(q_ref, k_ref, v_ref, b_start_loc_ref, b_seq_len_ref, CAUSAL)
        benchmark_suit.assert_close(triton_ref_fn, torch_ref_fn, atol=VALIDATION_ATOL, rtol=VALIDATION_RTOL,
                                    err_msg='prefill_attention')
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
