# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supporst page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1
import torch

from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd, )

import triton_kernels_benchmark as benchmark_suit


def _test_context_attention_once(head_dim, is_causal):
    # Set up a simple test case
    num_heads = 4
    seq_lens = [8, 12]
    max_seq_len = max(seq_lens)

    # Create random input tensors
    q = torch.randn(sum(seq_lens), num_heads, head_dim, device='xpu')
    k = torch.randn(sum(seq_lens), num_heads, head_dim, device='xpu')
    v = torch.randn(sum(seq_lens), num_heads, head_dim, device='xpu')
    o = torch.zeros(sum(seq_lens), num_heads, head_dim, device='xpu')

    # Create b_start_loc and b_seq_len tensors
    b_start_loc = torch.tensor([0, seq_lens[0]], device='xpu')
    b_seq_len = torch.tensor(seq_lens, device='xpu')

    print(f'SeqLens: {sum(seq_lens)}, HeadDim: {head_dim}, NumHead: {num_heads}, is_causal: {is_causal}')
    context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=is_causal)

    cu_seq_lens = [0] * (len(seq_lens) + 1)
    for i, seq_len in enumerate(seq_lens):
        cu_seq_lens[i + 1] = cu_seq_lens[i] + seq_len

    for i in range(len(seq_lens)):
        start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
        o_torch = torch.nn.functional.scaled_dot_product_attention(
            q[start:end].permute(1, 0, 2),
            k[start:end].permute(1, 0, 2),
            v[start:end].permute(1, 0, 2),
            is_causal=is_causal,
        ).permute(1, 0, 2)

        cos_sim = torch.nn.functional.cosine_similarity(o[start:end].flatten(), o_torch.flatten(), dim=0)
        assert cos_sim.item() > 1 - (1e-5)
        assert torch.allclose(o[start:end], o_torch, atol=1e-2)


def test_context_attention():
    # head_dim = [128, 96, 80, 13]
    # for dim in head_dim:
    #     for is_causal in [True, False]:
    #         print(
    #             f'context attention with head_dim={dim}, is_causal={is_causal}')
    #         _test_context_attention_once(dim, is_causal)
    #         print(f'Passed')
    _test_context_attention_once(128, True)


# pylint: disable=unused-argument
@benchmark_suit.perf_report(
    benchmark_suit.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['BATCH', 'SEQ_LENS', 'Q_HEAD_NUM', 'KV_HEAD_NUM', 'HEAD_DIM', 'CAUSAL', 'MODE', 'VALIDATE'],
        x_vals=[  #
            [bs, [1024], 32, 8, 128, causal, 'fwd', True] for causal in [True, False] for bs in [1, 16, 32, 64, 128]
        ] + [  # noqa
            [bs, [1024], 32, 32, 96, causal, 'fwd', True] for causal in [True, False] for bs in [1, 16, 32, 64, 128]
        ] + [  # noqa
            [bs, [1024], 28, 4, 128, causal, 'fwd', True] for causal in [True, False] for bs in [1, 16, 32, 64, 128]
        ] + [  # noqa
            # [4, [1024], 48, 48, 64, causal, 'fwd', True] for causal in [True, False]
            # ] + [
            # [bs, [16384 // bs], h, h, dhead, causal, 'fwd', True] for bs in [1, 2, 4, 8, 16, 32] for (h, dhead) in [(16, 128), (32, 64)] for causal in [False, True]
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
def benchmark(BATCH, SEQ_LENS, Q_HEAD_NUM, KV_HEAD_NUM, HEAD_DIM, CAUSAL, MODE, VALIDATE, provider):
    dtype = torch.bfloat16
    device = 'xpu'
    N_CTX = sum(SEQ_LENS)
    max_seq_len = max(SEQ_LENS)

    # Create random input tensors
    q = torch.randn((BATCH * N_CTX, Q_HEAD_NUM, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((BATCH * N_CTX, KV_HEAD_NUM, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((BATCH * N_CTX, KV_HEAD_NUM, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
    o = torch.zeros((BATCH * N_CTX, Q_HEAD_NUM, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)

    # Create b_start_loc and b_seq_len tensors
    b_start_loc = torch.tensor([0, SEQ_LENS[0]], device=device)
    b_seq_len = torch.tensor(SEQ_LENS, device=device)

    quantiles = [0.5, 0.0, 1.0]
    if provider == 'triton':

        def triton_fn():
            return context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=CAUSAL)

        if VALIDATE:
            # FIXME: use torch sdpa for result check after https://github.com/intel/intel-xpu-backend-for-triton/issues/2042 fixed
            atol = 1e-1 if N_CTX == 16384 else 1e-2

            def torch_fn():
                return torch.nn.functional.scaled_dot_product_attention(q.cpu().permute(1, 0, 2),
                                                                        k.cpu().permute(1, 0, 2),
                                                                        v.cpu().permute(1, 0, 2),
                                                                        is_causal=CAUSAL).permute(1, 0,
                                                                                                  2).to(torch.float32)

            benchmark_suit.assert_close(triton_fn, torch_fn, atol=atol, rtol=1e-3, err_msg='triton to torch')

        _, min_ms, max_ms, mean, cv = benchmark_suit.do_bench(triton_fn, n_warmup=10, n_repeat=10, quantiles=quantiles)

    else:
        raise NotImplementedError(f'Unsupported provider {provider}')

    def tflops(ms):        return 2 * BATCH * (Q_HEAD_NUM + KV_HEAD_NUM) * \
N_CTX * N_CTX * HEAD_DIM * (1e-12) / (ms * 1e-3)

    def gbps(ms):        return 2 * BATCH * (Q_HEAD_NUM + KV_HEAD_NUM) * \
N_CTX * HEAD_DIM * 2 * (1e-9) / (ms * 1e-3)

    return (gbps(mean), gbps(max_ms), gbps(min_ms)), (tflops(mean), tflops(max_ms), tflops(min_ms)), cv


if __name__ == '__main__':
    # Functionality test
    test_context_attention()
    # # Performance test
    # benchmark.run(show_plots=False, print_data=True)
