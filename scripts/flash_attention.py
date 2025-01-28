#!/usr/bin/env python3

import argparse

import torch
import triton

from triton_kernels_benchmark.flash_attention_benchmark import _attention, tune_attn_fwd


def get_options():
    """Gather CL options."""
    parser = argparse.ArgumentParser(prog='flash-attention', description='Run Intel XPU Flash-Attention implementation')

    model = parser.add_argument_group(title='Model description',
                                      description='Options setting different model metaparameters')
    model.add_argument('-Z', type=int, required=True, help='Batch size')
    model.add_argument('-H', type=int, required=True, help='Head count')
    model.add_argument('-N-CTX', type=int, required=True, help='Sequence length')
    model.add_argument('-D-HEAD', type=int, required=True, help='Embedding dimension')
    model.add_argument('-causal', action='store_true', help='Run causal attention')
    model.add_argument('-backward', action='store_true', help='Run backward attention')

    config = parser.add_argument_group(title='Tuning configuration',
                                       description='Options setting different tuning parameters')
    config.add_argument('-BLOCK-M', action='extend', nargs='+', type=int, help='Sizes of m')
    config.add_argument('-BLOCK-N', action='extend', nargs='+', type=int, help='Size of N')
    config.add_argument('-stages', action='extend', nargs='+', type=int, help='Numbers of stages')
    config.add_argument('-warps', action='extend', nargs='+', type=int, help='Numbers of warps')
    return parser.parse_args()


def get_configs(options):
    """Get autotuning configurations."""
    bm_values = options.BLOCK_M if options.BLOCK_M else [128, 256]
    bn_values = options.BLOCK_N if options.BLOCK_N else [32, 64]
    stages_values = options.stages if options.stages else [3, 4]
    warps_values = options.warps if options.warps else [8, 16, 32]
    return [
        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'grf_mode': 'large', 'one_matrix_per_load_for_bt': True},
                      num_stages=s, num_warps=w)
        for BM in bm_values
        for BN in bn_values
        for s in stages_values
        for w in warps_values
    ]


def run(options):
    """Run the XPU backend FlashAttention benchmark implementation."""
    dtype = torch.float16
    q = torch.randn((options.Z, options.H, options.N_CTX, options.D_HEAD), device='xpu', dtype=dtype,
                    requires_grad=True)
    k = torch.randn_like(q, device='xpu', dtype=dtype, requires_grad=True)
    v = torch.randn_like(q, device='xpu', dtype=dtype, requires_grad=True)
    sm_scale = 0.125

    tune_attn_fwd.configs = get_configs(options)

    attention = _attention.apply
    triton_o = attention(q, k, v, options.causal, sm_scale)
    if options.backward:
        triton_o.backward(torch.randn_like(triton_o), retain_graph=True)


if __name__ == '__main__':
    run(get_options())
