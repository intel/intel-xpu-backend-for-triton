#!/usr/bin/env python3

import argparse

import torch

from triton_kernels_benchmark.flash_attention_benchmark import _attention


def get_options():
    """Gather CL options."""
    parser = argparse.ArgumentParser(prog='flash-attention', description='Run Intel XPU Flash-Attention implementation')
    parser.add_argument('-Z', type=int, required=True, help='Batch size')
    parser.add_argument('-H', type=int, required=True, help='Head count')
    parser.add_argument('-N-CTX', type=int, required=True, help='Sequence length')
    parser.add_argument('-D-HEAD', type=int, required=True, help='Embedding dimension')
    parser.add_argument('-causal', action='store_true', help='Run causal attention')
    parser.add_argument('-backward', action='store_true', help='Run backward attention')
    return parser.parse_args()


def run(options):
    """Run the XPU backend FlashAttention benchmark implementation."""
    dtype = torch.float16
    q = torch.randn((options.Z, options.H, options.N_CTX, options.D_HEAD), device='xpu', dtype=dtype,
                    requires_grad=True)
    k = torch.randn_like(q, device='xpu', dtype=dtype, requires_grad=True)
    v = torch.randn_like(q, device='xpu', dtype=dtype, requires_grad=True)
    sm_scale = 0.125
    attention = _attention.apply
    triton_o = attention(q, k, v, options.causal, sm_scale)
    if options.backward:
        triton_o.backward(torch.randn_like(triton_o), retain_graph=True)


if __name__ == '__main__':
    run(get_options())
