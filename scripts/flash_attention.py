#!/usr/bin/env python3

import argparse

from triton_kernels_benchmark.flash_attention_benchmark import benchmark


def get_options():
    parser = argparse.ArgumentParser(prog='flash-attention', description='Run Intel XPU Flash-Attention implementation')
    parser.add_argument('-Z', type=int, required=True, help='Batch size')
    parser.add_argument('-H', type=int, required=True, help='Head count')
    parser.add_argument('-N-CTX', type=int, required=True, help='Sequence length')
    parser.add_argument('-D-HEAD', type=int, required=True, help='Embedding dimension')
    parser.add_argument('-causal', action='store_true', help='Run causal attention')
    parser.add_argument('-backward', action='store_true', help='Run backward attention')
    return parser.parse_args()


if __name__ == '__main__':
    options = get_options()
    PROVIDER = 'triton'
    MODE = 'bwd' if options.backward else 'fwd'
    benchmark.fn(options.Z, options.H, options.N_CTX, options.D_HEAD, options.causal, MODE, PROVIDER)
