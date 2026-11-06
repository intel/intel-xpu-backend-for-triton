import triton
from typing import List
DEVICE="xpu"
def get_matmul_autotune_configs() -> List[triton.Config]:
    if DEVICE == 'cuda':
        return get_cuda_matmul_autotune_configs()
    configs = [triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256'}, num_stages=s, num_warps=32) for s in [1, 2, 3]] + [triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': m}, num_stages=s, num_warps=w) for s in [2, 3, 4] for (m, w) in [('256', 32), ('128', 64)]] + [triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256'}, num_stages=s, num_warps=32) for s in [2]] + [triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': m}, num_stages=s, num_warps=w) for s in [2, 3] for (m, w) in [('256', 32), ('128', 64)]]
    configs += [triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'grf_mode': '128'}, num_stages=2, num_warps=4), triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1, 'grf_mode': '256'}, num_stages=2, num_warps=4)]
    return configs
