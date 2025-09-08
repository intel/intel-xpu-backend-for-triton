import torch
import triton


def compute_grid_size(routing_data, m, n, block_m, block_n):
    if routing_data is not None:
        grid_m = routing_data.n_blocks(m, block_m)
    else:
        grid_m = triton.cdiv(m, block_m)
    grid_n = (n + block_n - 1) // block_n
    return grid_m * grid_n


def compute_block_n(n: int):
    # block_n:
    return max(16, min(128, triton.next_power_of_2(n)))


def compute_block_k(k: int | None, is_persistent: bool, precision_config):
    if k is not None:
        block_k = max(32, min(128, triton.next_power_of_2(k)))
    has_mx_weight_scale = precision_config is not None and precision_config.weight_scale is not None
    if is_persistent and has_mx_weight_scale:
        block_k = min(block_k, 128)
    return block_k


def compute_split_k(block_k: int, k: int | None, grid_size: int) -> int:
    device_props = torch.xpu.get_device_properties(0)
    n_sms = device_props.multi_processor_countgpu_subslice_count
    split_k = n_sms // grid_size
    if k is not None:
        # avoid split_k for small k
        num_block_k = triton.cdiv(k, block_k)
        split_k = min(split_k, num_block_k // 4)
    split_k = max(split_k, 1)
    return split_k


def compute_num_warps(block_m, block_n):
    return max(block_m * block_n // 4096, 4)
