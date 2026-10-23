import torch
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
from torch._C import _xpu_getCurrentRawStream as get_raw_stream

triton_helpers.set_driver_to_gpu()


@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 16}, reduction_hint=ReductionHint.DEFAULT, filename=__file__, triton_meta={
        'signature':
        {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device':
        DeviceProperties(
            type='xpu', index=0, multi_processor_count=32, cc={
                'architecture': 37627101184, 'device_id': 26447, 'driver_version': '1.17.39576', 'gpu_eu_count': 256,
                'gpu_subslice_count': 32, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True,
                'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True,
                'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'is_integrated_gpu': False,
                'last_level_cache_size': 33554432, 'local_mem_size': 393216, 'max_compute_units': 256,
                'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'memory_bus_width': 64, 'memory_clock_rate': 0,
                'name': 'Intel(R) Graphics [0x674f]', 'platform_name':
                'Intel(R) oneAPI Unified Runtime over Level-Zero V2', 'sub_group_sizes': [16, 32], 'total_memory':
                130555052032, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '35.11.0'
            }, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None,
            max_threads_per_block=1024, warp_size=None), 'constants': {}, 'native_matmul':
        False, 'enable_fp_fusion':
        True, 'launch_pdl':
        False, 'disable_ftz':
        False, 'configs': [{(0, ): [['tt.divisibility', 16]], (2, ): [['tt.divisibility', 16]]}]
    }, inductor_meta={
        'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_stack_sum_30', 'mutated_arg_names': [], 'optimize_mem':
        True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1,
        'autotune_hints': set(), 'tiling_scores': {'x': 212992, 'r0_': 0}, 'backend_hash':
        'EAACE1462B0B6D9C31A22CEF2406FC7AABD9D3FC0277FB9D93C2A3C9091E0511', 'assert_indirect_indexing': True,
        'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches':
        False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False,
        'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False,
        'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False,
        'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True,
        'are_deterministic_algorithms_enabled': False
    })
@triton.jit
def triton_per_fused_stack_sum_30(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 8192
    r0_numel = 11
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + (x3 + 8192 * r0_2), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0 + 976 * x1), tmp4, None)


def run_standalone():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
        with torch.xpu._DeviceGuard(0):
            torch.xpu.set_device(0)
            stream = get_raw_stream(0)
    elif torch.cuda.is_available():
        device = "cuda"
        stream = torch.cuda.current_stream().cuda_stream
    else:
        raise RuntimeError("No XPU/CUDA available")

    # IMPORTANT: the kernel stores output as fp16, while it sums in fp32.
    # Using a full arange(8192*11) makes each row sum exceed fp16 max (~65504),
    # so the output saturates to inf. Keep the values small enough for fp16 output.
    # Match the actual Inductor graph in the debug output:
    #   %view_61 : Tensor "f16[11, 256, 32][8192, 32, 1]xpu:0"
    #   %sum_5   : Tensor "f16[256, 32][32, 1]xpu:0" = sum(view_61, dim=0)
    # The reduction dimension is the leading axis, and the output is stored in a
    # packed [256, 976] buffer with the valid region at [:, :32].
    x = torch.arange(11 * 256 * 32, device=device, dtype=torch.float16).reshape(11, 256, 32).contiguous()
    flat_x = x.reshape(-1)
    out = torch.empty(256 * 976, device=device, dtype=torch.float16)

    XBLOCK = 128
    grid = (triton.cdiv(x.shape[0], XBLOCK), )

    # Use the raw stream handle that the generated Triton launcher expects.
    triton_per_fused_stack_sum_30.run(
        flat_x,
        out,
        flat_x.numel(),
        11,
        stream=stream,
    )

    if device == "xpu":
        torch.xpu.synchronize()
    else:
        torch.cuda.synchronize()

    got = out.view(256, 976)[:, :32].to(torch.float32)
    ref = x.sum(dim=0).to(torch.float32)

    print('kernel sample:', got[:2, :8])
    print('reference sample:', ref[:2, :8])
    print('matches:', torch.allclose(got, ref, rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    run_standalone()
