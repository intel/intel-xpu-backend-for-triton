# pylint: skip-file
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Based on this commit:
# b5545d9d5cab2625ac04a19f552631a2034c8f47
# https://github.com/vllm-project/vllm/blob/b5545d9d5cab2625ac04a19f552631a2034c8f47/vllm/attention/ops/triton_unified_attention.py
"""
Unified Attention benchmark
==========================

This benchmark is based on the test_triton_unified_attention.py tests
"""
import os
import atexit
import csv
import sys
from itertools import product
from typing import Optional

import torch

import triton_kernels_benchmark as benchmark_suite
from triton_kernels_benchmark.benchmark_testing import BENCHMARKING_CONFIG

# This supports both current upstream and pinned version
try:
    from vllm.attention.ops.triton_unified_attention import unified_attention
except ImportError:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
except ImportError as e:
    raise ImportError(
        "Could not import unified_attention from vLLM. Please ensure vLLM is installed and accessible.") from e
from vllm.platforms import current_platform
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func as sycl_tla_attention

_unified_attention_module = sys.modules[unified_attention.__module__]
_original_unified_attention = unified_attention
_LAST_UNIFIED_ATTENTION_LOCALS: dict[str, object] = {}


def _trace_unified_attention(frame, event, arg):
    if event == 'return' and frame.f_code is _original_unified_attention.__code__:
        _LAST_UNIFIED_ATTENTION_LOCALS.clear()
        _LAST_UNIFIED_ATTENTION_LOCALS.update(frame.f_locals)
    return _trace_unified_attention


def _recording_unified_attention(*args, **kwargs):
    previous_trace = sys.gettrace()
    _LAST_UNIFIED_ATTENTION_LOCALS.clear()
    sys.settrace(_trace_unified_attention)
    try:
        return _original_unified_attention(*args, **kwargs)
    finally:
        sys.settrace(previous_trace)


unified_attention = _recording_unified_attention

float8_info = torch.finfo(current_platform.fp8_dtype())

TOTAL_MEMORY_BYTES = benchmark_suite.get_total_gpu_memory_bytes()

IS_FP8 = (os.getenv('FP8', '0') == '1')
AUTOTUNE_DECISIONS_FILE = 'unified-attention-autotune-decisions.csv'
AUTOTUNE_DECISION_ROWS: list[dict[str, object]] = []
AUTOTUNE_KEY_COLUMNS = [
    'BLOCK_SIZE',
    'HEAD_SIZE',
    'IS_FP8_INPUT',
    'IS_3D',
    'BLOCK_Q',
    'num_queries_per_kv',
    'TILE_SIZE',
    'BLOCK_M',
    'NUM_SEQS_BUCKET',
    'MAX_SEQLEN_K_BUCKET',
    'EFFECTIVE_MAX_SEQLEN_K_BUCKET',
]
AUTOTUNE_DECISION_FIELDNAMES = [
    'benchmark_name',
    'td_patched',
    'benchmark_dtype',
    'provider',
    'q_heads',
    'k_heads',
    'head_size',
    'qdtype',
    'seq_lens',
    'sliding_window',
    'soft_cap',
    'num_blocks',
    'block_size',
    'is_prefill',
    'autotune_key_names',
    'autotune_key_values',
    'key_BLOCK_SIZE',
    'key_HEAD_SIZE',
    'key_IS_FP8_INPUT',
    'key_IS_3D',
    'key_BLOCK_Q',
    'key_num_queries_per_kv',
    'key_TILE_SIZE',
    'key_BLOCK_M',
    'key_NUM_SEQS_BUCKET',
    'key_MAX_SEQLEN_K_BUCKET',
    'key_EFFECTIVE_MAX_SEQLEN_K_BUCKET',
    'selected_config',
    'selected_TILE_SIZE',
    'selected_BLOCK_Q',
    'actual_TILE_SIZE',
    'actual_BLOCK_Q',
    'actual_num_warps',
    'actual_num_stages',
    'selected_NUM_SEGMENTS_PER_SEQ',
    'selected_num_warps',
    'selected_num_stages',
    'configs_after_prune',
    'autotune_cache_size',
    'mean_ms',
    'cv',
]


def _reports_dir_from_argv() -> str:
    reports_dir = os.getenv('UA_AUTOTUNE_DECISION_REPORTS', '')
    if reports_dir:
        return reports_dir

    argv = sys.argv[1:]
    for idx, arg in enumerate(argv):
        if arg == '--reports' and idx + 1 < len(argv):
            reports_dir = argv[idx + 1]
        elif arg.startswith('--reports='):
            reports_dir = arg.split('=', 1)[1]
    return reports_dir


def _format_csv_value(value: object) -> str:
    if value is None:
        return ''
    if isinstance(value, (list, tuple, dict)):
        return repr(value)
    return str(value)


def _config_kwargs(config: object) -> dict[str, object]:
    if config is None:
        return {}
    kwargs = dict(getattr(config, 'kwargs', {}) or {})
    for attr in ('num_warps', 'num_stages', 'num_ctas'):
        value = getattr(config, attr, None)
        if value is not None:
            kwargs[attr] = value
    return kwargs


def _format_config(config: object) -> str:
    if isinstance(config, str):
        return config
    kwargs = _config_kwargs(config)
    return ';'.join(f'{key}={_format_csv_value(kwargs[key])}' for key in sorted(kwargs))


def _config_value(config: object, key: str) -> object:
    return _config_kwargs(config).get(key, '')


def _observed_value(*names: str) -> object:
    for name in names:
        if name in _LAST_UNIFIED_ATTENTION_LOCALS:
            return _LAST_UNIFIED_ATTENTION_LOCALS[name]
    return ''


def _actual_tile_size(config: object) -> object:
    config_value = _config_value(config, 'TILE_SIZE')
    if config_value != '':
        return config_value

    observed_tile_size = _observed_value('tile_size', 'TILE_SIZE')
    if observed_tile_size != '':
        return observed_tile_size

    use_3d = _LAST_UNIFIED_ATTENTION_LOCALS.get('use_3d')
    if use_3d is True:
        return _observed_value('TILE_SIZE_DECODE')
    if use_3d is False:
        return _observed_value('TILE_SIZE_PREFILL')
    return ''


def _int_value(value: object) -> int | None:
    if value == '' or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _actual_block_q(config: object, q_heads: int, k_heads: int) -> object:
    config_value = _config_value(config, 'BLOCK_Q')
    if config_value != '':
        return config_value

    observed = _observed_value('BLOCK_Q')
    if observed != '':
        return observed

    block_m = _int_value(_actual_block_m(config))
    num_queries = _int_value(_num_queries_per_kv(q_heads, k_heads))
    if block_m is not None and num_queries not in (None, 0):
        return block_m // num_queries
    return ''


def _actual_config_value(config: object, key: str, default: object = '') -> object:
    config_value = _config_value(config, key)
    if config_value != '':
        return config_value

    observed = _observed_value(key)
    if observed != '':
        return observed
    return default


def _is_fp8_input(qdtype: torch.dtype | None) -> bool:
    observed = _observed_value('is_fp8_input', 'IS_FP8_INPUT', 'Q_IS_FP8')
    if observed != '':
        return bool(observed)
    return qdtype is not None and getattr(qdtype, 'itemsize', 0) == 1


def _is_3d() -> object:
    return _observed_value('use_3d', 'IS_3D')


def _num_queries_per_kv(q_heads: int, k_heads: int) -> object:
    observed = _observed_value('num_queries_per_kv')
    if observed != '':
        return observed
    return q_heads // k_heads


def _actual_block_m(config: object) -> object:
    config_value = _config_value(config, 'BLOCK_M')
    if config_value != '':
        return config_value
    return _observed_value('BLOCK_M')


def _autotune_key_values(qdtype: torch.dtype | None, head_size: int, seq_lens: list[tuple[int, int]],
                         block_size: int, config: object, q_heads: int, k_heads: int) -> dict[str, object]:
    return {
        'BLOCK_SIZE': block_size,
        'HEAD_SIZE': head_size,
        'IS_FP8_INPUT': _is_fp8_input(qdtype),
        'IS_3D': _is_3d(),
        'BLOCK_Q': _actual_block_q(config, q_heads, k_heads),
        'num_queries_per_kv': _num_queries_per_kv(q_heads, k_heads),
        'TILE_SIZE': _actual_tile_size(config),
        'BLOCK_M': _actual_block_m(config),
        'NUM_SEQS_BUCKET': _observed_value('NUM_SEQS_BUCKET', 'num_seqs_bucket'),
        'MAX_SEQLEN_K_BUCKET': _observed_value('MAX_SEQLEN_K_BUCKET', 'max_seqlen_k_bucket'),
        'EFFECTIVE_MAX_SEQLEN_K_BUCKET': _observed_value(
            'EFFECTIVE_MAX_SEQLEN_K_BUCKET', 'effective_max_seqlen_k_bucket'
        ),
    }


def _get_pruned_configs(kernel: object, meta: dict[str, object]) -> list[object]:
    if kernel is None or not hasattr(kernel, 'prune_configs'):
        return []

    old_nargs = getattr(kernel, 'nargs', None)
    try:
        kernel.nargs = {}
        return list(kernel.prune_configs(dict(meta)))
    except Exception as exc:  # pragma: no cover - diagnostic path only
        return [f'error={type(exc).__name__}:{exc}']
    finally:
        try:
            kernel.nargs = old_nargs
        except Exception:
            pass


def _autotune_cache(kernel: object, is_3d: object) -> object:
    if is_3d is True:
        cache = getattr(_unified_attention_module, 'UNIFIED_ATTENTION_3D_TILE_CACHE', None)
        if cache is not None:
            return cache
    return getattr(kernel, 'cache', {}) or {}


def _record_autotune_decision(
    *,
    benchmark_name: str,
    provider: str,
    q_heads: int,
    k_heads: int,
    head_size: int,
    qdtype: torch.dtype | None,
    seq_lens: list[tuple[int, int]],
    sliding_window: int | None,
    soft_cap: float | None,
    num_blocks: int,
    block_size: int,
    mean_ms: float,
    cv: float,
) -> None:
    kernel = getattr(_unified_attention_module, 'kernel_unified_attention', None)
    is_3d = _is_3d()
    selected_config = None
    if is_3d is True:
        selected_config = getattr(_unified_attention_module, 'LAST_UNIFIED_ATTENTION_3D_CONFIG', None)
    if selected_config is None and kernel is not None:
        selected_config = getattr(kernel, 'best_config', None)
    if selected_config is None:
        selected_config = ''

    key_names = list(getattr(kernel, 'keys', []) or []) if kernel is not None else []
    key_values = _autotune_key_values(qdtype, head_size, seq_lens, block_size, selected_config, q_heads, k_heads)
    pruned_configs = _get_pruned_configs(kernel, key_values)
    autotune_cache = _autotune_cache(kernel, is_3d)
    selected_num_segments = _config_value(selected_config, 'NUM_SEGMENTS_PER_SEQ')
    if is_3d is True and selected_num_segments == '':
        selected_num_segments = _observed_value('num_segments', 'num_par_softmax_segments')

    row = {
        'benchmark_name': benchmark_name,
        'td_patched': os.getenv('TD_PATCHED', '0'),
        'benchmark_dtype': 'fp8' if IS_FP8 else 'bf16',
        'provider': provider,
        'q_heads': q_heads,
        'k_heads': k_heads,
        'head_size': head_size,
        'qdtype': _format_csv_value(qdtype),
        'seq_lens': _format_csv_value(seq_lens),
        'sliding_window': _format_csv_value(sliding_window),
        'soft_cap': _format_csv_value(soft_cap),
        'num_blocks': num_blocks,
        'block_size': block_size,
        'is_prefill': max(query_len for query_len, _ in seq_lens) > 1,
        'autotune_key_names': ';'.join(key_names),
        'autotune_key_values': _format_csv_value({key: key_values.get(key, '') for key in key_names}),
        'selected_config': _format_config(selected_config),
        'selected_TILE_SIZE': _config_value(selected_config, 'TILE_SIZE'),
        'selected_BLOCK_Q': _config_value(selected_config, 'BLOCK_Q'),
        'actual_TILE_SIZE': _actual_tile_size(selected_config),
        'actual_BLOCK_Q': _actual_block_q(selected_config, q_heads, k_heads),
        'actual_num_warps': _actual_config_value(selected_config, 'num_warps', 4),
        'actual_num_stages': _actual_config_value(selected_config, 'num_stages', 2),
        'selected_NUM_SEGMENTS_PER_SEQ': selected_num_segments,
        'selected_num_warps': _config_value(selected_config, 'num_warps'),
        'selected_num_stages': _config_value(selected_config, 'num_stages'),
        'configs_after_prune': '|'.join(_format_config(config) for config in pruned_configs),
        'autotune_cache_size': len(autotune_cache),
        'mean_ms': mean_ms,
        'cv': cv,
    }
    for key in AUTOTUNE_KEY_COLUMNS:
        row[f'key_{key}'] = key_values.get(key, '')
    AUTOTUNE_DECISION_ROWS.append(row)


def _write_autotune_decisions() -> None:
    if not AUTOTUNE_DECISION_ROWS:
        return

    reports_dir = _reports_dir_from_argv()
    if not reports_dir:
        return

    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, AUTOTUNE_DECISIONS_FILE)
    write_header = not os.path.exists(report_path) or os.path.getsize(report_path) == 0
    with open(report_path, 'a', newline='', encoding='utf-8') as report_file:
        writer = csv.DictWriter(report_file, fieldnames=AUTOTUNE_DECISION_FIELDNAMES, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerows(AUTOTUNE_DECISION_ROWS)
    print(f'Wrote unified attention autotune decisions to {report_path}')
    AUTOTUNE_DECISION_ROWS.clear()


atexit.register(_write_autotune_decisions)


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len] * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (torch.triu(empty_mask,
                                              diagonal=kv_len - (query_len + sliding_window) + 1).bool().logical_not())
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        torch.softmax(attn, dim=-1, out=attn)
        attn = attn.to(v.dtype)  # cast in a second step to reduce peak memory usage
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def _dtype_size(dtype):
    """Return element size in bytes for a torch dtype."""
    if dtype is None:
        return 0
    if hasattr(dtype, 'itemsize'):
        return dtype.itemsize
    # Fallback for dtype objects without itemsize
    return torch.tensor([], dtype=dtype).element_size()


def is_enough_memory(x_val, safety_factor=0.80):
    """
    Check whether all tensors created by the benchmark fit in GPU memory.

    Mirrors every allocation inside ``benchmark()`` and the expected peak memory
    usage of ``ref_paged_attn()``. Returns *True* when the total fits within
    ``safety_factor`` of the device's total memory.
    """
    q_heads, k_heads, head_size, qdtype, seq_lens, sliding_window, soft_cap, num_blocks, block_size = x_val
    dtype = torch.bfloat16

    num_seqs = len(seq_lens)
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    total_query_tokens = sum(query_lens)
    max_kv_len = max(kv_lens)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size

    d_bytes = _dtype_size(dtype)  # e.g. 2 for bfloat16

    # --- tensors allocated in benchmark() ---
    # query:       (total_query_tokens, q_heads, head_size)  dtype
    query_mem = total_query_tokens * q_heads * head_size * d_bytes
    # key_cache:   (num_blocks, block_size, k_heads, head_size) dtype
    kv_cache_mem = 2 * num_blocks * block_size * k_heads * head_size * d_bytes
    # output:      same shape/dtype as query  (empty_like)
    output_mem = query_mem
    # cu_query_lens: (num_seqs + 1,) int32
    cu_mem = (num_seqs + 1) * 4
    # kv_lens_tensor: (num_seqs,) int32
    kvl_mem = num_seqs * 4
    # block_tables: (num_seqs, max_num_blocks_per_seq) int32
    bt_mem = num_seqs * max_num_blocks_per_seq * 4

    # quantised duplicates (when qdtype is set the originals are *kept*)
    q_quant_mem = 0
    if qdtype is not None:
        qd_bytes = _dtype_size(qdtype)
        # maybe_quantized_query, _key_cache, _value_cache
        q_quant_mem += total_query_tokens * q_heads * head_size * qd_bytes
        q_quant_mem += 2 * num_blocks * block_size * k_heads * head_size * qd_bytes
        # k_descale, v_descale: (num_seqs, k_heads) float32
        q_quant_mem += 2 * num_seqs * k_heads * 4

    triton_memory = (query_mem + kv_cache_mem + output_mem + cu_mem + kvl_mem + bt_mem + q_quant_mem)

    # --- peak memory usage in ref_paged_attn() ---
    ref_memory = 0
    for query_len, kv_len in seq_lens:
        kv_repeat_mem = 0
        if q_heads != k_heads:
            # kv copies: (kv_len, q_heads, head_size) dtype
            kv_repeat_mem = 2 * kv_len * q_heads * head_size * d_bytes
        # attn: (q_heads, query_len, kv_len) float32
        attn_mem = q_heads * query_len * kv_len * 4
        softmax_mem = q_heads * query_len * kv_len * 4
        # empty_mask: (query_len, kv_len) float32
        empty_mask_mem = query_len * kv_len * 4
        # mask: (query_len, kv_len) bool
        mask_mem = query_len * kv_len
        sliding_window_mask_mem = 0
        if sliding_window is not None:
            # sliding_window_mask: (query_len, kv_len) bool
            sliding_window_mask_mem = query_len * kv_len

        ref_memory = max(ref_memory,
                         kv_repeat_mem + attn_mem + softmax_mem + empty_mask_mem + mask_mem + sliding_window_mask_mem)

    ref_phase_memory = query_mem + kv_cache_mem + bt_mem + ref_memory
    # Double-counted: output and expected_output both live during assert_close.
    expected_output_mem = output_mem if BENCHMARKING_CONFIG["verify"] else 0
    triton_phase_memory = triton_memory + expected_output_mem
    total_memory = max(ref_phase_memory, triton_phase_memory)

    threshold = TOTAL_MEMORY_BYTES * safety_factor
    enough = total_memory < threshold

    return enough


# There is an error right now, for FP8 matmul with block size 16 is not supported
# Input shapes should have M >= 1, N >= 16 and K >= 32
MMAP_BLOCK_SIZES = [64] if IS_FP8 else [16, 64]
NUM_BLOCKS = [32768, 2048]
# Heavy seq_lens skip the block_size=16 cells to keep CI runtime bounded.
# block_size=64 still exercises both num_blocks values (2048 and 32768).
SEQ_LENS_LIGHT = [
    # One 4k input prefill
    [(4096, 4096)],
    # Pure decoding, 8 batches
    [(1, k) for k in [1513, 4100, 530, 123, 4803, 434, 3015, 34]],
    # Long-context pure decoding
    [(1, k) for k in [7168, 8192, 12288, 16384]],
]
SEQ_LENS_HEAVY = [
    # One 7k input prefill
    [(7168, 7168)],
    # Chunked prefill: 4 batches of 2k tokens
    [(2048, 2048)] * 4,
    # Chunked prefill: 2 batches of 8k tokens
    [(8192, 8192)] * 2,
    # High-concurrency continuous batching: 248 decode steps with kv_len mix
    # + 8 small (256-token) prefill chunks
    [(1, k)
     for k in ([1513, 4100, 530, 123, 4803, 434, 3015, 34, 256, 1024, 768, 2048, 192, 384, 1280, 96] * 16)[:248]] +
    [(256, 256)] * 8,
]
SEQ_LENS = SEQ_LENS_LIGHT + SEQ_LENS_HEAVY
# Models: (q_heads, k_heads, head_size, qdtype, sliding_window, soft_cap)
# sliding_window: None = full attention, int = sliding window size.
# soft_cap: None = disabled, float = soft_cap value.
# Models that use both attention types appear twice (one entry each).
MODELS_BF16 = [
    # llama3.1-8B / Qwen3-4B-thinking - full attention
    (32, 8, 128, None, None, None),
    # llama3.3-70B - full attention
    (64, 8, 128, None, None, None),
    # llama4 Scout - sliding window attention (window size 8192)
    (64, 8, 128, None, 8192, None),
    # Qwen2-7B - full attention
    (28, 4, 128, None, None, None),
    # Qwen2.5-235B - full attention
    (64, 4, 128, None, None, None),
    # Qwen2.5-235B - sliding window attention (window size 256)
    (64, 4, 128, None, 256, None),
]

MODELS_FP8 = [
    # llama4 scout - full attention
    (64, 8, 128, torch.float8_e4m3fn, None, None),
    # llama4 scout - sliding window attention (window size 8192)
    (64, 8, 128, torch.float8_e4m3fn, 8192, None),
]


def _build_attention_configs(model_configs):
    heavy = {id(s) for s in SEQ_LENS_HEAVY}
    configs = []
    for model_config in model_configs:
        *base_config, sliding_window, soft_cap = model_config
        for seq_lens, num_blocks, block_size in product(SEQ_LENS, NUM_BLOCKS, MMAP_BLOCK_SIZES):
            # Heavy seq_lens only run with block_size=64 to bound CI time.
            if id(seq_lens) in heavy and block_size != 64:
                continue
            x_val = (*base_config, seq_lens, sliding_window, soft_cap, num_blocks, block_size)
            qdtype = x_val[3]
            if qdtype is not None and qdtype.itemsize < 2 and x_val[-1] < 32:
                print("Skipping configuration due to incompatible q_dtype and block_size.")
                continue
            if is_enough_memory(x_val=x_val):
                configs.append(x_val)
            else:
                print(f"Skipping configuration due to memory constraints: {x_val}")
    return configs


ATTENTION_CONFIGS_BF16 = _build_attention_configs(MODELS_BF16)
ATTENTION_CONFIGS_FP8 = _build_attention_configs(MODELS_FP8)

# To debug if the benchmark runs at all, without waiting for all configurations to run
if os.getenv('DEBUG_BENCH', '0') == '1':
    ATTENTION_CONFIGS_BF16 = ATTENTION_CONFIGS_BF16[:1]
    ATTENTION_CONFIGS_FP8 = ATTENTION_CONFIGS_FP8[:1]


def get_unified_attention_benchmark(
    providers_filter: Optional[list[str]] = None,
    is_fp8=False,
    is_td_patched=False,
):
    supported_providers = {
        'triton' + ('-td' if is_td_patched else ''): 'triton' + ('-td' if is_td_patched else ''),
        'pytorch': 'pytorch',
    }
    if os.getenv("TRITON_INTERPRET", "0") == "1" and is_td_patched:
        # Skip triton providers if interpreter is used because if fails
        del supported_providers['triton']

    if not is_fp8:
        supported_providers['sycl-tla'] = 'sycl-tla'

    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    configs = ATTENTION_CONFIGS_FP8 if is_fp8 else ATTENTION_CONFIGS_BF16

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=[
                'q_heads', 'k_heads', 'head_size', 'qdtype', 'seq_lens', 'sliding_window', 'soft_cap', 'num_blocks',
                'block_size'
            ],
            x_vals=configs,
            line_arg='provider',
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[('green', '-'), ('blue', '--'), ('orange', ':')],
            ylabel=['GB/s', 'TFlops'],
            plot_name='unified-attention-performance' + ('-td' if is_td_patched else ''),
            args={},
        ))
    def benchmark(q_heads, k_heads, head_size, qdtype, seq_lens, sliding_window, soft_cap, num_blocks, block_size,
                  provider):
        print("Config:", q_heads, k_heads, head_size, qdtype, seq_lens, sliding_window, soft_cap, num_blocks,
              block_size, provider)
        dtype = torch.bfloat16
        torch.manual_seed(20)
        n_warmup = 100
        quantiles = [0.5, 0.0, 1.0]

        torch.set_default_device("xpu")

        num_seqs = len(seq_lens)
        query_lens = [x[0] for x in seq_lens]
        kv_lens = [x[1] for x in seq_lens]
        assert q_heads % k_heads == 0
        max_query_len = max(query_lens)
        max_kv_len = max(kv_lens)
        window_size = ((sliding_window - 1, 0) if sliding_window is not None else (-1, -1))
        scale = head_size**-0.5

        query = torch.randn(sum(query_lens), q_heads, head_size, dtype=dtype)
        key_cache = torch.randn(num_blocks, block_size, k_heads, head_size, dtype=dtype)
        value_cache = torch.randn_like(key_cache)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32)

        def torch_fn():
            return ref_paged_attn(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                query_lens=query_lens,
                kv_lens=kv_lens,
                block_tables=block_tables,
                scale=scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
            )

        if provider == 'pytorch':
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                torch_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )

        elif provider.startswith('triton'):
            expected_output = torch_fn() if BENCHMARKING_CONFIG["verify"] else None

            cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
            kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
            output = torch.empty_like(query)

            maybe_quantized_query = query
            maybe_quantized_key_cache = key_cache
            maybe_quantized_value_cache = value_cache
            q_descale = None
            k_descale = None
            v_descale = None
            if qdtype is not None:
                # QKV are drawn from N(0, 1): no need for a fp8 scaling factor
                maybe_quantized_query = query.to(qdtype)
                maybe_quantized_key_cache = key_cache.to(qdtype)
                maybe_quantized_value_cache = value_cache.to(qdtype)

                scale_shape = (num_seqs, k_heads)
                q_descale = None  # Not yet supported
                k_descale = torch.rand(scale_shape, dtype=torch.float32)
                v_descale = torch.rand(scale_shape, dtype=torch.float32)

            def triton_fn():
                unified_attention(
                    q=maybe_quantized_query,
                    k=maybe_quantized_key_cache,
                    v=maybe_quantized_value_cache,
                    out=output,
                    cu_seqlens_q=cu_query_lens,
                    seqused_k=kv_lens_tensor,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=scale,
                    causal=True,
                    window_size=window_size,
                    block_table=block_tables,
                    softcap=soft_cap if soft_cap is not None else 0,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    use_td=is_td_patched,
                )
                return output

            atol, rtol = 2.5e-2, 1e-2
            if qdtype is not None:
                atol, rtol = 3 / 8 + 1e-6, 1.5e-1
            benchmark_suite.assert_close(triton_fn, lambda: expected_output, atol=atol, rtol=rtol,
                                         err_msg='triton to torch')
            del expected_output

            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                triton_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )
            _record_autotune_decision(
                benchmark_name='unified-attention-performance' + ('-td' if is_td_patched else ''),
                provider=provider,
                q_heads=q_heads,
                k_heads=k_heads,
                head_size=head_size,
                qdtype=qdtype,
                seq_lens=seq_lens,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                num_blocks=num_blocks,
                block_size=block_size,
                mean_ms=mean_ms,
                cv=cv,
            )
        elif provider == 'sycl-tla':
            expected_output = torch_fn() if BENCHMARKING_CONFIG["verify"] else None

            cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
            kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

            def sycl_tla_fn():
                return sycl_tla_attention(
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    max_seqlen_q=max_query_len,
                    cu_seqlens_q=cu_query_lens,
                    max_seqlen_k=max_kv_len,
                    seqused_k=kv_lens_tensor,
                    softmax_scale=scale,
                    causal=True,
                    window_size=list(window_size),
                    block_table=block_tables,
                    softcap=soft_cap if soft_cap is not None else 0,
                )

            benchmark_suite.assert_close(sycl_tla_fn, lambda: expected_output, atol=2.5e-2, rtol=1e-2,
                                         err_msg='sycl-tla to torch')
            del expected_output

            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                sycl_tla_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )
        else:
            raise NotImplementedError(f'Unsupported provider {provider}')

        # Calculate performance metrics
        def gbps(ms):
            # Input Q/K/V use quantized dtype if available, output is always bf16
            in_bytes = qdtype.itemsize if qdtype is not None else dtype.itemsize
            out_bytes = dtype.itemsize  # output is always bf16
            total_query_tokens = sum(query_lens)
            # Memory: Query (in) + Key cache (in) + Value cache (in) + Output (out)
            total_bytes = (
                total_query_tokens * q_heads * head_size * in_bytes +  # Query
                sum([min(l, sliding_window) if sliding_window is not None else l
                     for l in kv_lens]) * k_heads * head_size * in_bytes * 2 +  # KV cache accessed
                total_query_tokens * q_heads * head_size * out_bytes  # Output
            )
            return total_bytes * (1e-9) / (ms * 1e-3)

        def tflops(ms):
            # Attention FLOPs: Q@K (2*d*seq_len*kv_len) + Softmax (~seq_len*kv_len) + Attn@V (2*d*seq_len*kv_len)
            total_flops = 0
            for i, (q_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
                if sliding_window is not None:
                    kv_len = min(kv_len, sliding_window)
                # Q@K^T and Attn@V operations
                flops_per_head = 2 * head_size * q_len * kv_len * 2  # 2 matmuls
                total_flops += flops_per_head * q_heads
            return total_flops * (1e-12) / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == '__main__':
    is_td_patched = os.getenv('TD_PATCHED', '0') == '1'
    _benchmark_attention = get_unified_attention_benchmark(is_fp8=IS_FP8, is_td_patched=is_td_patched)
    _benchmark_attention.run(show_plots=False, print_data=True)
