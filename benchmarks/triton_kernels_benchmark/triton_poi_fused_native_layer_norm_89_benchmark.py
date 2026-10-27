"""Compatibility wrapper for the legacy `triton_poi_fused_native_layer_norm_89` name."""

from triton_kernels_benchmark import triton_per_fused_native_layer_norm_89_benchmark as _base

KERNEL_NAME = _base.KERNEL_NAME
KERNEL_ALIASES = _base.KERNEL_ALIASES
KERNEL_PATH_ENV_VAR = _base.KERNEL_PATH_ENV_VAR
R0_NUMEL = _base.R0_NUMEL
DEFAULT_XNUMELS = _base.DEFAULT_XNUMELS

_get_default_kernel_path = _base._get_default_kernel_path
_resolve_kernel_path = _base._resolve_kernel_path
_load_kernel = _base._load_kernel
_get_xpu_raw_stream = _base._get_xpu_raw_stream
_make_inputs = _base._make_inputs
_make_outputs = _base._make_outputs
_run_reference = _base._run_reference
_run_triton = _base._run_triton
_perf_rate = _base._perf_rate
get_benchmark = _base.get_benchmark

__all__ = [
    "KERNEL_NAME",
    "KERNEL_ALIASES",
    "KERNEL_PATH_ENV_VAR",
    "R0_NUMEL",
    "DEFAULT_XNUMELS",
    "_get_default_kernel_path",
    "_resolve_kernel_path",
    "_load_kernel",
    "_get_xpu_raw_stream",
    "_make_inputs",
    "_make_outputs",
    "_run_reference",
    "_run_triton",
    "_perf_rate",
    "get_benchmark",
]
