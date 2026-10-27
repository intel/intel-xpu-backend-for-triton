"""Standalone benchmark for the extracted layer-norm M2 reduction (kernel 89).

Run from the repository root:
  PYTHONPATH=benchmarks python3 -m triton_kernels_benchmark.triton_red_fused_native_layer_norm_89_benchmark

Input: FP16 [xnumel, 512]. Output: FP32 [xnumel] containing the sum of
squared deviations, not variance. Kernel 90 divides this output by 512.
Set TRITON_RED_FUSED_NATIVE_LAYER_NORM_89_PATH to override the extracted dump.
See the kernel-88 benchmark for shared timing and path options.
"""

from __future__ import annotations

from typing import Any

from triton_kernels_benchmark.triton_red_fused_native_layer_norm_88_benchmark import _get_benchmark

KERNEL_NAME = "triton_red_fused_native_layer_norm_89"


def get_benchmark(
    providers_filter: list[str] | None = None,
    kernel_path: str | None = None,
    x_vals: list[int] | None = None,
) -> Any:
    """Build the M2-reduction report for FP16 rows of length 512.

    Arguments:
      providers_filter: Providers to benchmark ("triton" and/or "torch").
      kernel_path: Optional extracted kernel dump override.
      x_vals: Positive row counts; defaults include partial-block cases.

    Returns:
      A benchmark-suite report with bandwidth and latency metrics.
    """
    return _get_benchmark(KERNEL_NAME, providers_filter, kernel_path, x_vals)


if __name__ == "__main__":
    get_benchmark().run(show_plots=False, print_data=True)
