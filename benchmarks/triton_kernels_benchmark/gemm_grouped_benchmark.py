"""
Grouped GEMM benchmark for the Torch Inductor grouped_mm Triton template (issue #7086).

Runs torch._grouped_mm through torch.compile (the "triton" provider) and compares
it against eager torch._grouped_mm (the "onednn" provider, also the reference).
Two layouts are supported via the `variant` option: "2d3d" (MoE-style, with offsets)
and "3d3d" (batched, no offsets).

The grouped_mm Triton template is only enabled on XPU by pytorch/pytorch#185457.
Until the torch pin includes it, GROUPED_MM_PATCH_XPU (on by default) enables the
template on XPU with TMA off; it does nothing once torch supports it natively.

Env knobs: GROUPED_MM_PATCH_XPU, GROUPED_MM_TRITON_ONLY (TRITON-only autotune),
GROUPED_MM_VARIANT (for __main__), TRITON_RELAX_PROFILING_CHECK=1.

The "triton" provider times whatever torch.compile + max_autotune selects; under the
default autotune ATen can win, so the selected backend is reported per shape (set
GROUPED_MM_TRITON_ONLY=1 to force the Triton template).
"""
from functools import lru_cache
from typing import Optional
import os

import torch
import torch._inductor  # pylint: disable=protected-access
import torch._inductor.config as inductor_config  # pylint: disable=protected-access
from torch._inductor import select_algorithm  # pylint: disable=protected-access
from torch._inductor.kernel import mm_grouped  # pylint: disable=protected-access
# Raised when the Triton template fails to compile/autotune; caught to report NaN.
from torch._inductor.exc import InductorError  # pylint: disable=protected-access
from torch._dynamo.exc import BackendCompilerFailed  # pylint: disable=protected-access
from torch._inductor.select_algorithm import NoValidChoicesError  # pylint: disable=protected-access

import triton_kernels_benchmark as benchmark_suite

_COMPILE_ERRORS = (InductorError, BackendCompilerFailed, NoValidChoicesError)

# Backend autotune selected for the last grouped_mm compile (set by the feedback saver).
_LAST_SELECTED_BACKEND = {}
_SETUP_STATE = {}


def _grouped_mm_feedback_saver(timings, name, *_args, **_kwargs):
    """Inductor feedback-saver hook: record the fastest choice's backend.

    Extra positional args vary across torch versions; only timings/name are used.
    """
    if name != "grouped_mm" or not timings:
        return
    best = min(timings, key=timings.get)
    backend = "triton" if isinstance(best, select_algorithm.TritonTemplateCaller) else "aten"
    _LAST_SELECTED_BACKEND["grouped_mm"] = (backend, type(best).__name__)


def _enable_xpu_grouped_mm_template() -> None:
    """Allow Inductor to pick the grouped_mm Triton template on XPU, with TMA off.

    Touches private, in-flux torch._inductor.kernel.mm_grouped API, so it is guarded:
    a signature change (e.g. a torch-pin bump landing pytorch/pytorch#185457) degrades
    to "template stays unavailable" rather than crashing the benchmark suite.
    """
    mg = mm_grouped

    if getattr(mg, "_xpu_grouped_mm_patched", False):
        return

    try:
        _orig_can_use = mg.can_use_triton_kernel

        def _can_use_with_xpu(mat_a, mat_b, offs, bias, scale_result):
            if _orig_can_use(mat_a, mat_b, offs, bias, scale_result):
                return True
            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                return False
            if not mg.has_triton():
                return False
            # bias / scale_result are not supported.
            if bias is not None or scale_result is not None:
                return False
            # 2D operand needs offsets; 3Dx3D needs none.
            if len(mat_a.get_size()) == 2 or len(mat_b.get_size()) == 2:
                return offs is not None
            return offs is None

        mg.can_use_triton_kernel = _can_use_with_xpu

        # Force USE_TMA_LOAD=False on every template choice (pointer-based loads).
        _template = mg.triton_grouped_mm_template
        _orig_append = _template.maybe_append_choice

        def _append_pointer_loads(choices, **kwargs):
            if "USE_TMA_LOAD" in kwargs and (hasattr(torch, "xpu") and torch.xpu.is_available()):
                kwargs["USE_TMA_LOAD"] = False
            return _orig_append(choices, **kwargs)

        _template.maybe_append_choice = _append_pointer_loads
        mg._xpu_grouped_mm_patched = True  # pylint: disable=protected-access
    except AttributeError as e:
        print(f"gemm_grouped_benchmark: skipping XPU grouped_mm patch ({e}); "
              "the Triton template will be unavailable on XPU.")


def _setup_inductor() -> None:
    """Apply the Inductor/Dynamo config this benchmark needs.

    Called from get_benchmark() (not at import) so the global state is only mutated
    when this benchmark is actually selected, not for every triton-benchmarks command.
    """
    if os.getenv("GROUPED_MM_PATCH_XPU", "1") == "1":
        _enable_xpu_grouped_mm_template()
    # Let Inductor autotune the Triton GEMM templates.
    inductor_config.max_autotune = True
    inductor_config.max_autotune_gemm = True
    if os.getenv("GROUPED_MM_TRITON_ONLY", "0") == "1":
        # Time only the Triton template (no ATen fallback).
        inductor_config.max_autotune_gemm_backends = "TRITON"
        inductor_config.autotune_fallback_to_aten = False
    # Report which backend autotune picks (register the saver once, process-global).
    if not _SETUP_STATE.get("feedback_saver_registered"):
        try:
            select_algorithm.add_feedback_saver(_grouped_mm_feedback_saver)
            _SETUP_STATE["feedback_saver_registered"] = True
        except AttributeError as e:
            print(f"gemm_grouped_benchmark: could not register autotune feedback saver ({e}); "
                  "selected backend will not be reported.")
    # Each (G, M, N, K) recompiles under dynamic=False.
    torch._dynamo.config.recompile_limit = 100  # pylint: disable=protected-access


@lru_cache
def _compiled_grouped_mm(with_offs: bool):
    if with_offs:

        def f_offs(a, b, offs):
            return torch._grouped_mm(a, b, offs=offs)  # pylint: disable=protected-access

        return torch.compile(f_offs, dynamic=False)

    def f_no_offs(a, b):
        return torch._grouped_mm(a, b)  # pylint: disable=protected-access

    return torch.compile(f_no_offs, dynamic=False)


def make_inputs(G, M, N, K, variant):
    """Build (a, b, offs) for the requested grouped-mm layout (bf16, equal groups)."""
    if variant == "2d3d":
        a = torch.rand((G * M, K), device="xpu", dtype=torch.bfloat16)
        b = torch.rand((G, K, N), device="xpu", dtype=torch.bfloat16)
        offs = torch.arange(1, G + 1, device="xpu", dtype=torch.int32) * M
        return a, b, offs
    if variant == "3d3d":
        a = torch.rand((G, M, K), device="xpu", dtype=torch.bfloat16)
        b = torch.rand((G, K, N), device="xpu", dtype=torch.bfloat16)
        return a, b, None
    raise ValueError(f"Unsupported variant {variant!r}; expected '2d3d' or '3d3d'")


# [G, M, N, K] with equal-sized groups.
X_VALS = [  #
    [8, 512, 4096, 4096],
    [8, 1024, 4096, 4096],
    [16, 512, 2048, 2048],
    [16, 1024, 4096, 4096],
    [32, 256, 4096, 4096],
    [64, 128, 2048, 2048],
    [128, 64, 1024, 1024],
]

DEVICE_NAME = torch.xpu.get_device_name()
DEVICE_TOTAL_MEMORY = torch.xpu.get_device_properties().total_memory


def is_enough_memory(x_val):
    # a, b, out are bf16 (2 bytes), plus an eager reference copy.
    G, M, N, K = x_val
    required_memory = G * M * K * 2 + G * K * N * 2 + 2 * (G * M * N * 2)
    enough_memory = required_memory < DEVICE_TOTAL_MEMORY
    if not enough_memory:
        print(f"'{x_val}' combination skipped for '{DEVICE_NAME}'; {required_memory=} but {DEVICE_TOTAL_MEMORY=}")
    return enough_memory


X_VALS = [x_val for x_val in X_VALS if is_enough_memory(x_val)]


def get_benchmark(
    providers_filter: Optional[list[str]] = None,
    variant: str = "2d3d",
    plot_name: str = "grouped-gemm-performance",
):
    """
    Returns a Mark object containing a Benchmark object constructed at runtime and parameterized by the provided option values.
    The benchmark can then be executed by calling the :code:`.run` method on the return value.
    """
    supported_providers = {
        "triton": "Triton",  # Inductor-compiled grouped_mm
        "onednn": "OneDNN",  # eager torch._grouped_mm (reference)
    }
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["G", "M", "N", "K"],
            x_vals=X_VALS,
            line_arg="provider",
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[("green", "-"), ("blue", "-")],
            ylabel=["GB/s", "TFlops"],
            plot_name=plot_name,
            args={},
        ))
    def benchmark(G, M, N, K, provider):
        do_bench = benchmark_suite.get_do_bench(n_warmup=200, n_repeat=10, quantiles=[0.5, 0.0, 1.0])

        torch.manual_seed(0)
        a, b, offs = make_inputs(G, M, N, K, variant)
        with_offs = offs is not None

        eager_fn = lambda: (
            torch._grouped_mm(a, b, offs=offs)  # pylint: disable=protected-access
            if with_offs else torch._grouped_mm(a, b))  # pylint: disable=protected-access

        if provider == "onednn":
            _, min_ms, max_ms, mean_ms, cv = do_bench(eager_fn)

        elif provider == "triton":
            _setup_inductor()
            try:
                compiled = _compiled_grouped_mm(with_offs)
                triton_fn = (lambda: compiled(a, b, offs)) if with_offs else (lambda: compiled(a, b))
                benchmark_suite.assert_close(triton_fn, eager_fn, atol=1e-4, rtol=1e-2,
                                             err_msg="inductor grouped_mm to eager")
                _, min_ms, max_ms, mean_ms, cv = do_bench(triton_fn)
            except _COMPILE_ERRORS as e:
                # Template failed to compile (no ATen fallback under TRITON_ONLY): report NaN.
                _compiled_grouped_mm.cache_clear()
                print(f"gemm_grouped_benchmark[{variant}] G={G} M={M} N={N} K={K}: "
                      f"Triton template unavailable ({type(e).__name__}); reporting NaN.")
                return (float("nan"), ) * 3, (float("nan"), ) * 3, float("nan")
            # Report the measured backend (unknown on an autotune cache hit).
            backend, caller = _LAST_SELECTED_BACKEND.get("grouped_mm", ("unknown", "n/a"))
            if backend == "aten":
                print(f"gemm_grouped_benchmark[{variant}] G={G} M={M} N={N} K={K}: "
                      f"'triton' provider measured backend=aten ({caller}) - not the "
                      "Triton template; set GROUPED_MM_TRITON_ONLY=1 to force it.")
            elif backend == "unknown":
                print(f"gemm_grouped_benchmark[{variant}] G={G} M={M} N={N} K={K}: "
                      "selected backend unknown (autotune cache hit); run with a clean "
                      "Inductor cache to report triton-vs-aten.")

        else:
            raise NotImplementedError(f"Unsupported provider {provider}")

        # Equal groups: total work = G * 2*M*N*K; bytes = a + b + out (bf16).
        tflops = lambda ms: 2 * G * M * N * K * (1e-12) / (ms * 1e-3)
        gbps = lambda ms: (G * M * K + G * K * N + G * M * N) * 2 * (1e-9) / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == "__main__":
    _benchmark = get_benchmark(variant=os.getenv("GROUPED_MM_VARIANT", "2d3d"))
    _benchmark.run(show_plots=False, print_data=True)
