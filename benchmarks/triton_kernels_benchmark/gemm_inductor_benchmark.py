"""Compare TorchInductor ATen/Triton GEMM backends and manual grouped tile-order/Split-K/Stream-K GEMM.

The cases are the FP16 GEMMs for which ATen beat Triton in disassembles/log.txt.
Pass --case repeatedly to benchmark selected one-based cases, or --list to inspect
their shapes and strides without requiring an XPU. Results use the
triton_kernels_benchmark report format.

Manual providers reuse the kernels in gemm_benchmark, gemm_splitk_benchmark,
and gemm_streamk_benchmark (not Inductor's grouped_mm). All manual providers use
the original inputs and strides without padding, packing, or copying. The Intel
compiler lowers descriptors to block I/O or masked pointers. Split-K and Stream-K run
one GEMM per batch; their timings include zeroing and all launches. All manual
providers accumulate in FP32, add any addmm bias, and return FP16 outputs.
"""

import argparse
import dataclasses
from dataclasses import dataclass

import torch
import torch._inductor.config as inductor_config
import triton

# triton_kernels_benchmark is imported where it is used, because importing it fails without a GPU and --list must work
# without one.
# pylint: disable=import-outside-toplevel


@dataclass(frozen=True)
class GemmCase:
    op: str
    shapes: tuple[tuple[int, ...], ...]
    strides: tuple[tuple[int, ...], ...]
    logged_aten_ms: float
    logged_triton_ms: float


# Shapes and strides are captured that inductor Triton GEMM are not in good perf.
GEMM_CASES = (
    GemmCase("bmm", ((80, 256, 640), (80, 640, 960)), ((640, 51200, 1), (614400, 960, 1)), 0.261250, 0.263958),
    GemmCase("bmm", ((32, 256, 200), (32, 200, 1600)), ((51200, 200, 1), (320000, 1600, 1)), 0.088437, 0.091875),
    GemmCase("addmm", ((256, 2560), (256, 7076), (7076, 2560)), ((0, 1), (7088, 1), (2560, 1)), 0.131146, 0.194792),
    GemmCase("mm", ((256, 744), (744, 744)), ((744, 1), (744, 1)), 0.012083, 0.015312),
    GemmCase("addmm", ((256, 512), (256, 356), (356, 512)), ((0, 1), (356, 1), (512, 1)), 0.010417, 0.010625),
    GemmCase("addmm", ((256, 512), (256, 490), (490, 512)), ((0, 1), (490, 1), (512, 1)), 0.006354, 0.012812),
    GemmCase("addmm", ((256, 256), (256, 640), (640, 256)), ((0, 1), (640, 1), (256, 1)), 0.005834, 0.006042),
    GemmCase("addmm", ((256, 256), (256, 252), (252, 256)), ((0, 1), (252, 1), (256, 1)), 0.007604, 0.008125),
    GemmCase("addmm", ((256, 256), (256, 7347), (7347, 256)), ((0, 1), (7360, 1), (256, 1)), 0.038646, 0.152917),
    GemmCase("addmm", ((256, 256), (256, 7343), (7343, 256)), ((0, 1), (7360, 1), (256, 1)), 0.032708, 0.153437),
    GemmCase("addmm", ((256, 256), (256, 512), (512, 256)), ((0, 1), (2560, 1), (256, 1)), 0.005000, 0.005729),
    GemmCase("addmm", ((256, 256), (256, 856), (856, 256)), ((0, 1), (856, 1), (256, 1)), 0.006562, 0.015729),
    GemmCase("addmm", ((256, 256), (256, 1616), (1616, 256)), ((0, 1), (1664, 1), (256, 1)), 0.010625, 0.014583),
    GemmCase("addmm", ((256, 128), (256, 560), (560, 128)), ((0, 1), (560, 1), (128, 1)), 0.006250, 0.008855),
    GemmCase("addmm", ((256, 128), (256, 992), (992, 128)), ((0, 1), (992, 1), (128, 1)), 0.006979, 0.007396),
    GemmCase("addmm", ((8448, 1536), (8448, 512), (512, 1536)), ((0, 1), (512, 1), (1, 512)), 0.084271, 0.166875),
    GemmCase("addmm", ((8448, 512), (8448, 512), (512, 512)), ((0, 1), (512, 1), (512, 1)), 0.039062, 0.070417),
    GemmCase("addmm", ((8448, 512), (8448, 1024), (1024, 512)), ((0, 1), (1024, 1), (512, 1)), 0.061771, 0.121042),
    GemmCase("addmm", ((8448, 1024), (8448, 512), (512, 1024)), ((0, 1), (512, 1), (1, 512)), 0.062084, 0.128229),
    GemmCase("addmm", ((256, 256), (256, 1072), (1072, 256)), ((0, 1), (1088, 1), (256, 1)), 0.009375, 0.011145),
    GemmCase("addmm", ((256, 128), (256, 744), (744, 128)), ((0, 1), (744, 1), (128, 1)), 0.006667, 0.016667),
    GemmCase("addmm", ((256, 640), (256, 1288), (1288, 640)), ((0, 1), (1344, 1), (640, 1)), 0.017708, 0.026771),
    GemmCase("addmm", ((256, 640), (256, 1080), (1080, 640)), ((0, 1), (1088, 1), (640, 1)), 0.017187, 0.022708),
    GemmCase("addmm", ((256, 640), (256, 1778), (1778, 640)), ((0, 1), (1792, 1), (640, 1)), 0.024792, 0.032708),
    GemmCase("addmm", ((256, 640), (256, 576), (576, 640)), ((0, 1), (576, 1), (640, 1)), 0.007396, 0.007813),
    GemmCase("addmm", ((256, 640), (256, 976), (976, 640)), ((0, 1), (976, 1), (640, 1)), 0.012396, 0.014479),
    GemmCase("addmm", ((256, 640), (256, 1004), (1004, 640)), ((0, 1), (1004, 1), (640, 1)), 0.013854, 0.022292),
    GemmCase("addmm", ((256, 640), (256, 688), (688, 640)), ((0, 1), (688, 1), (640, 1)), 0.010521, 0.010833),
    GemmCase("addmm", ((256, 256), (256, 2540), (2540, 256)), ((0, 1), (2540, 1), (256, 1)), 0.011458, 0.047708),
    GemmCase("addmm", ((256, 256), (256, 1288), (1288, 256)), ((0, 1), (1288, 1), (256, 1)), 0.011250, 0.025938),
    GemmCase("addmm", ((256, 512), (256, 6052), (6052, 512)), ((0, 1), (6080, 1), (512, 1)), 0.046666, 0.122917),
    GemmCase("addmm", ((8704, 512), (8704, 512), (512, 512)), ((0, 1), (512, 1), (512, 1)), 0.039271, 0.072292),
    GemmCase("addmm", ((8704, 512), (8704, 1024), (1024, 512)), ((0, 1), (1024, 1), (512, 1)), 0.061458, 0.124271),
    GemmCase("addmm", ((256, 257), (256, 4544), (4544, 257)), ((0, 1), (4544, 1), (257, 1)), 0.039896, 0.071354),
)

# Benchmark mode -> report label.
MODES = {
    "aten_only": "Inductor ATEN",
    "triton_aten": "Inductor TRITON+ATEN",
    "triton_only": "Inductor TRITON",
    "manual_grouped": "Manual Triton grouped tiles",
    "manual_splitk": "Manual Triton Split-K",
    "manual_streamk": "Manual Triton Stream-K",
}
# Inductor mode -> max_autotune_gemm_backends.
INDUCTOR_BACKENDS = {"aten_only": "ATEN", "triton_aten": "TRITON,ATEN", "triton_only": "TRITON"}


def storage_size(shape: tuple[int, ...], stride: tuple[int, ...]) -> int:
    """Number of elements a strided tensor spans."""
    return 1 + sum((dim - 1) * step for dim, step in zip(shape, stride))


def make_inputs(case: GemmCase) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.randn(storage_size(shape, stride), device="xpu", dtype=torch.float16).as_strided(shape, stride)
        for shape, stride in zip(case.shapes, case.strides))


def _dedup_gemm_configs(configs):
    seen = set()
    out = []
    for config in configs:
        key = tuple(dataclasses.asdict(config).items())
        if key not in seen:
            seen.add(key)
            out.append(config)
    return out


def _register_custom_xpu_mm_heuristic():
    """Register a custom XPU MM template heuristic to keep the reproducer faithful.

    This matches the logic in `gemm_bmm_80x256x640.py` so the compiled path still
    exercises the same autotune search space and settings as the source reproducer.
    """
    from torch._inductor.heuristics.template.registry import _HEURISTIC_CACHE, register_template_heuristic
    from torch._inductor.heuristics.template.triton import GemmConfig, MMTemplateConfigMixin, XPUConfigHeuristic
    from torch._inductor.heuristics.template.triton_addmm import AddMMConfigMixin
    from torch._inductor.kernel.bmm import bmm_template
    from torch._inductor.kernel.mm import mm_template

    custom_mm_configs = [
        GemmConfig(128, 128, 64, 2, 32),
        GemmConfig(128, 128, 32, 2, 16),
        GemmConfig(16, 512, 64, 2, 8),
        GemmConfig(32, 256, 32, 2, 16),
        GemmConfig(32, 256, 128, 3, 32),
        GemmConfig(128, 128, 64, 2, 16),
        GemmConfig(128, 128, 64, 3, 16),
        GemmConfig(128, 128, 64, 3, 32),
        GemmConfig(128, 128, 128, 2, 16),
        GemmConfig(128, 256, 16, 4, 32),
        GemmConfig(128, 256, 32, 3, 32),
        GemmConfig(128, 256, 64, 2, 32),
        GemmConfig(128, 512, 32, 2, 32),
        GemmConfig(128, 512, 32, 3, 32),
        GemmConfig(128, 512, 32, 4, 32),
        GemmConfig(256, 128, 64, 2, 32),
        GemmConfig(256, 256, 32, 5, 32),
        GemmConfig(512, 128, 32, 1, 32),
    ]

    @register_template_heuristic(mm_template.uid, "xpu")
    @register_template_heuristic(bmm_template.uid, "xpu")
    class MyXPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, XPUConfigHeuristic):

        def __init__(self) -> None:
            super().__init__()
            self.mm_configs = _dedup_gemm_configs([*self.mm_configs, *custom_mm_configs])
            self.exhaustive_configs = self.mm_configs

    @register_template_heuristic(mm_template.uid, "xpu", op_name="addmm")
    @register_template_heuristic(bmm_template.uid, "xpu", op_name="baddbmm")
    class MyXPUAddmmTemplateConfigHeuristic(AddMMConfigMixin, MyXPUMMTemplateConfigHeuristic):  # pylint: disable=unused-variable
        pass

    _HEURISTIC_CACHE.clear()


def _manual_gemm(mode: str, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    from triton_kernels_benchmark import gemm_splitk_benchmark, gemm_streamk_benchmark
    from triton_kernels_benchmark.gemm_benchmark import (matmul_kernel_with_tensor_descriptors,
                                                         matmul_kernel_with_tensor_descriptors_batched)

    bias = inputs[0] if len(inputs) == 3 else None
    a, b = inputs[-2:]
    m, k = a.shape[-2:]
    n = b.shape[-1]
    # Split-K accumulates with atomics, so its output must start at zero.
    alloc = torch.zeros if mode == "manual_splitk" else torch.empty
    output = alloc((*a.shape[:-2], m, n), device=a.device, dtype=torch.float32)

    if mode == "manual_grouped":
        assert all(x.data_ptr() % 16 == 0 for x in (a, b, output)), "Matrix base addresses must be 16-byte aligned"
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]), a.shape[0]
                             if a.ndim == 3 else 1)
        if a.ndim == 3:
            matmul_kernel_with_tensor_descriptors_batched[grid](a, b, output, a.shape[0], m, n, k, *a.stride(),
                                                                *b.stride(), *output.stride())
        else:
            matmul_kernel_with_tensor_descriptors[grid](a, b, output, m, n, k, *a.stride(), *b.stride(),
                                                        *output.stride())
    else:
        matmul = {"manual_splitk": gemm_splitk_benchmark.matmul, "manual_streamk": gemm_streamk_benchmark.matmul}[mode]
        # Both kernels are 2D only, so a bmm runs one GEMM per batch.
        for a_i, b_i, c_i in zip(a, b, output) if a.ndim == 3 else [(a, b, output)]:
            matmul(a_i, b_i, c_i)

    if bias is not None:
        output = output + bias
    return output.to(a.dtype)


def get_benchmark(case_ids: list[int], modes: list[str]):
    import triton_kernels_benchmark as benchmark_suite

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["case"],
            x_vals=case_ids,
            line_arg="mode",
            line_vals=modes,
            line_names=[MODES[mode] for mode in modes],
            styles=[("blue", "-"), ("green", "-"), ("green", "--"), ("orange", "-"), ("red", "-"),
                    ("purple", "-")][:len(modes)],
            ylabel=["GB/s", "TFlops"],
            plot_name="gemm-aten-winning-cases",
            args={},
        ))
    def benchmark(case: int, mode: str):
        do_bench = benchmark_suite.get_do_bench(n_warmup=200, n_repeat=10, quantiles=[0.5, 0.0, 1.0])
        gemm_case = GEMM_CASES[case - 1]
        inputs = make_inputs(gemm_case)
        operation = getattr(torch, gemm_case.op)
        torch_fn = lambda: operation(*inputs)

        if mode.startswith("manual_"):
            manual_fn = lambda: _manual_gemm(mode, inputs)
            benchmark_suite.assert_close(manual_fn, torch_fn, atol=5e-2, rtol=1e-2, err_msg=f"{mode} to torch")
            _, min_ms, max_ms, mean_ms, cv = do_bench(manual_fn)
        else:
            torch.compiler.reset()
            with inductor_config.patch(max_autotune_gemm=True, max_autotune_gemm_backends=INDUCTOR_BACKENDS[mode],
                                       force_disable_caches=True):
                compiled = torch.compile(operation, mode="max-autotune-no-cudagraphs")
                compiled_fn = lambda: compiled(*inputs)
                compiled_fn()  # Compile and autotune outside the timed region.
                benchmark_suite.assert_close(compiled_fn, torch_fn, atol=5e-2, rtol=1e-2, err_msg=f"{mode} to torch")
                _, min_ms, max_ms, mean_ms, cv = do_bench(compiled_fn)
            torch.compiler.reset()

        a, b = inputs[-2:]
        batch = a.shape[0] if a.ndim == 3 else 1
        m, k = a.shape[-2:]
        n = b.shape[-1]
        # FP16 elements the inputs span (an addmm bias with stride 0 reads only one row), plus the FP16 output.
        num_bytes = 2 * (sum(storage_size(x.shape, x.stride()) for x in inputs) + batch * m * n)
        gbps = lambda ms: num_bytes * 1e-9 / (ms * 1e-3)
        tflops = lambda ms: 2 * batch * m * n * k * 1e-12 / (ms * 1e-3)
        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


def print_cases(case_ids: list[int]):
    for case_id in case_ids:
        case = GEMM_CASES[case_id - 1]
        print(f"{case_id}: {case.op}, shapes={case.shapes}, strides={case.strides}")
        print(f"   log winner: ATen={case.logged_aten_ms:.6f} ms, best Triton={case.logged_triton_ms:.6f} ms",
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    parser.add_argument("--case", type=int, action="append", dest="case_ids",
                        help="one-based case number; repeat to select multiple")
    parser.add_argument("--reports", default="", help="directory for benchmark-suite CSV/plot reports")
    parser.add_argument("--n-runs", type=int, default=1, help="number of benchmark-suite report repetitions")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    args = parser.parse_args()

    case_ids = args.case_ids or list(range(1, len(GEMM_CASES) + 1))
    if any(case_id <= 0 or case_id > len(GEMM_CASES) for case_id in case_ids):
        parser.error(f"--case must be between 1 and {len(GEMM_CASES)}")
    print_cases(case_ids)
    if args.list:
        return

    if not torch.xpu.is_available():
        raise RuntimeError("This benchmark requires an XPU-enabled PyTorch installation and an available Intel GPU")
    torch.manual_seed(0)
    modes = list(MODES) if args.mode == "all" else [args.mode]
    if {"triton_aten", "triton_only"} & set(modes):
        _register_custom_xpu_mm_heuristic()
    print(f"Device: {torch.xpu.get_device_name()}", flush=True)
    from triton_kernels_benchmark.benchmark_testing import MarkArgs

    get_benchmark(case_ids, modes).run(show_plots=False, print_data=True,
                                       mark_args=MarkArgs(reports=args.reports, n_runs=args.n_runs))


if __name__ == "__main__":
    main()
