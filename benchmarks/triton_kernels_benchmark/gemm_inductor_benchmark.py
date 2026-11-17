"""Compare TorchInductor ATen/Triton GEMM backends and manual grouped tile-order/Split-K/Stream-K GEMM.

The cases are FP16 GEMMs where Inductor's Triton GEMM was slower than ATen.
Pass --case repeatedly to benchmark selected one-based cases, or --list to inspect
their shapes and strides, and which matrices the manual kernels access through
masked pointers, without requiring an XPU. Results use the triton_kernels_benchmark
report format.

Manual providers reuse the kernels in gemm_benchmark, gemm_postop_addmatrix_benchmark,
gemm_splitk_benchmark and gemm_streamk_benchmark (not Inductor's grouped_mm), on the
original inputs and strides without padding, packing or copying. The kernels use a
tensor descriptor when the strides allow one, and masked pointers otherwise. Split-K and
Stream-K are 2D only, so for the bmm cases they run one GEMM per batch; their timings
include zeroing and all launches. Each of those GEMMs has only a few output tiles, far
too few to fill the GPU, so these bmm numbers understate both algorithms. Their CI
benchmarks have a single config tuned for large square shapes, so here each case first
picks the fastest tile (and SPLIT_K for Split-K), outside the timed region, and prints it.

All manual providers accumulate in FP32, add any addmm bias and return FP16 outputs.
The grouped-tile kernels add the bias and store FP16 in their epilogue, as Inductor's
template does; the addmm cases use the post-op add-matrix kernel, tuned over
gemm_benchmark's configs. Split-K and Stream-K reduce partial sums in an FP32 buffer, so
they add the bias and cast in one more pass, which their timings include.

The Inductor modes print whether Inductor picked its Triton template or an ATen kernel
for each case.
"""

from __future__ import annotations

import argparse
import functools
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch._inductor.config as inductor_config
from torch._inductor.utils import run_and_get_code
import triton
import triton.language as tl

# triton_kernels_benchmark is imported where it is used, because importing it fails without a GPU and --list must work
# without one.
# pylint: disable=import-outside-toplevel
if TYPE_CHECKING:
    from torch._inductor.heuristics.template.triton import GemmConfig
    from triton_kernels_benchmark.benchmark_testing import Mark


@dataclass(frozen=True)
class GemmCase:
    op: str
    shapes: tuple[tuple[int, ...], ...]
    strides: tuple[tuple[int, ...], ...]


# Shapes and strides where Inductor's Triton GEMM was slower than ATen.
GEMM_CASES = (
    GemmCase("bmm", ((80, 256, 640), (80, 640, 960)), ((640, 51200, 1), (614400, 960, 1))),
    GemmCase("bmm", ((32, 256, 200), (32, 200, 1600)), ((51200, 200, 1), (320000, 1600, 1))),
    GemmCase("addmm", ((256, 2560), (256, 7076), (7076, 2560)), ((0, 1), (7088, 1), (2560, 1))),
    GemmCase("mm", ((256, 744), (744, 744)), ((744, 1), (744, 1))),
    GemmCase("addmm", ((256, 512), (256, 356), (356, 512)), ((0, 1), (356, 1), (512, 1))),
    GemmCase("addmm", ((256, 512), (256, 490), (490, 512)), ((0, 1), (490, 1), (512, 1))),
    GemmCase("addmm", ((256, 256), (256, 640), (640, 256)), ((0, 1), (640, 1), (256, 1))),
    GemmCase("addmm", ((256, 256), (256, 252), (252, 256)), ((0, 1), (252, 1), (256, 1))),
    GemmCase("addmm", ((256, 256), (256, 7347), (7347, 256)), ((0, 1), (7360, 1), (256, 1))),
    GemmCase("addmm", ((256, 256), (256, 7343), (7343, 256)), ((0, 1), (7360, 1), (256, 1))),
    GemmCase("addmm", ((256, 256), (256, 512), (512, 256)), ((0, 1), (2560, 1), (256, 1))),
    GemmCase("addmm", ((256, 256), (256, 856), (856, 256)), ((0, 1), (856, 1), (256, 1))),
    GemmCase("addmm", ((256, 256), (256, 1616), (1616, 256)), ((0, 1), (1664, 1), (256, 1))),
    GemmCase("addmm", ((256, 128), (256, 560), (560, 128)), ((0, 1), (560, 1), (128, 1))),
    GemmCase("addmm", ((256, 128), (256, 992), (992, 128)), ((0, 1), (992, 1), (128, 1))),
    GemmCase("addmm", ((8448, 1536), (8448, 512), (512, 1536)), ((0, 1), (512, 1), (1, 512))),
    GemmCase("addmm", ((8448, 512), (8448, 512), (512, 512)), ((0, 1), (512, 1), (512, 1))),
    GemmCase("addmm", ((8448, 512), (8448, 1024), (1024, 512)), ((0, 1), (1024, 1), (512, 1))),
    GemmCase("addmm", ((8448, 1024), (8448, 512), (512, 1024)), ((0, 1), (512, 1), (1, 512))),
    GemmCase("addmm", ((256, 256), (256, 1072), (1072, 256)), ((0, 1), (1088, 1), (256, 1))),
    GemmCase("addmm", ((256, 128), (256, 744), (744, 128)), ((0, 1), (744, 1), (128, 1))),
    GemmCase("addmm", ((256, 640), (256, 1288), (1288, 640)), ((0, 1), (1344, 1), (640, 1))),
    GemmCase("addmm", ((256, 640), (256, 1080), (1080, 640)), ((0, 1), (1088, 1), (640, 1))),
    GemmCase("addmm", ((256, 640), (256, 1778), (1778, 640)), ((0, 1), (1792, 1), (640, 1))),
    GemmCase("addmm", ((256, 640), (256, 576), (576, 640)), ((0, 1), (576, 1), (640, 1))),
    GemmCase("addmm", ((256, 640), (256, 976), (976, 640)), ((0, 1), (976, 1), (640, 1))),
    GemmCase("addmm", ((256, 640), (256, 1004), (1004, 640)), ((0, 1), (1004, 1), (640, 1))),
    GemmCase("addmm", ((256, 640), (256, 688), (688, 640)), ((0, 1), (688, 1), (640, 1))),
    GemmCase("addmm", ((256, 256), (256, 2540), (2540, 256)), ((0, 1), (2540, 1), (256, 1))),
    GemmCase("addmm", ((256, 256), (256, 1288), (1288, 256)), ((0, 1), (1288, 1), (256, 1))),
    GemmCase("addmm", ((256, 512), (256, 6052), (6052, 512)), ((0, 1), (6080, 1), (512, 1))),
    GemmCase("addmm", ((8704, 512), (8704, 512), (512, 512)), ((0, 1), (512, 1), (512, 1))),
    GemmCase("addmm", ((8704, 512), (8704, 1024), (1024, 512)), ((0, 1), (1024, 1), (512, 1))),
    GemmCase("addmm", ((256, 257), (256, 4544), (4544, 257)), ((0, 1), (4544, 1), (257, 1))),
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


class TileConfig(NamedTuple):
    block_m: int
    block_n: int
    num_warps: int
    split_k: int = 1


# The Split-K and Stream-K CI benchmarks have one config, tuned for large square shapes. Here they are tuned per case
# over the grouped kernel's tiles plus Inductor's 128x128 tile, and Split-K also over SPLIT_K.
TILES = ((256, 256, 32), (256, 128, 32), (128, 128, 16), (64, 128, 32))
TUNED_CONFIGS = {
    "manual_splitk": [TileConfig(m, n, w, split_k) for m, n, w in TILES for split_k in (1, 2, 4, 8)],
    "manual_streamk": [TileConfig(m, n, w) for m, n, w in TILES],
}
BLOCK_K = 32
GROUP_M = 4


def storage_size(shape: tuple[int, ...], stride: tuple[int, ...]) -> int:
    """Number of elements a strided tensor spans."""
    return 1 + sum((dim - 1) * step for dim, step in zip(shape, stride))


def make_inputs(case: GemmCase) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.randn(storage_size(shape, stride), device="xpu", dtype=torch.float16).as_strided(shape, stride)
        for shape, stride in zip(case.shapes, case.strides))


def _dedup_gemm_configs(configs: list[GemmConfig]) -> list[GemmConfig]:
    seen = set()
    out = []
    for config in configs:
        key = tuple(asdict(config).items())
        if key not in seen:
            seen.add(key)
            out.append(config)
    return out


def _register_custom_xpu_mm_heuristic() -> None:
    """Add custom GEMM configs to Inductor's XPU mm/bmm/addmm/baddbmm template search space."""
    from torch._inductor.heuristics.template.registry import _HEURISTIC_CACHE, register_template_heuristic
    from torch._inductor.heuristics.template.triton import GemmConfig, XPUMMTemplateConfigHeuristic
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
    class MyXPUMMTemplateConfigHeuristic(XPUMMTemplateConfigHeuristic):

        def __init__(self) -> None:
            super().__init__()
            self.mm_configs = _dedup_gemm_configs([*self.mm_configs, *custom_mm_configs])
            self.exhaustive_configs = self.mm_configs

    @register_template_heuristic(mm_template.uid, "xpu", op_name="addmm")
    @register_template_heuristic(bmm_template.uid, "xpu", op_name="baddbmm")
    class MyXPUAddmmTemplateConfigHeuristic(AddMMConfigMixin, MyXPUMMTemplateConfigHeuristic):  # pylint: disable=unused-variable
        pass

    _HEURISTIC_CACHE.clear()


def _splitk(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, config: TileConfig) -> None:
    """One Split-K GEMM, launched as `gemm_splitk_benchmark.matmul` does but with the given config."""
    from triton_kernels_benchmark.gemm_splitk_benchmark import _kernel

    m, k = a.shape
    n = b.shape[1]
    grid = (triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n), config.split_k)
    _kernel.fn[grid](a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride(), acc_dtype=tl.float32,
                     BLOCK_M=config.block_m, BLOCK_N=config.block_n, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
                     SPLIT_K=config.split_k, num_warps=config.num_warps, num_stages=4, grf_mode="256")


def _streamk(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, config: TileConfig) -> None:
    """One Stream-K GEMM, launched as `gemm_streamk_benchmark.matmul` does but with the given tile."""
    from triton_kernels_benchmark.gemm_streamk_benchmark import first_wave, full_tiles, zero_streamk_tiles

    m, k = a.shape
    n = b.shape[1]
    programs = torch.xpu.get_device_capability(0)["gpu_subslice_count"]
    iters_per_tile = triton.cdiv(k, BLOCK_K)
    total_tiles = triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n)
    streamk_tiles = total_tiles % programs
    if total_tiles - streamk_tiles > programs:
        streamk_tiles += programs
    streamk_iters = streamk_tiles * iters_per_tile
    meta = {
        "BLOCK_SIZE_M": config.block_m, "BLOCK_SIZE_N": config.block_n, "BLOCK_SIZE_K": BLOCK_K, "GROUP_SIZE_M":
        GROUP_M, "num_warps": config.num_warps
    }
    strides = (*a.stride(), *b.stride(), *c.stride())
    if streamk_tiles > 0:
        zero_streamk_tiles[(streamk_tiles, )](c, m, n, k, *c.stride(), **meta)
    first_wave.fn[(programs, )](a, b, c, m, n, k, *strides, streamk_iters // programs, streamk_iters % programs,
                                iters_per_tile, **meta, num_stages=2, grf_mode="256")
    if total_tiles > streamk_tiles:
        full_tiles.fn[(total_tiles - streamk_tiles, )](a, b, c, m, n, k, *strides, streamk_tiles, **meta, num_stages=2,
                                                       grf_mode="256")


@functools.cache
def _addmatrix_kernel():
    """The post-op add-matrix kernel, tuned over the GEMM benchmark's configs like the no-bias cases.

    The strides are in the key, so a shape that comes with several layouts is tuned once per layout.
    """
    from triton_kernels_benchmark.gemm_benchmark import get_matmul_autotune_configs
    from triton_kernels_benchmark.gemm_postop_addmatrix_benchmark import grouped_tile_gemm_addmatrix_kernel

    key = [
        "M", "N", "K", "stride_am", "stride_ak", "stride_bk", "stride_bn", "stride_cm", "stride_cn", "stride_dm",
        "stride_dn"
    ]
    return triton.autotune(configs=get_matmul_autotune_configs(), key=key)(grouped_tile_gemm_addmatrix_kernel.fn)


@triton.jit
def _bias_cast_kernel(x_ptr, bias_ptr, y_ptr, numel, N: tl.constexpr, stride_bias_n: tl.constexpr,
                      HAS_BIAS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """y = x + bias, cast to y's type, in one pass over a contiguous (..., N) FP32 GEMM result."""
    offs = tl.program_id(axis=0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask)
    if HAS_BIAS:
        x += tl.load(bias_ptr + (offs % N) * stride_bias_n, mask=mask).to(tl.float32)
    tl.store(y_ptr + offs, x.to(y_ptr.dtype.element_ty), mask=mask)


def _manual_gemm(mode: str, inputs: tuple[torch.Tensor, ...], config: TileConfig | None = None) -> torch.Tensor:
    from triton_kernels_benchmark.gemm_benchmark import grouped_tile_gemm_kernel, grouped_tile_gemm_kernel_batched

    bias = inputs[0] if len(inputs) == 3 else None
    a, b = inputs[-2:]
    m, k = a.shape[-2:]
    n = b.shape[-1]
    output = torch.empty((*a.shape[:-2], m, n), device=a.device, dtype=a.dtype)

    if mode == "manual_grouped":
        # The kernels add the bias and store FP16 in their epilogue, as Inductor's template does.
        assert all(x.data_ptr() % 16 == 0 for x in (a, b, output)), "Matrix base addresses must be 16-byte aligned"
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]), a.shape[0]
                             if a.ndim == 3 else 1)
        if bias is not None:
            assert a.ndim == 2, "Only 2D GEMMs have a bias"
            _addmatrix_kernel()[grid](a, b, output, bias, m, n, k, *a.stride(), *b.stride(), *output.stride(),
                                      *bias.stride(), tl.float32)
        elif a.ndim == 3:
            grouped_tile_gemm_kernel_batched[grid](a, b, output, a.shape[0], m, n, k, *a.stride(), *b.stride(),
                                                   *output.stride())
        else:
            grouped_tile_gemm_kernel[grid](a, b, output, m, n, k, *a.stride(), *b.stride(), *output.stride())
        return output

    # Split-K and Stream-K reduce partial sums in an FP32 buffer, so the bias add and the cast take one more pass.
    # Split-K accumulates with atomics, so its buffer must start at zero.
    assert bias is None or bias.stride(-2) == 0, "The addmm bias must be a broadcast row"
    alloc = torch.zeros if mode == "manual_splitk" else torch.empty
    acc = alloc(output.shape, device=a.device, dtype=torch.float32)
    assert all(x.data_ptr() % 16 == 0 for x in (a, b, acc)), "Matrix base addresses must be 16-byte aligned"
    matmul = {"manual_splitk": _splitk, "manual_streamk": _streamk}[mode]
    # Both kernels are 2D only, so a bmm runs one GEMM per batch.
    for a_i, b_i, c_i in zip(a, b, acc) if a.ndim == 3 else [(a, b, acc)]:
        matmul(a_i, b_i, c_i, config)
    block = 1024
    bias_args = (bias, bias.stride(-1), True) if bias is not None else (acc, 0, False)
    _bias_cast_kernel[(triton.cdiv(acc.numel(), block), )](acc, bias_args[0], output, acc.numel(), n, *bias_args[1:],
                                                           BLOCK_SIZE=block)
    return output


def inductor_pick(code: list[str]) -> str:
    """Which GEMM Inductor's generated code calls: its Triton template or an ATen kernel."""
    markers = {"Triton": "triton_tem_", "ATen": "extern_kernels."}
    return "+".join(name for name, marker in markers.items() if any(marker in c for c in code)) or "unknown"


def get_benchmark(case_ids: list[int], modes: list[str]) -> Mark:
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
            config = None
            if mode in TUNED_CONFIGS:
                # Pick the fastest config outside the timed region.
                tune_bench = benchmark_suite.get_do_bench(n_warmup=25, n_repeat=10, quantiles=[0.5, 0.0, 1.0])
                config = min(TUNED_CONFIGS[mode], key=lambda c: tune_bench(lambda: _manual_gemm(mode, inputs, c))[3])
                print(f"Case {case}, {MODES[mode]}: picked {config}", flush=True)
            manual_fn = lambda: _manual_gemm(mode, inputs, config)
            benchmark_suite.assert_close(manual_fn, torch_fn, atol=5e-2, rtol=1e-2, err_msg=f"{mode} to torch")
            _, min_ms, max_ms, mean_ms, cv = do_bench(manual_fn)
        else:
            torch.compiler.reset()
            with inductor_config.patch(max_autotune_gemm=True, max_autotune_gemm_backends=INDUCTOR_BACKENDS[mode],
                                       force_disable_caches=True):
                compiled = torch.compile(operation, mode="max-autotune-no-cudagraphs")
                compiled_fn = lambda: compiled(*inputs)
                # Compile and autotune outside the timed region.
                _, code = run_and_get_code(compiled_fn)
                print(f"Case {case}, {MODES[mode]}: Inductor picked {inductor_pick(code)}", flush=True)
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


def uses_descriptor(shape: tuple[int, ...], stride: tuple[int, ...], itemsize: int) -> bool:
    """Whether the manual kernels access this matrix through a tensor descriptor, as `gemm_tile_utils` decides."""
    (rows, cols), (stride_row, stride_col) = shape[-2:], stride[-2:]
    if stride_row == 1 and stride_col != 1:
        rows, cols, stride_row, stride_col = cols, rows, stride_col, stride_row
    return stride_col == 1 and stride_row >= cols and stride_row * itemsize % 16 == 0


def pointer_path_operands(case: GemmCase) -> list[str]:
    """The matrices the manual kernels load or store through masked pointers instead of a tensor descriptor."""
    (a_shape, b_shape), (a_stride, b_stride) = case.shapes[-2:], case.strides[-2:]
    n = b_shape[-1]
    # The manual kernels write a contiguous FP32 output.
    operands = {"A": (a_shape, a_stride, 2), "B": (b_shape, b_stride, 2), "C": ((a_shape[-2], n), (n, 1), 4)}
    return [
        name for name, (shape, stride, itemsize) in operands.items() if not uses_descriptor(shape, stride, itemsize)
    ]


def print_cases(case_ids: list[int]) -> None:
    for case_id in case_ids:
        case = GEMM_CASES[case_id - 1]
        print(f"{case_id}: {case.op}, shapes={case.shapes}, strides={case.strides}", flush=True)
        if pointer := pointer_path_operands(case):
            print(f"   manual kernels use masked pointers for {', '.join(pointer)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    parser.add_argument("--case", type=int, action="append", dest="case_ids",
                        help="one-based case number; repeat to select multiple")
    parser.add_argument("--reports", default="", help="directory for benchmark-suite CSV/plot reports")
    parser.add_argument("--n_runs", type=int, default=1, help="number of benchmark-suite report repetitions")
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
