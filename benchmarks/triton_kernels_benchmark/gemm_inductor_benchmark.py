#!/usr/bin/env python3
"""Compare TorchInductor ATen/Triton GEMM backends and manual grouped tile-order/Split-K/Stream-K GEMM.

The cases are the FP16 GEMMs for which ATen beat Triton in disassembles/log.txt,
plus new cases from results/qc_aten_selected_20261113/shapes.json.
Pass --case repeatedly to benchmark selected one-based cases, or --list to inspect
their shapes and strides without requiring an XPU. Results use the
triton_kernels_benchmark report format.

On XPU, Inductor enables persistent tensor-descriptor GEMM candidates alongside
the ordinary Triton templates; autotuning selects among eligible candidates.
The XPU search spaces match the pruned QC model configurations from November 13:
13 ordinary MM, 12 BMM, 14 descriptor MM, and 3 persistent MM seeds.

Manual providers reuse the kernels in gemm_benchmark, gemm_splitk_benchmark,
and gemm_streamk_benchmark (not Inductor's grouped_mm). All manual providers use
the original inputs and strides without padding, packing, or copying. The Intel
compiler lowers descriptors to block I/O or masked pointers. Split-K and Stream-K run
one GEMM per batch; their timings include zeroing and all launches. All manual
providers accumulate in FP32, add any addmm bias, and return FP16 outputs.
On XPU, manual grouped GEMM shares the ordinary Inductor tuning configurations
and fuses bias addition and output conversion into the GEMM kernel. Standalone
manual GEMM callers retain their original configurations and output behavior.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from functools import cache
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if TYPE_CHECKING:
    import torch


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
    # Additional ATen-selected cases from the QC split-K-fixed log (before config pruning).
    GemmCase("addmm", ((20480, 256), (20480, 128), (128, 256)), ((0, 1), (128, 1), (256, 1)), 0.026980, 0.028750),
    GemmCase("addmm", ((16384, 256), (16384, 128), (128, 256)), ((0, 1), (128, 1), (256, 1)), 0.021667, 0.021979),
    GemmCase("addmm", ((8192, 256), (8192, 128), (128, 256)), ((0, 1), (128, 1), (256, 1)), 0.011563, 0.012604),
    GemmCase("addmm", ((256, 256), (256, 688), (688, 256)), ((0, 1), (688, 1), (256, 1)), 0.005000, 0.005312),
    GemmCase("mm", ((1228800, 64), (64, 256)), ((65, 1), (256, 1)), 1.271250, 1.318334),
    GemmCase("mm", ((176128, 32), (32, 64)), ((33, 1), (64, 1)), 0.070729, 0.071979),
    GemmCase("addmm", ((256, 256), (256, 2572), (2572, 256)), ((0, 1), (2572, 1), (256, 1)), 0.013229, 0.048646),
    GemmCase("addmm", ((256, 256), (256, 560), (560, 256)), ((0, 1), (560, 1), (256, 1)), 0.005000, 0.005208),
    GemmCase("addmm", ((256, 640), (256, 490), (490, 640)), ((0, 1), (490, 1), (640, 1)), 0.008958, 0.014687),
    GemmCase("addmm", ((256, 640), (256, 5052), (5052, 640)), ((0, 1), (5056, 1), (640, 1)), 0.047604, 0.125208),
    GemmCase("addmm", ((256, 64), (256, 204), (204, 64)), ((0, 1), (204, 1), (64, 1)), 0.005000, 0.007395),
)


def _sync():
    import torch

    if torch.xpu.is_available():
        torch.xpu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def make_inputs(case: GemmCase) -> tuple[torch.Tensor, ...]:
    import torch
    from triton_kernels_benchmark.benchmark_testing import DEVICE

    inputs = []
    for shape, stride in zip(case.shapes, case.strides):
        storage_size = 1 + sum((dim - 1) * step for dim, step in zip(shape, stride))
        storage = torch.randn(storage_size, device=DEVICE, dtype=torch.float16)
        inputs.append(storage.as_strided(shape, stride))
    return tuple(inputs)


@cache
def _register_custom_xpu_mm_heuristic(batched: bool = False):
    """Register and return the shared ordinary GEMM tuning space for XPU providers."""
    try:
        from torch._inductor.template_heuristics.triton import (
            BaseConfig,
            GemmConfig,
            MMTemplateConfigMixin,
            XPUConfigHeuristic,
            XPUDescriptorMMTemplateConfigHeuristic,
            XPUPersistentTMATemplateConfigHeuristic,
        )
        from torch._inductor.template_heuristics.triton_addmm import AddMMConfigMixin
        from torch._inductor.kernel.bmm import bmm_template
        from torch._inductor.kernel.mm import mm_template, persistent_tma_mm_template, xpu_descriptor_mm_template
        from torch._inductor.template_heuristics.registry import _HEURISTIC_CACHE, register_template_heuristic

        mm_configs: list[BaseConfig] = [
            GemmConfig(16, 512, 64, 2, 8),
            GemmConfig(64, 64, 64, 3, 8),
            GemmConfig(64, 128, 32, 4, 8),
            GemmConfig(32, 256, 128, 3, 32),
            GemmConfig(32, 32, 128, 2, 4),
            GemmConfig(32, 64, 32, 5, 8),
            GemmConfig(64, 64, 32, 2, 4),
            GemmConfig(128, 128, 32, 3, 4),
            GemmConfig(64, 32, 32, 5, 8),
            GemmConfig(64, 128, 128, 4, 4),
            GemmConfig(64, 64, 32, 2, 8),
            GemmConfig(128, 128, 64, 2, 32),
            GemmConfig(32, 256, 32, 2, 16),
        ]
        bmm_configs: list[BaseConfig] = [
            GemmConfig(128, 256, 64, 2, 32),
            GemmConfig(128, 512, 32, 2, 32),
            GemmConfig(32, 32, 16, 1, 2),
            GemmConfig(128, 128, 64, 2, 16),
            GemmConfig(64, 64, 16, 2, 4),
            GemmConfig(64, 64, 128, 5, 4),
            GemmConfig(128, 128, 32, 2, 8),
            GemmConfig(128, 128, 32, 3, 4),
            GemmConfig(64, 64, 32, 2, 8),
            GemmConfig(128, 128, 64, 3, 16),
            GemmConfig(512, 128, 32, 1, 32),
            GemmConfig(128, 512, 32, 4, 32, group_m=4),
        ]
        descriptor_configs: list[BaseConfig] = [
            GemmConfig(64, 32, 128, 5, 4),
            GemmConfig(32, 32, 128, 2, 4),
            GemmConfig(64, 32, 32, 5, 8),
            GemmConfig(64, 64, 64, 3, 8),
            GemmConfig(64, 128, 128, 4, 4),
            GemmConfig(32, 64, 32, 5, 8),
            GemmConfig(256, 128, 32, 3, 32),
            GemmConfig(64, 64, 32, 2, 4),
            GemmConfig(64, 128, 32, 3, 4),
            GemmConfig(64, 128, 32, 4, 8),
            GemmConfig(128, 64, 32, 4, 8),
            GemmConfig(32, 64, 128, 2, 2),
            GemmConfig(128, 64, 32, 3, 4),
            GemmConfig(256, 128, 32, 1, 32),
        ]
        persistent_configs: list[BaseConfig] = [
            GemmConfig(128, 128, 64, 3, 8),
            GemmConfig(128, 128, 64, 4, 8),
            GemmConfig(128, 128, 64, 5, 8),
        ]

        @register_template_heuristic(mm_template.uid, "xpu")
        class MyXPUMMTemplateConfigHeuristic(MMTemplateConfigMixin, XPUConfigHeuristic):

            def __init__(self) -> None:
                super().__init__()
                self.mm_configs = list(mm_configs)
                self.exhaustive_configs = self.mm_configs

        @register_template_heuristic(bmm_template.uid, "xpu")
        class MyXPUBMMTemplateConfigHeuristic(MyXPUMMTemplateConfigHeuristic):

            def __init__(self) -> None:
                super().__init__()
                self.mm_configs = list(bmm_configs)
                self.exhaustive_configs = self.mm_configs

        @register_template_heuristic(mm_template.uid, "xpu", op_name="addmm")
        class MyXPUAddmmTemplateConfigHeuristic(AddMMConfigMixin, MyXPUMMTemplateConfigHeuristic):
            pass

        @register_template_heuristic(bmm_template.uid, "xpu", op_name="baddbmm")
        class MyXPUBaddbmmTemplateConfigHeuristic(AddMMConfigMixin, MyXPUBMMTemplateConfigHeuristic):
            pass

        @register_template_heuristic(xpu_descriptor_mm_template.uid, "xpu")
        class MyXPUDescriptorMMTemplateConfigHeuristic(XPUDescriptorMMTemplateConfigHeuristic):

            def __init__(self) -> None:
                super().__init__()
                self.mm_configs = list(descriptor_configs)
                self.exhaustive_configs = self.mm_configs

        @register_template_heuristic(xpu_descriptor_mm_template.uid, "xpu", op_name="addmm")
        class MyXPUDescriptorAddmmTemplateConfigHeuristic(AddMMConfigMixin, MyXPUDescriptorMMTemplateConfigHeuristic):
            pass

        @register_template_heuristic(persistent_tma_mm_template.uid, "xpu")
        class MyXPUPersistentMMTemplateConfigHeuristic(XPUPersistentTMATemplateConfigHeuristic):

            def __init__(self) -> None:
                super().__init__()
                self.mm_configs = list(persistent_configs)
                self.exhaustive_configs = self.mm_configs

        @register_template_heuristic(persistent_tma_mm_template.uid, "xpu", op_name="addmm")
        class MyXPUPersistentAddmmTemplateConfigHeuristic(AddMMConfigMixin, MyXPUPersistentMMTemplateConfigHeuristic):
            pass

        _HEURISTIC_CACHE.clear()
        return bmm_configs if batched else mm_configs
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to register custom XPU MM heuristic: %s", exc)


def _gemm_dimensions(case: GemmCase) -> tuple[int, int, int, int]:
    if case.op == "bmm":
        (batch, m, k), (_, _, n) = case.shapes
        return batch, m, n, k
    if case.op == "mm":
        (m, k), (_, n) = case.shapes
        return 1, m, n, k
    (m, n), (_, k), (_, _) = case.shapes
    return 1, m, n, k


def _memory_bytes(case: GemmCase) -> int:
    batch, m, n, _ = _gemm_dimensions(case)
    input_elements = sum(math.prod(shape) for shape in case.shapes)
    output_elements = batch * m * n
    return 2 * (input_elements + output_elements)


def _perf_rate(bytes_count: float, ms: float) -> float:
    if ms <= 0:
        return float("inf")
    return bytes_count * 1e-9 / (ms * 1e-3)


def _tflops(batch: int, m: int, n: int, k: int, ms: float) -> float:
    if ms <= 0:
        return float("inf")
    flops = 2.0 * batch * m * n * k
    return flops * 1e-12 / (ms * 1e-3)


def _manual_gemm(mode: str, inputs: tuple[torch.Tensor, ...]):
    import torch
    import triton

    bias = inputs[0] if len(inputs) == 3 else None
    a, b = inputs[-2:]
    m, k = a.shape[-2:]
    n = b.shape[-1]
    fused_grouped = mode == "manual_grouped" and a.device.type == "xpu"
    output = torch.empty((*a.shape[:-2], m, n), device=a.device, dtype=a.dtype if fused_grouped else torch.float32)

    if mode == "manual_grouped":
        from triton_kernels_benchmark.gemm_benchmark import (
            get_grouped_kernel_with_inductor_configs,
            has_aligned_matrix_bases,
            matmul_kernel_with_tensor_descriptors,
            matmul_kernel_with_tensor_descriptors_batched,
        )
        epilogue_args = {}
        if fused_grouped:
            if a.ndim == 3:
                matmul_kernel_with_tensor_descriptors_batched = get_grouped_kernel_with_inductor_configs(batched=True)
            else:
                matmul_kernel_with_tensor_descriptors = get_grouped_kernel_with_inductor_configs()
            epilogue_args = {
                "FUSE_EPILOGUE": True,
                "bias_ptr": bias,
                "stride_bias_m": bias.stride(-2) if bias is not None else 0,
                "stride_bias_n": bias.stride(-1) if bias is not None else 0,
            }

        transpose_a, transpose_b = a.stride(-2) == 1, b.stride(-2) == 1
        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
            a.shape[0] if a.ndim == 3 else 1,
        )
        if a.ndim == 3:
            matmul_kernel_with_tensor_descriptors_batched[grid](
                a,
                b,
                output,
                a.shape[0],
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                *output.stride(),
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                A_BASE_ALIGNED=has_aligned_matrix_bases(a),
                B_BASE_ALIGNED=has_aligned_matrix_bases(b),
                C_BASE_ALIGNED=has_aligned_matrix_bases(output),
                **epilogue_args,
            )
        else:
            matmul_kernel_with_tensor_descriptors[grid](
                a,
                b,
                output,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                *output.stride(),
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                A_BASE_ALIGNED=has_aligned_matrix_bases(a),
                B_BASE_ALIGNED=has_aligned_matrix_bases(b),
                C_BASE_ALIGNED=has_aligned_matrix_bases(output),
                **epilogue_args,
            )
    else:
        if mode == "manual_splitk":
            from triton_kernels_benchmark.gemm_splitk_benchmark import matmul
        elif mode == "manual_streamk":
            from triton_kernels_benchmark.gemm_streamk_benchmark import matmul
        else:
            raise ValueError(f"Unsupported manual GEMM mode: {mode}")
        if a.ndim == 3:
            for batch in range(a.shape[0]):
                c_batch = output[batch]
                if mode == "manual_splitk":
                    c_batch.zero_()
                matmul(a[batch], b[batch], c_batch)
        else:
            if mode == "manual_splitk":
                output.zero_()
            matmul(a, b, output)

    if fused_grouped:
        return output
    result = output
    if bias is not None:
        result = result + bias
    return result.to(a.dtype)


def _benchmark_mode(mode: str, operation, inputs: tuple[torch.Tensor, ...]):
    import torch
    import torch._inductor.config as cfg
    import triton_kernels_benchmark as benchmark_suite

    do_bench = benchmark_suite.get_do_bench(n_warmup=200, n_repeat=10, quantiles=[0.5, 0.0, 1.0])

    if mode.startswith("manual_"):
        with torch.no_grad():
            fn = lambda: _manual_gemm(mode, inputs)
            torch.testing.assert_close(fn(), operation(*inputs), rtol=1e-2, atol=5e-2)
            _, min_ms, max_ms, mean_ms, cv = do_bench(fn)
        return {
            "mode": mode,
            "mean_ms": mean_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "cv": cv,
        }

    if mode in {"aten_only", "triton_aten", "triton_only"}:
        torch.compiler.reset()
        backends = {
            "aten_only": "ATEN",
            "triton_aten": "TRITON,ATEN",
            "triton_only": "TRITON",
        }[mode]
        with cfg.patch(
                max_autotune_gemm=True,
                max_autotune_gemm_backends=backends,
                force_disable_caches=True,
                **({"triton.enable_persistent_tma_matmul": True} if inputs[-2].device.type == "xpu" else {}),
        ):
            compiled = torch.compile(
                operation,
                backend="inductor",
                mode="max-autotune-no-cudagraphs",
            )
            with torch.no_grad():
                for _ in range(3):
                    compiled(*inputs)
                _sync()
                torch.testing.assert_close(compiled(*inputs), operation(*inputs), rtol=1e-2, atol=5e-2)
                _, min_ms, max_ms, mean_ms, cv = do_bench(lambda: compiled(*inputs))
        del compiled
        torch.compiler.reset()
        return {
            "mode": mode,
            "mean_ms": mean_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "cv": cv,
        }

    raise ValueError(f"Unsupported benchmark mode: {mode}")


def get_benchmark(case_ids: list[int], modes: list[str]):
    import triton_kernels_benchmark as benchmark_suite

    labels = {
        "aten_only": "Inductor-ATEN",
        "triton_aten": "Inductor-TRITON+ATEN",
        "triton_only": "Inductor-TRITON",
        "manual_grouped": "Manual-Triton-grouped-tiles",
        "manual_splitk": "Manual-Triton-Split-K",
        "manual_streamk": "Manual-Triton-Stream-K",
    }

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["case"],
            x_vals=case_ids,
            line_arg="mode",
            line_vals=modes,
            line_names=[labels[mode] for mode in modes],
            styles=[("blue", "-"), ("green", "-"), ("green", "--"), ("orange", "-"), ("red", "-"),
                    ("purple", "-")][:len(modes)],
            ylabel=["GB/s", "TFlops"],
            plot_name="gemm-aten-winning-cases",
            args={},
        ))
    def benchmark(case: int, mode: str):
        import torch

        gemm_case = GEMM_CASES[case - 1]
        inputs = make_inputs(gemm_case)
        operation = getattr(torch, gemm_case.op)
        result = _benchmark_mode(mode, operation, inputs)
        batch, m, n, k = _gemm_dimensions(gemm_case)
        bytes_count = _memory_bytes(gemm_case)

        gbps = lambda ms: _perf_rate(bytes_count, ms)
        tflops = lambda ms: _tflops(batch, m, n, k, ms)
        return (
            (gbps(result["mean_ms"]), gbps(result["max_ms"]), gbps(result["min_ms"])),
            (tflops(result["mean_ms"]), tflops(result["max_ms"]), tflops(result["min_ms"])),
            result["cv"],
        )

    return benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode", choices=[
            "aten_only",
            "triton_aten",
            "triton_only",
            "manual_grouped",
            "manual_splitk",
            "manual_streamk",
            "all",
        ], default="all")
    parser.add_argument(
        "--case",
        type=int,
        action="append",
        dest="case_ids",
        help="one-based case number; repeat to select multiple",
    )
    parser.add_argument("--reports", default="", help="directory for benchmark-suite CSV/plot reports")
    parser.add_argument("--n-runs", type=int, default=1, help="number of benchmark-suite report repetitions")
    parser.add_argument("--list", action="store_true", help="list cases without requiring an XPU")
    args = parser.parse_args()

    case_ids = args.case_ids or list(range(1, len(GEMM_CASES) + 1))
    if any(case_id <= 0 or case_id > len(GEMM_CASES) for case_id in case_ids):
        parser.error(f"--case must be between 1 and {len(GEMM_CASES)}")
    if args.list:
        for case_id in case_ids:
            case = GEMM_CASES[case_id - 1]
            print(f"{case_id}: {case.op}, shapes={case.shapes}, strides={case.strides}")
            print(f"   log winner: ATen={case.logged_aten_ms:.6f} ms, "
                  f"best Triton={case.logged_triton_ms:.6f} ms")
        return

    import torch

    if not torch.xpu.is_available():
        raise RuntimeError("This benchmark requires an XPU-enabled PyTorch installation and an available Intel GPU")
    torch.manual_seed(0)
    if args.mode in {"all", "triton_aten", "triton_only"}:
        _register_custom_xpu_mm_heuristic()
    print(f"Device: {torch.xpu.get_device_name()}", flush=True)
    for case_id in case_ids:
        case = GEMM_CASES[case_id - 1]
        print(f"\nCase {case_id}: {case.op}, shapes={case.shapes}, strides={case.strides}", flush=True)
        print(
            f"  log winner: ATen={case.logged_aten_ms:.6f} ms, "
            f"best Triton={case.logged_triton_ms:.6f} ms",
            flush=True,
        )
    if args.mode == "all":
        modes = ["aten_only", "triton_only", "manual_grouped", "manual_splitk", "manual_streamk"]
    else:
        modes = [args.mode]
    from triton_kernels_benchmark.benchmark_testing import MarkArgs

    get_benchmark(case_ids, modes).run(
        show_plots=False,
        print_data=True,
        mark_args=MarkArgs(reports=args.reports, n_runs=args.n_runs),
    )


if __name__ == "__main__":
    main()
