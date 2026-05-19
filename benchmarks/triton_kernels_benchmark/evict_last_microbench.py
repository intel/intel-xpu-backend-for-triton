#!/usr/bin/env python3
"""
Microbenchmark for EVICT_LAST cache control promotion.

Isolates the L1-keep effect on DPAS-A loads in a K-loop with warp-broadcast
on K (the canonical pattern Phase 2 promotes).

Link to plan: /home/jovyan/.claude/plans/functional-weaving-seahorse.md §3.1

This benchmark is one-shot perf-evidence for the evict_last optimization,
NOT a CI-wired benchmark. It does NOT integrate into triton-benchmarks.yml.
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl
from triton_kernels_benchmark.benchmark_testing import do_bench_elapsed_time


@triton.jit
def gemm_a_reuse_kernel(  # pylint: disable=unused-argument
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    GEMM kernel with A-tile reuse in the K-loop.

    The A-tile load is the candidate for evict_last promotion:
    - warp-broadcast on K dimension (spatial known reuse)
    - streaming address in K-loop (temporal side reports Streaming)
    - feeds tt.dot directly
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)  # << candidate for evict_last
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))


def run_kernel(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, device, verify=False):
    """Launch the kernel once and optionally verify correctness."""
    a = torch.randn(M, K, dtype=torch.float16, device=device)
    b = torch.randn(K, N, dtype=torch.float16, device=device)
    c = torch.empty(M, N, dtype=torch.float16, device=device)

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    gemm_a_reuse_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps=num_warps,
    )

    if verify:
        c_ref = (a.to(torch.float32) @ b.to(torch.float32)).to(torch.float16)
        torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)

    return a, b, c


def bench_shape(
    M,
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    num_warps,
    device,
    repeat_runs,
    verify_first_run,
):
    """
    Run repeat_runs independent measurements and return mean/stdev.

    Each repeat_run is one do_bench_elapsed_time call producing one mean.
    """
    means_ms = []
    for run_idx in range(repeat_runs):
        a, b, c = run_kernel(
            M,
            N,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_warps,
            device,
            verify=verify_first_run and run_idx == 0,
        )

        # Default args bind a/b/c to the current iteration's tensors —
        # avoids pylint cell-var-from-loop and is the idiomatic fix.
        def launch_fn(a=a, b=b, c=c):
            grid = lambda meta: (  # noqa: E731
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(N, meta["BLOCK_N"]),
            )
            gemm_a_reuse_kernel[grid](
                a,
                b,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                num_warps=num_warps,
            )

        mean_ms = do_bench_elapsed_time(
            launch_fn,
            n_warmup=100,
            n_repeat=200,
            return_mode="mean",
            device=device,
        )
        means_ms.append(mean_ms)

    mean_of_means = sum(means_ms) / len(means_ms)
    if len(means_ms) > 1:
        stdev_of_means = (sum((x - mean_of_means)**2 for x in means_ms) / (len(means_ms) - 1))**0.5
    else:
        stdev_of_means = 0.0

    return mean_of_means, stdev_of_means


def count_evict_last(dump_dir: Optional[Path]) -> int:
    """
    Count evict_last occurrences in *.ttgir files under dump_dir.

    Returns -1 if TRITON_KERNEL_DUMP / TRITON_DUMP_DIR not set.
    """
    if dump_dir is None:
        return -1
    if not dump_dir.exists():
        return -1

    count = 0
    for ttgir_file in dump_dir.glob("**/*.ttgir"):
        with open(ttgir_file, encoding="utf-8") as f:
            for line in f:
                if "evict_last" in line:
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Microbenchmark for EVICT_LAST promotion")
    parser.add_argument(
        "--shape",
        choices=["primary", "streaming", "budget-overflow", "n-sweep"],
        required=True,
        help="Shape configuration to benchmark",
    )
    parser.add_argument("--device", default="xpu", help="Device to run on")
    parser.add_argument(
        "--repeat-runs",
        type=int,
        default=5,
        help="Number of independent measurement runs",
    )
    parser.add_argument(
        "--csv",
        default="/tmp/evict_last_micro/results.csv",
        help="CSV output path",
    )
    parser.add_argument("--build-label", default="unknown", help="Build label (baseline/patched)")
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify correctness on first run",
    )
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    args = parser.parse_args()

    # Shape configurations per plan §3.1
    if args.shape == "primary":
        configs = [{
            "M": 256,
            "N": 4096,
            "K": 4096,
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 32,
            "num_warps": 4,
        }]
    elif args.shape == "streaming":
        configs = [{
            "M": 4096,
            "N": 4096,
            "K": 64,
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 32,
            "num_warps": 4,
        }]
    elif args.shape == "budget-overflow":
        configs = [{
            "M": 512,
            "N": 512,
            "K": 512,
            "BLOCK_M": 256,
            "BLOCK_N": 256,
            "BLOCK_K": 64,
            "num_warps": 4,
        }]
    elif args.shape == "n-sweep":
        # Hold M=256, BLOCK_M=128, BLOCK_K=32, K=4096 fixed; sweep N
        configs = [{
            "M": 256,
            "N": N,
            "K": 4096,
            "BLOCK_M": 128,
            "BLOCK_N": 256,
            "BLOCK_K": 32,
            "num_warps": 4,
        } for N in [512, 1024, 2048, 4096, 8192]]
    else:
        raise ValueError(f"Unknown shape: {args.shape}")

    # Prepare CSV output directory
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    # Check if TRITON_KERNEL_DUMP and TRITON_DUMP_DIR are set
    dump_dir = None
    if os.environ.get("TRITON_KERNEL_DUMP") == "1":
        dump_dir_str = os.environ.get("TRITON_DUMP_DIR")
        if dump_dir_str:
            dump_dir = Path(dump_dir_str)

    # Run benchmarks
    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "timestamp",
            "build_label",
            "shape_name",
            "M",
            "N",
            "K",
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
            "mean_ms",
            "stdev_ms",
            "mean_tflops",
            "stdev_tflops",
            "tritongpu_evict_last_count",
            "notes",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for cfg in configs:
            m_dim = cfg["M"]
            n_dim = cfg["N"]
            k_dim = cfg["K"]
            block_m = cfg["BLOCK_M"]
            block_n = cfg["BLOCK_N"]
            block_k = cfg["BLOCK_K"]
            num_warps = cfg["num_warps"]
            print(f"Running {args.shape}: M={m_dim}, N={n_dim}, K={k_dim}, "
                  f"BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}, "
                  f"num_warps={num_warps}, repeat_runs={args.repeat_runs}")

            mean_ms, stdev_ms = bench_shape(
                cfg["M"],
                cfg["N"],
                cfg["K"],
                cfg["BLOCK_M"],
                cfg["BLOCK_N"],
                cfg["BLOCK_K"],
                cfg["num_warps"],
                args.device,
                args.repeat_runs,
                args.verify,
            )

            # Compute TFLOPS: 2 * M * N * K / (mean_ms * 1e-3) * 1e-12
            tflops = 2 * cfg["M"] * cfg["N"] * cfg["K"] / (mean_ms * 1e-3) * 1e-12
            stdev_tflops = (2 * cfg["M"] * cfg["N"] * cfg["K"] / (mean_ms * 1e-3) * 1e-12 *
                            (stdev_ms / mean_ms) if mean_ms > 0 else 0.0)

            # Count evict_last in TTGIR
            evict_last_count = count_evict_last(dump_dir)

            row = {
                "timestamp": datetime.now().isoformat(),
                "build_label": args.build_label,
                "shape_name": args.shape,
                "M": cfg["M"],
                "N": cfg["N"],
                "K": cfg["K"],
                "BLOCK_M": cfg["BLOCK_M"],
                "BLOCK_N": cfg["BLOCK_N"],
                "BLOCK_K": cfg["BLOCK_K"],
                "num_warps": cfg["num_warps"],
                "mean_ms": f"{mean_ms:.6f}",
                "stdev_ms": f"{stdev_ms:.6f}",
                "mean_tflops": f"{tflops:.3f}",
                "stdev_tflops": f"{stdev_tflops:.3f}",
                "tritongpu_evict_last_count": evict_last_count,
                "notes": "",
            }
            writer.writerow(row)
            print(f"  Result: {mean_ms:.3f} +/- {stdev_ms:.3f} ms, "
                  f"{tflops:.3f} +/- {stdev_tflops:.3f} TFLOPS, "
                  f"evict_last count: {evict_last_count}")

    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
