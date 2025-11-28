"""
Original code by @bertmaher; profiling added by @apgoucher
"""

import argparse
import cProfile
import csv
import os
import pstats
import time

import numpy as np
import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def nop_args(
    t1,
    t2,
    t3,
    t4,
    t5,
    nc1,
    nc2,
    nc3,
    nc4,
    nc5,
    nc6,
    nc7,
    nc8,
    nc9,
    c1: tl.constexpr,
    c2: tl.constexpr,
    c3: tl.constexpr,
    c4: tl.constexpr,
    c5: tl.constexpr,
):
    pass


def do_bench_walltime(fn):
    print("Compiling...")
    fn()
    torch.xpu.synchronize()

    for _ in range(1000):
        fn()
    torch.xpu.synchronize()

    n_repeat = 10000

    mses = []

    for _ in range(25):
        print("Running %d benchmarking iterations..." % n_repeat)
        # Benchmark
        torch.xpu.synchronize()
        start_time = time.time()
        for _ in range(n_repeat):
            fn()
        torch.xpu.synchronize()
        end_time = time.time()
        wall_time_ms = (end_time - start_time) * 1e3 / n_repeat
        mses.append(wall_time_ms)

    mses = np.array(mses)

    print("Running profiler...")
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(n_repeat):
        fn()
    torch.xpu.synchronize()
    profile.disable()
    stats = pstats.Stats(profile)
    stats.sort_stats("time")
    stats.print_stats()
    return mses


def main(use_tensor_desc: bool, reports_dir: str = None):
    if use_tensor_desc:
        targs = [TensorDescriptor.from_tensor(torch.zeros(1, 16, device="xpu"), block_shape=[1, 16]) for _ in range(5)]
    else:
        targs = [torch.zeros(1, device="xpu") for _ in range(5)]
    ncargs = [0, 1, 1024, 2**31 - 1, 2**64 - 1, False, True, None, (16, 16)]
    cargs = [32, False, True, 0, 64]

    usecs = do_bench_walltime(lambda: nop_args[
        1,
    ](*targs, *ncargs, *cargs)) * 1000.0

    print(usecs)
    print(sorted(usecs)[len(usecs) >> 1])

    if reports_dir:
        os.makedirs(reports_dir, exist_ok=True)
        csv_path = os.path.join(reports_dir, "launch_overhead_report.csv")
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["input_type", "median_usecs"])

            input_type = "TensorDescriptor" if use_tensor_desc else "Tensor"
            writer.writerow([input_type, round(sorted(usecs)[len(usecs) >> 1], 2)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark launch overhead for Triton kernels")
    parser.add_argument("--reports", type=str, default=None, help="Path to directory for CSV reports")
    args = parser.parse_args()

    print("launch overhead of kernel with Tensor inputs")
    main(use_tensor_desc=False, reports_dir=args.reports)
    print("launch overhead of kernel with TensorDescriptor inputs")
    main(use_tensor_desc=True, reports_dir=args.reports)
