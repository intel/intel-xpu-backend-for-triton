"""Reverse scan A/B for intel/intel-xpu-backend-for-triton#8887.

The CI benchmarks time only forward scans (`prefix_sums`). This script times reverse
`tl.associative_scan` the way the CI times kernels (`triton_kernels_benchmark.do_bench`,
device time only) and checks every result against torch.

Run it with each Triton build, then compare:

    python scripts/reverse_scan_bench.py --out main.jsonl   # with main
    python scripts/reverse_scan_bench.py --out pr.jsonl     # with #8887
    python scripts/reverse_scan_bench.py --compare main.jsonl pr.jsonl

--compare also takes globs (e.g. 'main.r*.jsonl') to use the median of several runs.
All modes need torch, Triton and `triton_kernels_benchmark` importable.
"""
import argparse
import collections
import glob
import json
import math
import statistics

import torch
import triton
import triton.language as tl
import triton_kernels_benchmark as benchmark_suite
import triton_kernels_benchmark.benchmark_testing as benchmark_testing

# Keep the raw device times of each do_bench call: their interquartile mean resolves
# differences below the timer resolution of these sub-microsecond kernels.
_raw_times = []
_summarize = benchmark_testing._summarize_statistics


def _keep_raw_times(times, quantiles, return_mode):
    _raw_times[:] = [times.clone()]
    return _summarize(times, quantiles, return_mode)


benchmark_testing._summarize_statistics = _keep_raw_times


@triton.jit
def _add(a, b):
    return a + b


@triton.jit
def _max(a, b):
    return tl.maximum(a, b)


@triton.jit
def scan_rows(x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr, OP: tl.constexpr):
    # Each program reverse-scans one [M, N] tile along its rows.
    offs = tl.program_id(0) * M * N + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + offs)
    if OP == "max":
        y = tl.associative_scan(x, 1, _max, reverse=True)
    else:
        y = tl.associative_scan(x, 1, _add, reverse=True)
    tl.store(y_ptr + offs, y)


def configs():
    # The scan axis always covers the whole sub-group, the case where main used the
    # sub-group scan builtin (flipping the lanes around it).
    for lanes in (16, 32):
        for warps in (1, 4):
            for elems_per_lane in (1, 2, 4, 8):
                for dtype, op in (("float32", "add"), ("int32", "add"), ("float16", "add"), ("int32", "max"),
                                  ("float32", "max")):
                    for programs in (1, 4096):
                        yield dict(lanes=lanes, warps=warps, N=lanes * warps * elems_per_lane, dtype=dtype, op=op,
                                   programs=programs)


def reference(x, op):
    x = x.flip(1)
    y = torch.cummax(x, dim=1).values if op == "max" else torch.cumsum(x.double() if x.is_floating_point() else x, 1)
    return y.flip(1).to(x.dtype)


def run(out):
    do_bench = benchmark_suite.get_do_bench(n_warmup=200, n_repeat=300, quantiles=[0.5, 0.0, 1.0])
    with open(out, "w") as f:
        for c in configs():
            torch.manual_seed(0)
            dtype = getattr(torch, c["dtype"])
            shape = (c["programs"], c["N"])
            x = (torch.rand(shape, device="xpu", dtype=dtype)
                 if dtype.is_floating_point else torch.randint(-1000, 1000, shape, device="xpu", dtype=dtype))
            y = torch.empty_like(x)
            fn = lambda: scan_rows[(c["programs"], )](x, y, 1, c["N"], c["op"], num_warps=c["warps"],
                                                       warp_size=c["lanes"])
            fn()
            ref = reference(x, c["op"])
            tol = dict(rtol=2e-2, atol=1.0) if dtype == torch.float16 else dict(rtol=1e-4, atol=1e-3 * c["N"]**0.5)
            ok = torch.allclose(y.float(), ref.float(), **tol) if dtype.is_floating_point else torch.equal(y, ref)
            do_bench(fn)
            t = torch.sort(_raw_times[0]).values
            iqm_us = t[t.numel() // 4:t.numel() - t.numel() // 4].mean().item() * 1e3
            rec = dict(c, time_us=round(iqm_us, 4), correct=bool(ok))
            f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)


def load(patterns):
    times = collections.defaultdict(list)
    for path in sorted(p for pattern in patterns.split(",") for p in glob.glob(pattern)):
        for line in open(path):
            r = json.loads(line)
            assert r["correct"], f"wrong result in {path}: {r}"
            times[tuple(r[k] for k in ("lanes", "warps", "N", "dtype", "op", "programs"))].append(r["time_us"])
    return {k: statistics.median(v) for k, v in times.items()}


def compare(main_runs, pr_runs):
    main, pr = load(main_runs), load(pr_runs)
    ratios = {k: pr[k] / main[k] for k in main.keys() & pr.keys()}
    geomean = math.exp(sum(map(math.log, ratios.values())) / len(ratios))
    print(f"{len(ratios)} reverse scans, PR / main time: geomean {geomean:.3f}, "
          f"best {min(ratios.values()):.3f}, worst {max(ratios.values()):.3f}")
    print("slowest for the PR (lanes, warps, N, dtype, op, programs):")
    for k in sorted(ratios, key=ratios.get, reverse=True)[:5]:
        print(f"  {ratios[k]:.3f}  {k}  main {main[k]:.3f} us, PR {pr[k]:.3f} us")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="write the timings of the Triton build in use to this file")
    parser.add_argument("--compare", nargs=2, metavar=("MAIN", "PR"), help="compare two outputs")
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args.out or "reverse_scan.jsonl")


if __name__ == "__main__":
    main()
