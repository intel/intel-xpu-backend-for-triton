# SPDX-License-Identifier: Apache-2.0
"""Tune or benchmark CI fixtures with --manifest inputs.json [--tune --clear-cache]."""

# Exact integer checks reject bools; deferred GPU imports allow standalone patch checks.
# pylint: disable=unidiomatic-typecheck,import-outside-toplevel

import argparse
import ast
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from types import SimpleNamespace

BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmarks/triton_kernels_benchmark/vllm/unified_attention"
PATCH = Path(__file__).with_name("tuner-ci.patch")


@contextmanager
def prepared_tuner():
    with tempfile.TemporaryDirectory(prefix="ua-ci-tuner-") as folder:
        target = Path(folder) / "tune_unified_attention.py"
        shutil.copyfile(BENCHMARK_DIR / target.name, target)
        subprocess.run(["git", "apply", str(PATCH)], cwd=folder, check=True)
        yield target


def load_tuner():
    with prepared_tuner() as path:
        spec = importlib.util.spec_from_file_location("tune_unified_attention", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load patched tuner from {path}")
        tuner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tuner)
    return tuner


def normalize_workloads(raw):
    """Validate the benchmark's arguments, plus an optional workload id."""
    if not isinstance(raw, list):
        raise ValueError("Manifest must be a JSON list")
    fields = {
        "q_heads", "k_heads", "head_size", "qdtype", "seq_lens", "sliding_window", "soft_cap", "num_blocks",
        "block_size"
    }
    cases, ids = [], set()
    for index, source in enumerate(raw):
        if not isinstance(source, dict) or not fields <= source.keys() or source.keys() - fields - {"id"}:
            raise ValueError(f"Workload {index} must contain benchmark arguments: {sorted(fields)}")
        case = dict(source)
        case_id = case.setdefault("id", f"case-{index:04d}")
        if not isinstance(case_id, str) or case_id in ids:
            raise ValueError("Every workload needs a unique string id")
        ids.add(case_id)
        for name in ("q_heads", "k_heads", "head_size", "num_blocks", "block_size"):
            if type(case[name]) is not int or case[name] <= 0:
                raise ValueError(f"{case_id}: {name} must be a positive integer")
        if case["q_heads"] % case["k_heads"]:
            raise ValueError("q_heads must be divisible by k_heads")
        if case["qdtype"] not in (None, "float8_e4m3fn"):
            raise ValueError("CI fixtures require qdtype=null (BF16) or float8_e4m3fn")
        if case["block_size"] % (32 if case["qdtype"] else 16):
            raise ValueError("TD block_size must be divisible by 16 (32 for FP8)")
        pairs = case["seq_lens"]
        if not isinstance(pairs, list) or not pairs or any(not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(
                type(length) is not int for length in pair) or not 0 < pair[0] <= pair[1] for pair in pairs):
            raise ValueError(f"{case_id}: seq_lens must contain positive [Q, KV] pairs with Q <= KV")
        window = case["sliding_window"]
        if window is not None and (type(window) is not int or window <= 0):
            raise ValueError("sliding_window must be null or a positive integer")
        cap = case["soft_cap"]
        if cap is not None and (type(cap) not in (int, float) or not math.isfinite(cap) or cap < 0):
            raise ValueError("soft_cap must be null or finite and nonnegative")
        cases.append(case)
    return cases


class _CapturedAttention(Exception):
    """Stop the CI fixture before attention executes."""


def _capture(**kwargs):
    raise _CapturedAttention(kwargs)


@cache
def _benchmark_fixture(device):
    import torch
    import triton

    # Extract the fixture without importing CI's provider and reporting setup.
    path = BENCHMARK_DIR / "unified_attention_benchmark.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "benchmark")
    node.decorator_list = []
    namespace = {
        "torch": torch,
        "triton": triton,
        "DEVICE": device,
        "BENCHMARKING_CONFIG": {"verify": False},
        "is_td_patched": True,
        "unified_attention": _capture,
        "benchmark_suite": SimpleNamespace(assert_close=lambda fn, *args, **kwargs: fn()),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # pylint: disable=exec-used
    return namespace["benchmark"]


def allocate_ci(case, device, seed):
    """Use CI's fixed seed, tensor layouts, scales and scratch-buffer sizes."""
    import torch

    del seed  # The fixture sets its own seed, independently of tuning order.
    kwargs = {name: value for name, value in case.items() if name != "id"}
    kwargs["qdtype"] = getattr(torch, kwargs["qdtype"]) if kwargs["qdtype"] else None
    previous_device = torch.get_default_device()
    try:
        _benchmark_fixture(str(device))(**kwargs, provider="triton-td")
    except _CapturedAttention as result:
        return result.args[0]
    finally:
        torch.set_default_device(previous_device)
    raise RuntimeError("CI fixture did not reach the attention call")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--check-patch", action="store_true",
                        help="Check tuner patch compatibility without importing PyTorch or vLLM")
    inputs.add_argument("--manifest")
    parser.add_argument("--tune", action="store_true", help="Search candidates and export winners")
    parser.add_argument("--save-dir", default=str(BENCHMARK_DIR / "configs"))
    parser.add_argument("--measurements", help="Optional detailed tuning checkpoint JSON")
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--confirmation-rounds", type=int, default=5)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--warmup-ms", type=float, default=10)
    parser.add_argument("--rep-ms", type=float, default=50)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Worst-regret tie tolerance before geometric-mean tie-breaking")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Evict 256 MB before each graph replay, excluding eviction from timing")
    args = parser.parse_args(argv)
    if args.check_patch:
        if len(argv) != 1:
            parser.error("--check-patch cannot be combined with tuning options")
        with prepared_tuner():
            print("UA CI tuner patch applies cleanly")
        return
    for name in ("rounds", "warmup_ms", "rep_ms", "confirmation_rounds", "finalists"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            option = name.replace("_", "-")
            parser.error(f"--{option} must be finite and positive")
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("--tolerance must be finite and nonnegative")
    if args.measurements and not args.tune:
        parser.error("--measurements requires --tune")
    workloads = normalize_workloads(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    if not workloads:
        print("No inputs for this part; skipping tuning or benchmarking")
        return

    tuner = load_tuner()
    timer = tuner.benchmark_callable
    if args.clear_cache:
        from timing import cold_graph_timer

        timer = cold_graph_timer
    scope = "attention_sequence_cold_graph_device" if args.clear_cache else "attention_sequence_graph_device"
    tuner.main(args, workloads, allocate=allocate_ci, timer=timer, timing_scope=scope)


if __name__ == "__main__":
    main()
