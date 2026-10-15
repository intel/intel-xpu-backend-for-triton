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
        for options in (("--check", ), ()):
            subprocess.run(["git", "apply", *options, str(PATCH)], cwd=folder, check=True)
        yield target


def load_tuner():
    with prepared_tuner() as path:
        spec = importlib.util.spec_from_file_location("tune_unified_attention", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load patched tuner from {path}")
        tuner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tuner)
    return tuner


# pylint: disable-next=too-many-branches
def normalize_workloads(raw):
    if not isinstance(raw, list) or not raw:
        raise ValueError("Manifest must be a nonempty JSON list")
    cases, ids = [], set()
    for index, source in enumerate(raw):
        case = dict(source)
        case_id = case.setdefault("id", f"case-{index:04d}")
        if not isinstance(case_id, str) or case_id in ids:
            raise ValueError("Every workload needs a unique string id")
        ids.add(case_id)
        case.setdefault("q_layout", "contiguous")
        if case["q_layout"] not in ("contiguous", "qkv"):
            raise ValueError("q_layout must be contiguous or qkv")
        case.setdefault("kv_layout", "contiguous")
        if case["kv_layout"] not in ("contiguous", "interleaved"):
            raise ValueError("kv_layout must be contiguous or interleaved")
        case.setdefault("dtype", "bfloat16")
        case["dtype"] = {"bf16": "bfloat16", "fp16": "float16", "fp8":
                         "float8_e4m3fn"}.get(case["dtype"], case["dtype"])
        if case["dtype"] not in ("bfloat16", "float16", "float8_e4m3fn"):
            raise ValueError("Supported query/K/V dtypes: bfloat16, float16, float8_e4m3fn")
        case.setdefault("out_dtype", "bfloat16" if case["dtype"].startswith("float8") else case["dtype"])
        if case["out_dtype"] not in ("bfloat16", "float16"):
            raise ValueError("Output dtype must be bfloat16 or float16")
        for name, default in (
            ("batch", 1),
            ("block_size", 32),
            ("head_size", 128),
            ("q_heads", 32),
            ("kv_heads", 8),
            ("num_segments", 16),
            ("seq_threshold_3d", 32),
        ):
            case.setdefault(name, default)
            minimum = 0 if name == "seq_threshold_3d" else 1
            if type(case[name]) is not int or case[name] < minimum:
                raise ValueError(f"{case_id}: {name} must be an integer >= {minimum}")
        if case["q_heads"] % case["kv_heads"]:
            raise ValueError("q_heads must be divisible by kv_heads")
        for plural, scalar in (("query_lens", "query_len"), ("kv_lens", "kv_len")):
            if plural not in case:
                if scalar not in case:
                    raise ValueError(f"{case_id}: provide {plural} or {scalar}")
                case[plural] = [case[scalar]] * case["batch"]
            if len(case[plural]) != case["batch"] or any(
                    type(length) is not int or length <= 0 for length in case[plural]):
                raise ValueError(f"{case_id}: invalid {plural}")
        if any(q > k for q, k in zip(case["query_lens"], case["kv_lens"])):
            raise ValueError("Every query length must be <= its KV length")
        case.setdefault("sliding_window", 0)
        case.setdefault("softcap", 0.0)
        if type(case["sliding_window"]) is not int or case["sliding_window"] < 0:
            raise ValueError("sliding_window must be a nonnegative integer")
        if not isinstance(case["softcap"], (int, float)) or not math.isfinite(case["softcap"]) or case["softcap"] < 0:
            raise ValueError("softcap must be finite and nonnegative")
        if case["block_size"] % 16:
            raise ValueError("The initial TD workloads require block_size divisible by 16")
        if case["dtype"].startswith("float8_") and case["block_size"] % 32:
            raise ValueError("Native FP8 + TD requires block_size divisible by 32; pointer fallback is disabled")
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
    expected = {
        "q_layout": "contiguous",
        "kv_layout": "contiguous",
        "out_dtype": "bfloat16",
        "seq_threshold_3d": 32,
        "num_segments": 16,
    }
    for name, value in expected.items():
        if case[name] != value:
            raise ValueError(f"CI fixtures require {name}={value!r}")
    if case["dtype"] not in ("bfloat16", "float8_e4m3fn"):
        raise ValueError("CI fixtures support BF16 and FP8 inputs")
    if case.get("softmax_scale", case["head_size"]**-0.5) != case["head_size"]**-0.5:
        raise ValueError("CI fixtures require softmax_scale=head_size**-0.5")
    if "num_blocks" not in case or case["num_blocks"] <= 0:
        raise ValueError("CI fixtures require a positive num_blocks")
    previous_device = torch.get_default_device()
    try:
        _benchmark_fixture(str(device))(q_heads=case["q_heads"], k_heads=case["kv_heads"], head_size=case["head_size"],
                                        qdtype=torch.float8_e4m3fn if case["dtype"] == "float8_e4m3fn" else None,
                                        seq_lens=list(zip(case["query_lens"],
                                                          case["kv_lens"])), sliding_window=case["sliding_window"]
                                        or None, soft_cap=case["softcap"] or None, num_blocks=case["num_blocks"],
                                        block_size=case["block_size"], provider="triton-td")
    except _CapturedAttention as result:
        return result.args[0]
    finally:
        torch.set_default_device(previous_device)
    raise RuntimeError("CI fixture did not reach the attention call")


def main(argv=None):
    patch_parser = argparse.ArgumentParser(add_help=False)
    patch_parser.add_argument("--check-patch", action="store_true")
    patch_args, remaining = patch_parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if patch_args.check_patch:
        if remaining:
            patch_parser.error("--check-patch cannot be combined with tuning options")
        with prepared_tuner():
            print("UA CI tuner patch applies cleanly")
        return

    from timing import cold_graph_timer

    tuner = load_tuner()
    parser = tuner.create_parser(description=__doc__)
    parser.add_argument("--check-patch", action="store_true",
                        help="Check tuner patch compatibility without importing PyTorch or vLLM")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clear-cache", action="store_true",
                        help="Evict 256 MB before each graph replay, excluding eviction from timing")
    parser.set_defaults(seed=1729, save_dir=str(BENCHMARK_DIR / "configs"))
    args = parser.parse_args(argv)
    workloads = normalize_workloads(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    timer = cold_graph_timer if args.clear_cache else tuner.graph_timer
    scope = "attention_sequence_cold_graph_device" if args.clear_cache else "attention_sequence_graph_device"
    tuner.main(args, workloads, allocate=allocate_ci, timer=timer, timing_scope=scope)


if __name__ == "__main__":
    main()
