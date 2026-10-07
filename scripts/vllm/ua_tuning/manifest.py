# SPDX-License-Identifier: Apache-2.0
"""Tune synthetic workloads from a manifest with the shared UA tuner.

Run ``--tune --manifest inputs.json --save-dir profiles`` to generate configs.
Omit ``--tune`` to benchmark existing configs on the same inputs.
Manifests specify shapes and layouts, not CI random values or FP8 scales.
"""

# Exact integer checks reject bools in workload manifests.
# pylint: disable=unidiomatic-typecheck

import importlib.util
import json
import math
from pathlib import Path

from timing import cold_graph_timer

BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmarks/triton_kernels_benchmark/vllm/unified_attention"
# Load the shared tuner without importing the benchmark reporting package.
_spec = importlib.util.spec_from_file_location("tune_unified_attention", BENCHMARK_DIR / "tune_unified_attention.py")
assert _spec is not None and _spec.loader is not None
tuner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tuner)


# pylint: disable-next=too-many-branches
def normalize_workloads(raw):
    if not isinstance(raw, list) or not raw:
        raise ValueError("Manifest must be a nonempty JSON list")
    cases, ids = [], set()
    for index, source in enumerate(raw):
        case = dict(source)
        case.setdefault("id", f"case-{index:04d}")
        if not isinstance(case["id"], str) or case["id"] in ids:
            raise ValueError("Every workload needs a unique string id")
        ids.add(case["id"])
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
                raise ValueError(f"{case['id']}: {name} must be an integer >= {minimum}")
        if case["q_heads"] % case["kv_heads"]:
            raise ValueError("q_heads must be divisible by kv_heads")
        for plural, scalar in (("query_lens", "query_len"), ("kv_lens", "kv_len")):
            if plural not in case:
                if scalar not in case:
                    raise ValueError(f"{case['id']}: provide {plural} or {scalar}")
                case[plural] = [case[scalar]] * case["batch"]
            if len(case[plural]) != case["batch"] or any(
                    type(length) is not int or length <= 0 for length in case[plural]):
                raise ValueError(f"{case['id']}: invalid {plural}")
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


def main(argv=None, *, allocate=tuner.allocate_case, description=__doc__):
    """Supply an allocator to reproduce a fixture instead of synthetic inputs."""
    parser = tuner.create_parser(description=description)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clear-cache", action="store_true",
                        help="Evict 256 MB before each graph replay, excluding eviction from timing")
    parser.set_defaults(seed=1729, save_dir=str(BENCHMARK_DIR / "profiles"))
    args = parser.parse_args(argv)
    workloads = normalize_workloads(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    timer = cold_graph_timer if args.clear_cache else tuner.graph_timer
    scope = "attention_sequence_cold_graph_device" if args.clear_cache else "attention_sequence_graph_device"
    tuner.main(args, workloads, allocate=allocate, timer=timer, timing_scope=scope)


if __name__ == "__main__":
    main()
