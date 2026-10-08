# SPDX-License-Identifier: Apache-2.0
r"""Tune XPU attention with graph replay and export device-specific JSON configs.

Generate attention workloads from model configuration and requested batches.
Measure candidates over three rounds using graph replay. Replay counts adapt
to measured latency, with fewer calls captured per graph for slow candidates.

Select one config for workloads sharing a selection key. Minimize the largest
slowdown relative to each workload's best measured config, then prefer the
lowest geometric-mean latency ratio among candidates within the tolerance.

Run from this script's directory::

    python tune_unified_attention.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
      --tp-size 2 --batch-size 1 8 32 --query-len 1 --kv-len 1024 \
      --tune --save-dir configs

The script reads model configuration without loading weights and generates
synthetic attention tensors. Use ``--batch-size``, ``--query-len`` and
``--kv-len`` for workload ranges. Model inputs default to QKV-sliced queries
and interleaved KV; use ``--q-layout contiguous --kv-layout contiguous`` for
separate contiguous tensors. Omit ``--tune`` to benchmark existing configs
from ``--save-dir``.

For variable-length batches, pass ``--workloads workloads.json`` with a JSON list::

    [
      [[1, 2048], [128, 1024], [512, 4096]],
      [[1, 512], [1, 8192]]
    ]

Each pair is [Q, KV]. The file replaces the sequence/batch range options;
model properties still come from ``--model``. No file is needed for CLI inputs.

Set ``VLLM_TRITON_USE_TD=0`` for pointer loads; unset or ``1`` uses TD.
Each run tunes one mode, with separate configs for TD and pointer loads.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from itertools import product
from pathlib import Path

import torch
from vllm import envs
from vllm.transformers_utils.config import get_config, get_hf_text_config
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.ops import triton_unified_attention_config as runtime
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

NUM_ROUNDS = 3
WARMUP_MS = 10
REP_MS = 50
SELECTION_TOLERANCE = 0.01


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def candidate_configs(key):
    """Generate candidate configs before legality pruning."""
    # MAX preserves the model's full GQA group, including non-power-of-two ratios.
    heads = ["MAX"] + [h for h in (1, 8) if h < key.num_queries_per_kv and key.num_queries_per_kv % h == 0]
    return [{"block_m": m, "tile_size": t, "num_warps": w, "num_stages": s, "grf_mode": g, "heads_per_program": h}
            for h in heads
            for m in (16, 32, 64, 128)
            for t in (16, 32, 64, 128)
            for w in (2, 4, 8, 16)
            for s in (1, 2, 3)
            for g in ("default", "256")]


def config_id(config):
    return canonical(config)


def model_windows(config, override):
    """Return distinct window sizes from the override or model, 0 means full attention."""
    if override is not None:
        return list(dict.fromkeys(override))
    window = getattr(config, "sliding_window", None) or 0
    if not getattr(config, "use_sliding_window", True) or getattr(config, "max_window_layers", None) == 0:
        return [0]
    if not isinstance(window, int):
        raise ValueError("Specify --sliding-window for models with non-scalar window settings")
    layer_types = getattr(config, "layer_types", None)
    if layer_types:
        if set(layer_types) - {"full_attention", "sliding_attention"}:
            raise ValueError("Model input currently supports full/sliding MHA and GQA layers")
        return list(dict.fromkeys(window if kind == "sliding_attention" else 0 for kind in layer_types))
    window_layers = getattr(config, "max_window_layers", None)
    if window and window_layers is not None and window_layers < config.num_hidden_layers:
        return [0, window]
    if window and getattr(config, "sliding_window_pattern", None):
        return [0, window]
    return [window]


def model_dimensions(config, tp_size):
    """Return per-GPU query heads, KV heads, and head size for tensor parallelism."""
    if getattr(config, "kv_lora_rank", None) is not None or getattr(config, "is_encoder_decoder", False):
        raise ValueError("Model input supports causal MHA/GQA, not MLA or encoder-decoder attention")
    q_heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or (1 if getattr(config, "multi_query", False) else q_heads)
    head_size = getattr(config, "head_dim", None)
    if head_size is None:
        if config.hidden_size % q_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads when head_dim is absent")
        head_size = config.hidden_size // q_heads
    if getattr(config, "global_head_dim", head_size) not in (None, head_size):
        raise ValueError("Models with different local/global head dimensions need separate workloads")
    if q_heads % kv_heads or q_heads % tp_size:
        raise ValueError("Query heads must be divisible by KV heads and tensor parallel size")
    if (kv_heads >= tp_size and kv_heads % tp_size) or (kv_heads < tp_size and tp_size % kv_heads):
        raise ValueError("KV heads must divide or be divisible by tensor parallel size")
    q_heads //= tp_size
    kv_heads = max(1, kv_heads // tp_size)  # Replicate KV heads when TP exceeds their count.
    return q_heads, kv_heads, head_size


def model_workloads(args):
    """Generate workloads from model properties and requested sequence batches."""
    config = get_config(model=args.model, trust_remote_code=args.trust_remote_code, revision=args.revision)
    if args.model_prefix:
        config = getattr(config, args.model_prefix)
    config = get_hf_text_config(config)
    q_heads, kv_heads, head_size = model_dimensions(config, args.tp_size)
    dtype = args.dtype
    if dtype == "auto":
        dtype = str(getattr(config, "dtype", None) or getattr(config, "torch_dtype", None) or torch.bfloat16)
        dtype = dtype.removeprefix("torch.")
    dtype = "float8_e4m3fn" if dtype == "fp8" else dtype
    if dtype not in ("bfloat16", "float16", "float8_e4m3fn"):
        raise ValueError("Specify --dtype bfloat16, float16 or fp8 for this model")
    use_td = envs.VLLM_TRITON_USE_TD is not False
    if args.block_size % (32 if dtype == "float8_e4m3fn" else 16):
        raise ValueError("block_size must be divisible by 16, or 32 for FP8")
    windows = model_windows(config, args.sliding_window)
    softcap = getattr(config, "attn_logit_softcapping", None) or 0.0
    softmax_scale = (getattr(config, "query_pre_attn_scalar", None) or head_size)**-0.5
    cases = []
    if args.sequence_batches is not None:
        batches = [tuple(zip(*sequences)) for sequences in args.sequence_batches]
    else:
        batch_sizes = args.batch_size if args.batch_size is not None else [1, 8, 32]
        batches = [([q] * batch, [kv] * batch)
                   for batch, q, kv in product(batch_sizes, args.query_len, args.kv_len)
                   if q <= kv]
    for (query_lens, kv_lens), window in product(batches, windows):
        cases.append({
            "id": f"model-{len(cases):04d}",
            "batch": len(query_lens),
            "query_lens": list(query_lens),
            "kv_lens": list(kv_lens),
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_size": head_size,
            "dtype": dtype,
            "out_dtype": "bfloat16" if dtype == "float8_e4m3fn" else dtype,
            "block_size": args.block_size,
            "use_td": use_td,
            "sliding_window": window,
            "softcap": softcap,
            "softmax_scale": softmax_scale,
            "q_layout": args.q_layout,
            "kv_layout": args.kv_layout,
            "seq_threshold_3d": 32,
            "num_segments": 16,
        })
    if not cases:
        raise ValueError("At least one query length must be <= a KV length")
    return cases


def successful(option):
    return option.get("status") == "ok" and bool(option.get("samples_ms"))


def key_statistics(cases):
    """Score configs measured successfully for every workload sharing a key."""
    # Map config IDs to successful measurements for each workload.
    rows = [{option["id"]: option for option in case["options"] if successful(option)} for case in cases]
    if not rows or any(not row for row in rows):
        return []
    # Keep config IDs measured successfully for every workload.
    common = set.intersection(*(set(row) for row in rows))
    best = [min(statistics.median(option["samples_ms"]) for option in row.values()) for row in rows]
    results = []
    # Score each shared config using the median timings across rounds.
    for identifier in sorted(common):
        times = [statistics.median(row[identifier]["samples_ms"]) for row in rows]
        # Divide each latency by that workload's best measured median.
        ratios = [latency / minimum for latency, minimum in zip(times, best)]
        results.append({
            "id": identifier,
            "config": rows[0][identifier]["config"],
            "max_regret": max(ratios) - 1,
            "geomean_ratio": math.exp(statistics.mean(map(math.log, ratios))),
        })
    return results


def choose_key(cases):
    """Minimize worst slowdown, then the geometric mean ratio within the tolerance."""
    stats = key_statistics(cases)
    if not stats:
        raise ValueError(f"No successful config shared by workloads {[case['id'] for case in cases]}")
    optimum = min(item["max_regret"] for item in stats)
    tied = [item for item in stats if item["max_regret"] <= optimum + SELECTION_TOLERANCE]
    return min(tied, key=lambda item: (item["geomean_ratio"], item["id"]))


def grouped_cases(cases):
    """Group workload results with a recorded key by AttentionKey."""
    grouped = defaultdict(list)
    for case in cases:
        if case.get("key") is not None:
            grouped[runtime.AttentionKey(**case["key"])].append(case)
    return grouped


def save_configs(winners, identity, output):
    """Write winning configs to one JSON file per device and static key."""
    partitions = defaultdict(list)
    for key, winner in winners.items():
        config = runtime.AttentionConfig(**winner)
        partitions[key.static_key()].append((key.dynamic_key(), config))
    profiles = {}
    for static, entries in partitions.items():
        name = runtime.profile_filename(identity, static)
        destination = Path(output) / name
        runtime.write_profile(destination, identity, entries, static_key=static)
        profiles[name] = str(destination)
    return {"profiles": profiles, "accepted_keys": sum(map(len, partitions.values()))}


def allocate_case(case, device, seed):
    """Create synthetic inputs and allocate output and scratch buffers for one workload."""
    torch.manual_seed(seed)
    dtype, out_dtype = getattr(torch, case["dtype"]), getattr(torch, case["out_dtype"])

    def random_tensor(shape):
        storage_dtype = torch.bfloat16 if case["dtype"].startswith("float8") else dtype
        return torch.empty(shape, device=device, dtype=storage_dtype).normal_(std=0.5).to(dtype)

    q_lens, kv_lens = case["query_lens"], case["kv_lens"]
    block, dim, q_heads, kv_heads = (case[name] for name in ("block_size", "head_size", "q_heads", "kv_heads"))
    counts = [(length + block - 1) // block for length in kv_lens]
    num_blocks = case.get("num_blocks", sum(counts))
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    pages = torch.randint(num_blocks, (sum(counts), ), device=device, dtype=torch.int64)
    table = torch.zeros((case["batch"], max(counts)), device=device, dtype=torch.int32)
    offset = 0
    for sequence, count in enumerate(counts):
        table[sequence, :count] = pages[offset:offset + count]
        offset += count
    storage_heads = q_heads + 2 * kv_heads if case["q_layout"] == "qkv" else q_heads
    q_storage = random_tensor((sum(q_lens), storage_heads, dim))
    q = q_storage[:, :q_heads, :]
    if case["kv_layout"] == "interleaved":
        kv_storage = random_tensor((num_blocks, block, kv_heads, 2 * dim))
        k, v = kv_storage.split(dim, dim=-1)
    else:
        k = random_tensor((num_blocks, block, kv_heads, dim))
        v = random_tensor(k.shape)
    out = torch.empty(q.shape, device=device, dtype=out_dtype)
    cumulative = [0]
    for length in q_lens:
        cumulative.append(cumulative[-1] + length)
    segments = case["num_segments"]
    buffers = {}
    # Allocate scratch buffers so decode batches can use the 3D path.
    if max(q_lens) == 1:
        padded = triton.next_power_of_2(dim)
        buffers = {
            "softmax_segm_output": torch.empty((sum(q_lens), q_heads, segments, padded), device=device,
                                               dtype=torch.float32), "softmax_segm_max":
            torch.empty((sum(q_lens), q_heads, segments), device=device, dtype=torch.float32), "softmax_segm_expsum":
            torch.empty((sum(q_lens), q_heads, segments), device=device, dtype=torch.float32)
        }
    kwargs = {
        "q": q, "k": k, "v": v, "out": out, "cu_seqlens_q": torch.tensor(cumulative, device=device, dtype=torch.int32),
        "max_seqlen_q": max(q_lens), "seqused_k": torch.tensor(kv_lens, device=device, dtype=torch.int32),
        "max_seqlen_k": max(kv_lens), "softmax_scale": case.get("softmax_scale", dim**-0.5), "causal": True,
        "window_size": (case["sliding_window"] - 1, 0) if case["sliding_window"] else (-1, -1), "block_table": table,
        "softcap": case["softcap"], "q_descale": None, "k_descale": None, "v_descale": None, "seq_threshold_3D":
        case["seq_threshold_3d"], "num_par_softmax_segments": segments, "use_td": case.get("use_td", True), **buffers
    }
    return kwargs


def synchronize(device):
    getattr(torch, device.type).synchronize(device)


def prepare_option(option, inputs, device):
    """Run a candidate once to trigger compilation and record resource limit failures."""
    config = runtime.AttentionConfig(**option["config"])
    try:
        with runtime.override_config(config):
            unified_attention(**inputs)
            synchronize(device)
        option["status"] = "ok"
    except triton.runtime.autotuner.OutOfResources as error:
        option.update(status="resource_error", error=f"{type(error).__name__}: {error}")
        synchronize(device)


def benchmark_callable(fn, device):
    """Return the callable's average latency in milliseconds using graph replay."""
    backend = getattr(torch, device.type)
    started = time.perf_counter_ns()
    fn()
    backend.synchronize(device)
    eager_ms = max((time.perf_counter_ns() - started) / 1e6, 1e-6)
    slow = eager_ms >= 25
    if slow:
        calls = max(1, min(10, int(REP_MS / eager_ms)))
        warmup_replays = max(1, min(10000, math.ceil(WARMUP_MS / (calls * eager_ms))))
    else:
        for _ in range(4):
            fn()
        backend.synchronize(device)
        eager_ms = max((time.perf_counter_ns() - started) / 5e6, 1e-6)
        calls, warmup_replays = 10, 5
    graph = backend.XPUGraph()
    try:
        with backend.graph(graph):
            for _ in range(calls):
                fn()
        for _ in range(warmup_replays):
            graph.replay()
        backend.synchronize(device)
        start, end = backend.Event(enable_timing=True), backend.Event(enable_timing=True)

        def batch(repeats):
            backend.synchronize(device)
            start.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
            backend.synchronize(device)
            event_ms = start.elapsed_time(end)
            if not math.isfinite(event_ms) or event_ms <= 0:
                raise RuntimeError(f"Invalid graph latency {event_ms}")
            return event_ms

        probe = 1 if slow else max(1, min(20, math.ceil(REP_MS / (calls * eager_ms))))
        estimate = batch(probe) / probe
        repeats = max(1, min(10000, math.ceil(REP_MS / estimate)))

        return batch(repeats) / (repeats * calls)
    finally:
        graph.reset()


def time_options(options, inputs, device, seed):
    """Benchmark candidates in shuffled rounds, storing timings and resource failures."""
    generator = random.Random(seed)
    for _ in range(NUM_ROUNDS):
        order = [option for option in options if option["status"] in ("pending", "ok")]
        generator.shuffle(order)
        for option in order:
            if option["status"] == "pending":
                prepare_option(option, inputs, device)
                if option["status"] != "ok":
                    continue
            config = runtime.AttentionConfig(**option["config"])
            try:
                with runtime.override_config(config):
                    value = benchmark_callable(lambda: unified_attention(**inputs), device)
                option["samples_ms"].append(value)
            except triton.runtime.autotuner.OutOfResources as error:
                option.update(status="resource_error", error=f"{type(error).__name__}: {error}")
                synchronize(device)


@torch.inference_mode()
def tune(args, workloads, device):
    """Measure candidate configs and export one winner per workload key."""
    identity = runtime.get_profile_identity(device)
    rows, winners = [], {}
    # Measure candidates separately for each workload.
    for index, case in enumerate(workloads):
        # Allocate attention inputs for this workload.
        inputs = allocate_case(case, device, args.seed + index)
        key = runtime.make_attention_key(**inputs)
        row = {"id": case["id"], "key": asdict(key), "options": []}
        rows.append(row)
        # Prune invalid configs for this key.
        configs = (config for config in candidate_configs(key)
                   if runtime.validate_config(runtime.AttentionConfig(**config), key, device.type))
        for config in configs:
            option = {"id": config_id(config), "config": config, "status": "pending", "samples_ms": []}
            row["options"].append(option)
        # Time the remaining candidates, resource-limit failures are skipped.
        time_options(
            row["options"],
            inputs,
            device,
            args.seed + index,
        )
        print(f"Measured {index + 1}/{len(workloads)}: {case['id']}", flush=True)
        del inputs
    # Select one config for all workloads sharing a key.
    for key, cases in grouped_cases(rows).items():
        winners[key] = choose_key(cases)["config"]
        print(f"Selected winner for {[case['id'] for case in cases]}: {config_id(winners[key])}", flush=True)
    return save_configs(winners, identity, args.save_dir)


@torch.inference_mode()
def benchmark(args, workloads, device):
    """Measure each workload using a saved config or the runtime fallback."""
    os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(Path(args.save_dir).resolve())
    runtime.clear_profile_cache()
    for index, case in enumerate(workloads):
        # Allocate attention inputs for this workload.
        inputs = allocate_case(case, device, args.seed + index)
        samples = [
            benchmark_callable(lambda inputs=inputs: unified_attention(**inputs), device) for _ in range(NUM_ROUNDS)
        ]
        print(f"{case['id']}: {statistics.median(samples) * 1000:.3f} us", flush=True)
        del inputs


def load_sequence_batches(path):
    """Load and validate batches of [Q, KV] sequence lengths from JSON."""
    batches = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(batches, list) or not batches:
        raise ValueError("Workloads must be a nonempty JSON list of batches")
    for batch in batches:
        if not isinstance(batch, list) or not batch:
            raise ValueError("Each batch must be a nonempty list of [Q, KV] pairs")
        for pair in batch:
            if not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(n, int) or isinstance(n, bool)
                                                                   for n in pair):
                raise ValueError("Each sequence must be an integer [Q, KV] pair")
            if not 0 < pair[0] <= pair[1]:
                raise ValueError("Sequence lengths must satisfy 0 < Q <= KV")
    return batches


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="store_true", help="Search candidates and export winners")
    parser.add_argument("--save-dir", type=str, default="./",
                        help="Profile output folder, or profile input folder when benchmarking")
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        help="Hugging Face model ID or local config directory; weights are not loaded")
    parser.add_argument("--tp-size", "-tp", "--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", help="Model config revision")
    parser.add_argument("--model-prefix", help="Select a nested model configuration before extracting its text config")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "fp8"), default="auto",
                        help="Query/K/V dtype")
    parser.add_argument("--batch-size", type=int, nargs="+", help="Sequence counts (default: 1 8 32)")
    parser.add_argument("--query-len", type=int, nargs="+",
                        help="Query tokens per sequence; 1 selects decode (default: 1)")
    parser.add_argument("--kv-len", type=int, nargs="+",
                        help="KV tokens per sequence, including query tokens (default: 1024)")
    parser.add_argument("--workloads", type=Path,
                        help="Optional JSON batches of [Q, KV] pairs; replaces batch/Q/KV range options")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sliding-window", type=int, nargs="+",
                        help="Override model windows; 0 selects full attention")
    parser.add_argument("--q-layout", choices=("contiguous", "qkv"), default="qkv")
    parser.add_argument("--kv-layout", choices=("contiguous", "interleaved"), default="interleaved")
    args = parser.parse_args(argv)
    args.sequence_batches = None
    if args.workloads is not None and any(value is not None
                                          for value in (args.batch_size, args.query_len, args.kv_len)):
        parser.error("--workloads cannot be combined with --batch-size, --query-len or --kv-len")
    if args.workloads is not None:
        try:
            # Load and validate the requested sequence batches.
            args.sequence_batches = load_sequence_batches(args.workloads)
        except (OSError, ValueError) as error:
            parser.error(f"{args.workloads}: {error}")
    args.query_len = args.query_len if args.query_len is not None else [1]
    args.kv_len = args.kv_len if args.kv_len is not None else [1024]
    if min((args.batch_size or []) + args.query_len + args.kv_len + [args.tp_size, args.block_size]) <= 0:
        parser.error("Batch sizes, lengths, tensor parallel size and block size must be positive")
    if args.sliding_window is not None and min(args.sliding_window) < 0:
        parser.error("Sliding windows must be nonnegative")
    return args


def main(args: argparse.Namespace):
    print(args)
    workloads = model_workloads(args)
    device = torch.device(args.device)
    if device.type != "xpu":
        raise ValueError("XPU is currently the only supported device")
    torch.xpu.set_device(device)
    print(f"{'Tuning' if args.tune else 'Benchmarking'} {len(workloads)} workloads")
    started = time.perf_counter()
    if args.tune:
        result = tune(args, workloads, device)
        print(f"Exported {result['accepted_keys']} keys to {args.save_dir}")
    else:
        benchmark(args, workloads, device)
    print(f"Finished in {time.perf_counter() - started:.1f} seconds")


if __name__ == "__main__":
    main(parse_args())
