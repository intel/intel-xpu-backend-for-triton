# SPDX-License-Identifier: Apache-2.0
r"""Tune XPU attention with graph replay and export device-specific JSON configs.

With unified_attention.patch applied, run from this script's directory::

    python tune_unified_attention.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
      --tp-size 2 --batch-size 1 8 32 --query-len 1 --kv-len 1024 \
      --tune --save-dir configs

The script reads model configuration without loading weights and generates
synthetic attention tensors. Use ``--batch-size``, ``--query-len`` and
``--kv-len`` for workload ranges. Model inputs default to QKV-sliced queries
and interleaved KV; use ``--q-layout contiguous --kv-layout contiguous`` for
separate contiguous tensors. Omit ``--tune`` to benchmark existing configs
from ``--save-dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from itertools import product
from pathlib import Path

import torch
from vllm.transformers_utils.config import get_config, get_hf_text_config
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.ops import triton_unified_attention_config as runtime
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

FORMAT_VERSION = 3


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def candidate_configs(key):
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
    return "fallback" if config is None else canonical(config)


def model_windows(config, override):
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
    if args.block_size % (32 if dtype == "float8_e4m3fn" else 16):
        raise ValueError("TD needs block_size divisible by 16, or 32 for FP8")
    windows = model_windows(config, args.sliding_window)
    softcap = getattr(config, "attn_logit_softcapping", None) or 0.0
    softmax_scale = (getattr(config, "query_pre_attn_scalar", None) or head_size)**-0.5
    cases = []
    batch_sizes = args.batch_size if args.batch_size is not None else [1, 8, 32]
    for batch, query_len, kv_len, window in product(batch_sizes, args.query_len, args.kv_len, windows):
        if query_len > kv_len:
            continue
        cases.append({
            "id": f"model-{len(cases):04d}",
            "batch": batch,
            "query_lens": [query_len] * batch,
            "kv_lens": [kv_len] * batch,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "head_size": head_size,
            "dtype": dtype,
            "out_dtype": "bfloat16" if dtype == "float8_e4m3fn" else dtype,
            "block_size": args.block_size,
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


def successful(option, field="samples_ms"):
    return option.get("status") == "ok" and bool(option.get(field))


def key_statistics(cases, field="samples_ms"):
    """Only compare candidates having a successful cell for every shape."""
    rows = [{option["id"]: option for option in case["options"] if successful(option, field)} for case in cases]
    if not rows or any("fallback" not in row for row in rows):
        return None
    common = set.intersection(*(set(row) for row in rows))
    # Keep initial sweep references even when they are not finalists.
    best = [
        min(
            statistics.median(option[phase])
            for option in case["options"]
            for phase in (field, "samples_ms")
            if successful(option, phase))
        for case in cases
    ]
    results = []
    for identifier in sorted(common):
        times = [statistics.median(row[identifier][field]) for row in rows]
        ratios = [latency / minimum for latency, minimum in zip(times, best)]
        results.append({
            "id": identifier,
            "config": rows[0][identifier]["config"],
            "max_regret": max(ratios) - 1,
            "geomean_ratio": math.exp(statistics.mean(map(math.log, ratios))),
        })
    return results


def choose_key(cases, *, tolerance=0.01, field="confirmation_ms"):
    stats = key_statistics(cases, field)
    if stats is None:
        raise ValueError(f"Missing or failed {field} fallback measurement for {[case['id'] for case in cases]}")
    optimum = min(item["max_regret"] for item in stats)
    tied = [item for item in stats if item["max_regret"] <= optimum + tolerance]
    return min(tied, key=lambda item: (item["geomean_ratio"], item["id"]))


def grouped_cases(cases):
    grouped = defaultdict(list)
    for case in cases:
        if case.get("key") is not None:
            grouped[runtime.AttentionKey(**case["key"])].append(case)
    return grouped


def finalists(cases, count=3):
    stats = key_statistics(cases)
    if stats is None:
        return ["fallback"]
    ranked = sorted(
        (item for item in stats if item["id"] != "fallback"),
        key=lambda item: (item["max_regret"], item["geomean_ratio"], item["id"]),
    )
    selected = ["fallback"] + [item["id"] for item in ranked[:count]]
    initial = choose_key(cases, field="samples_ms")
    selected.append(initial["id"])
    for case in cases:
        measured = [option for option in case["options"] if successful(option)]
        if measured:
            selected.append(min(measured, key=lambda option: statistics.median(option["samples_ms"]))["id"])
    return list(dict.fromkeys(selected))


def save_configs(data, output, *, tolerance=0.01):
    identity = data["identity"]
    partitions = defaultdict(list)
    for key, cases in grouped_cases(data["cases"]).items():
        winner = choose_key(cases, tolerance=tolerance)["config"]
        print(f"Winner for {[case['id'] for case in cases]}: {config_id(winner)}", flush=True)
        config = runtime.AttentionConfig(**winner) if winner is not None else None
        partitions[key.static_key()].append((key.dynamic_key(), config))
    profiles = {}
    for static, entries in partitions.items():
        name = runtime.profile_filename(identity, static)
        destination = Path(output) / name
        runtime.write_profile(destination, identity, entries, static_key=static)
        profiles[name] = str(destination)
    return {"profiles": profiles, "accepted_keys": sum(map(len, partitions.values()))}


def allocate_case(case, device, seed):
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
    if max(q_lens) == 1:
        padded = 1 << (dim - 1).bit_length()
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
        case["seq_threshold_3d"], "num_par_softmax_segments": segments, "use_td": True, **buffers
    }
    return kwargs


def synchronize(device):
    getattr(torch, device.type).synchronize(device)


def prepare_option(option, inputs, device):
    config = runtime.AttentionConfig(**option["config"]) if option["config"] is not None else None
    try:
        with runtime.override_config(config):
            unified_attention(**inputs)
            synchronize(device)
            option["key"] = asdict(runtime.last_resolution()[0])
        option["status"] = "ok"
    except triton.runtime.autotuner.OutOfResources as error:
        option.update(status="resource_error", error=f"{type(error).__name__}: {error}")
        synchronize(device)


@contextmanager
def graph_timer(fn, device, warmup_ms, rep_ms):
    """Use shorter graph batches for slow calls; time with boundary events."""
    backend = getattr(torch, device.type)
    started = time.perf_counter_ns()
    fn()
    backend.synchronize(device)
    eager_ms = max((time.perf_counter_ns() - started) / 1e6, 1e-6)
    slow = eager_ms >= 25
    if slow:
        calls = max(1, min(10, int(rep_ms / eager_ms)))
        warmup_replays = max(1, min(10000, math.ceil(warmup_ms / (calls * eager_ms))))
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
        start.record()
        end.record()
        backend.synchronize(device)

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

        probe = 1 if slow else max(1, min(20, math.ceil(rep_ms / (calls * eager_ms))))
        estimate = batch(probe) / probe
        if not slow:
            warmup_replays = max(1, min(10000, math.ceil(warmup_ms / estimate)))
            for _ in range(warmup_replays):
                graph.replay()
            backend.synchronize(device)
        repeats = max(1, min(10000, math.ceil(rep_ms / estimate)))

        def elapsed():
            return batch(repeats) / (repeats * calls)

        yield graph, elapsed
    finally:
        graph.reset()


def benchmark_callable(fn, device, warmup_ms, rep_ms, *, timer=graph_timer):
    """Time complete attention calls using graph replay and device events."""
    with timer(fn, device, warmup_ms, rep_ms) as (_, elapsed):
        return elapsed()


def time_options(options, inputs, device, rounds, warmup_ms, rep_ms, seed, field, *, timer=graph_timer):
    generator = random.Random(seed)
    for _ in range(rounds):
        order = [option for option in options if option["status"] == "ok"]
        generator.shuffle(order)
        for option in order:
            config = runtime.AttentionConfig(**option["config"]) if option["config"] is not None else None
            try:
                with runtime.override_config(config):
                    value = benchmark_callable(lambda: unified_attention(**inputs), device, warmup_ms, rep_ms,
                                               timer=timer)
                option.setdefault(field, []).append(value)
            except triton.runtime.autotuner.OutOfResources as error:
                option.update(status="resource_error", error=f"{type(error).__name__}: {error}")
                synchronize(device)


def tune(args, manifest, device, *, allocate=allocate_case, timer=graph_timer,
         timing_scope="attention_sequence_graph_device"):
    identity = runtime.get_profile_identity(device)
    data = {
        "format_version": FORMAT_VERSION,
        "complete": False,
        "identity": identity,
        "manifest": manifest,
        "settings": vars(args),
        "cases": [],
        "timing_scope": timing_scope,
        "started_at": time.time(),
    }
    checkpoint(args.measurements, data)
    try:
        with torch.inference_mode():
            for index, case in enumerate(manifest):
                inputs = allocate(case, device, args.seed + index)
                fallback = {"id": "fallback", "config": None, "status": "pending", "samples_ms": []}
                row = {"id": case["id"], "workload": case, "key": None, "options": [fallback]}
                data["cases"].append(row)
                prepare_option(fallback, inputs, device)
                if fallback["status"] != "ok":
                    raise RuntimeError(f"Fallback failed for {case['id']}: {fallback.get('error')}")
                row["key"] = fallback["key"]
                key = runtime.AttentionKey(**row["key"])  # pylint: disable=not-a-mapping
                configs = (config for config in candidate_configs(key)
                           if runtime.validate_config(runtime.AttentionConfig(**config), key, device.type))
                for config_index, config in enumerate(configs):
                    option = {"id": config_id(config), "config": config, "status": "pending", "samples_ms": []}
                    row["options"].append(option)
                    prepare_option(option, inputs, device)
                    if config_index % 32 == 31:
                        checkpoint(args.measurements, data)
                        print(f"Prepared {config_index + 1} candidates for {case['id']}", flush=True)
                time_options(
                    row["options"],
                    inputs,
                    device,
                    args.rounds,
                    args.warmup_ms,
                    args.rep_ms,
                    args.seed + index,
                    "samples_ms",
                    timer=timer,
                )
                checkpoint(args.measurements, data)
                print(f"Measured {index + 1}/{len(manifest)}: {case['id']}", flush=True)
                del inputs
            for group_index, cases in enumerate(grouped_cases(data["cases"]).values()):
                selected = finalists(cases, args.finalists)
                for row in cases:
                    index = next(i for i, case in enumerate(manifest) if case["id"] == row["id"])
                    inputs = allocate(row["workload"], device, args.seed + index)
                    options = [option for option in row["options"] if option["id"] in selected]
                    for option in options:
                        prepare_option(option, inputs, device)
                    time_options(
                        options,
                        inputs,
                        device,
                        args.confirmation_rounds,
                        args.warmup_ms,
                        args.rep_ms,
                        args.seed + 10000 + index,
                        "confirmation_ms",
                        timer=timer,
                    )
                    del inputs
                checkpoint(args.measurements, data)
                print(f"Confirmed key {group_index + 1}", flush=True)
        data["complete"] = True
    finally:
        data["finished_at"] = time.time()
        checkpoint(args.measurements, data)

    return save_configs(data, args.save_dir, tolerance=args.tolerance)


def checkpoint(path, data):
    if path:
        atomic_json(path, data)


@torch.inference_mode()
def benchmark(args, manifest, device, *, allocate=allocate_case, timer=graph_timer):
    os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(Path(args.save_dir).resolve())
    runtime.clear_profile_cache()
    for index, case in enumerate(manifest):
        inputs = allocate(case, device, args.seed + index)
        samples = [
            benchmark_callable(lambda inputs=inputs: unified_attention(**inputs), device, args.warmup_ms, args.rep_ms,
                               timer=timer) for _ in range(args.rounds)
        ]
        print(f"{case['id']}: {statistics.median(samples) * 1000:.3f} us", flush=True)
        del inputs


def create_parser(description=__doc__):
    parser = FlexibleArgumentParser(description=description)
    parser.add_argument("--tune", action="store_true", help="Search candidates and export winners")
    parser.add_argument("--save-dir", type=str, default="./",
                        help="Profile output folder, or profile input folder when benchmarking")
    parser.add_argument("--measurements", help="Optional detailed tuning checkpoint JSON")
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--confirmation-rounds", type=int, default=5)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--warmup-ms", type=float, default=10)
    parser.add_argument("--rep-ms", type=float, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Worst-regret tie tolerance before geometric-mean tie-breaking")
    return parser


def parse_args(argv=None):
    parser = create_parser()
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        help="Hugging Face model ID or local config directory; weights are not loaded")
    parser.add_argument("--tp-size", "-tp", "--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", help="Model config revision")
    parser.add_argument("--model-prefix", help="Select a nested model configuration before extracting its text config")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "fp8"), default="auto",
                        help="Query/K/V dtype")
    parser.add_argument("--batch-size", type=int, nargs="+", help="Sequence counts (default: 1 8 32)")
    parser.add_argument("--query-len", type=int, nargs="+", default=[1],
                        help="Query tokens per sequence; 1 selects decode")
    parser.add_argument("--kv-len", type=int, nargs="+", default=[1024],
                        help="KV tokens per sequence, including query tokens")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sliding-window", type=int, nargs="+",
                        help="Override model windows; 0 selects full attention")
    parser.add_argument("--q-layout", choices=("contiguous", "qkv"), default="qkv")
    parser.add_argument("--kv-layout", choices=("contiguous", "interleaved"), default="interleaved")
    args = parser.parse_args(argv)
    if min((args.batch_size or []) + args.query_len + args.kv_len + [args.tp_size, args.block_size]) <= 0:
        parser.error("Batch sizes, lengths, tensor parallel size and block size must be positive")
    if args.sliding_window is not None and min(args.sliding_window) < 0:
        parser.error("Sliding windows must be nonnegative")
    return args


def main(args: argparse.Namespace, workloads=None, *, allocate=allocate_case, timer=graph_timer,
         timing_scope="attention_sequence_graph_device"):
    print(args)
    for name in ("rounds", "warmup_ms", "rep_ms", "confirmation_rounds", "finalists"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        raise ValueError("--tolerance must be finite and nonnegative")
    if args.measurements and not args.tune:
        raise ValueError("--measurements requires --tune")
    if workloads is None:
        workloads = model_workloads(args)
    device = torch.device(args.device)
    if device.type != "xpu":
        raise ValueError("The candidate pool and profiles target XPU")
    torch.xpu.set_device(device)
    print(f"{'Tuning' if args.tune else 'Benchmarking'} {len(workloads)} workloads")
    started = time.perf_counter()
    if args.tune:
        result = tune(args, workloads, device, allocate=allocate, timer=timer, timing_scope=timing_scope)
        print(f"Exported {result['accepted_keys']} keys to {args.save_dir}")
    else:
        benchmark(args, workloads, device, allocate=allocate, timer=timer)
    print(f"Finished in {time.perf_counter() - started:.1f} seconds")


if __name__ == "__main__":
    main(parse_args())
