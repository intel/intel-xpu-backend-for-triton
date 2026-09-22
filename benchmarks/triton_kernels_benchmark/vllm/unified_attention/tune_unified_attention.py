# SPDX-License-Identifier: Apache-2.0
"""Measure production attention offline and export conservative workload profiles."""

from __future__ import annotations

# Exact integer checks reject bools in workload manifests.
# pylint: disable=unidiomatic-typecheck

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path

FORMAT_VERSION = 2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


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


def candidate_configs():
    return [{"block_m": m, "tile_size": t, "num_warps": w, "num_stages": s, "grf_mode": g}
            for m in (16, 32, 64)
            for t in (16, 32, 64, 128)
            for w in (2, 4, 8, 16)
            for s in (1, 2, 3)
            for g in ("default", "256")]


def config_id(config):
    return "fallback" if config is None else canonical(config)


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
            if type(case[name]) is not int or case[name] <= 0:
                raise ValueError(f"{case['id']}: {name} must be a positive integer")
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


def selection_features(workload):
    """Describe active host-side lengths for offline selection experiments."""
    q, kv = workload["query_lens"], workload["kv_lens"]
    if not q or len(q) != len(kv) or any(type(v) is not int or v <= 0 for v in (*q, *kv)):
        raise ValueError("Selection features require nonempty, positive active lengths")
    n, total_q, total_kv = len(q), sum(q), sum(kv)
    average_kv = total_kv / n
    return {
        "is_decode": all(length == 1 for length in q),
        "num_seqs": n,
        "query_work": total_q * workload["q_heads"],
        "total_q_tokens": total_q,
        "max_q_len": max(q),
        "max_kv_len": max(kv),
        "total_kv_tokens": total_kv,
        "avg_kv_len": average_kv,
        "q_raggedness": 1 - total_q / (n * max(q)),
        "kv_raggedness": 1 - total_kv / (n * max(kv)),
        "kv_cv": math.sqrt(sum((length - average_kv)**2 for length in kv) / n) / average_kv,
    }


def successful(option, field="samples_ms"):
    values = option.get(field, [])
    return (option.get("status") == "ok" and bool(values)
            and all(isinstance(value, (float, int)) and math.isfinite(value) and value > 0 for value in values))


def key_statistics(cases, field="samples_ms"):
    """Only compare candidates having a successful cell for every shape."""
    rows = [{option["id"]: option for option in case["options"] if successful(option, field)} for case in cases]
    if any("fallback" not in row for row in rows):
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
    fallback = [statistics.median(row["fallback"][field]) for row in rows]
    results = []
    for identifier in sorted(common):
        times = [statistics.median(row[identifier][field]) for row in rows]
        ratios = [latency / minimum for latency, minimum in zip(times, best)]
        fallback_ratios = [latency / baseline for latency, baseline in zip(times, fallback)]
        regressions = [ratio - 1 for ratio in fallback_ratios]
        results.append({
            "id": identifier,
            "config": rows[0][identifier]["config"],
            "max_regret": max(ratios) - 1,
            "geomean_ratio": math.exp(statistics.mean(map(math.log, ratios))),
            "geomean_fallback_ratio": math.exp(statistics.mean(map(math.log, fallback_ratios))),
            "max_fallback_regression": max(regressions),
            "ratios": ratios,
            "fallback_regressions": regressions,
        })
    return results


def choose_key(cases, *, tolerance=0.01, field="confirmation_ms"):
    stats = key_statistics(cases, field)
    if stats is None:
        return {"accepted": False, "reason": "missing or failed fallback measurement"}
    optimum = min(item["max_regret"] for item in stats)
    tied = [item for item in stats if item["max_regret"] <= optimum + tolerance]
    winner = min(tied, key=lambda item: (item["geomean_ratio"], item["id"]))
    return {
        "accepted": True,
        "winner": winner,
        "fallback": next(item for item in stats if item["id"] == "fallback"),
        "candidates": [item for item in stats if item["id"] != "fallback"],
    }


def grouped_cases(cases):
    grouped = defaultdict(list)
    for case in cases:
        if case.get("key") is not None:
            grouped[canonical(case["key"])].append(case)
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
    if initial["accepted"]:
        selected.append(initial["winner"]["id"])
    for case in cases:
        measured = [option for option in case["options"] if successful(option)]
        if measured:
            selected.append(min(measured, key=lambda option: statistics.median(option["samples_ms"]))["id"])
    return list(dict.fromkeys(selected))


def load_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime_module(path=None):
    if path:
        return load_by_path(path, "_ua_offline_profile_contract")
    return importlib.import_module("vllm.v1.attention.ops.triton_unified_attention_config")


# pylint: disable-next=too-many-branches
def select_results(data, runtime, output, *, tolerance=0.01):
    if data.get("format_version") != FORMAT_VERSION or data.get("mode") != "measure" or not data.get("complete"):
        raise ValueError("Selection requires a complete, supported measurement artifact")
    if data.get("candidate_set_hash") != digest(candidate_configs()):
        raise ValueError("Candidate-set identity differs from this tuner")
    if data.get("manifest_hash") != digest(data.get("manifest")):
        raise ValueError("Manifest hash mismatch")
    expected = {case["id"] for case in data["manifest"]}
    if {case["id"] for case in data["cases"]} != expected or len(data["cases"]) != len(expected):
        raise ValueError("The complete artifact must contain every manifest workload exactly once")
    identity = data.get("identity")
    if not identity or identity.get("execution_mode") != "eager":
        raise ValueError("A recorded eager device/compiler identity is required")
    for field, constant in (
        ("schema_version", "SCHEMA_VERSION"),
        ("key_version", "KEY_VERSION"),
        ("kernel_contract", "KERNEL_CONTRACT"),
    ):
        if identity.get(field) != getattr(runtime, constant):
            raise ValueError(f"Measurement {field} differs from the runtime contract")
    runtime_hash = hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()
    if data.get("runtime_config_sha256") != runtime_hash or identity["provenance"]["resolver_sha256"] != runtime_hash:
        raise ValueError("Measurements require the same runtime resolver source")
    accepted, partitions, report = defaultdict(list), {}, []
    manifest = {case["id"]: case for case in data["manifest"]}
    for case in data["cases"]:
        if case.get("key") is None:
            continue
        workload = manifest[case["id"]]
        key = runtime.AttentionKey(**case["key"])
        if key.q_dtype != workload["dtype"]:
            raise ValueError("Recorded query dtype differs from the manifest")
        if key.query_work != sum(workload["query_lens"]) * workload["q_heads"]:
            raise ValueError("Recorded query work differs from the manifest")
        if key.sliding_window != max(0, workload["sliding_window"]):
            raise ValueError("Recorded sliding window differs from the manifest")
        static = key.static_key()
        partitions[runtime.profile_filename(identity, static)] = static
    for serialized_key, cases in grouped_cases(data["cases"]).items():
        key = runtime.AttentionKey(**json.loads(serialized_key))
        partition = runtime.profile_filename(identity, key.static_key())
        result = choose_key(cases, tolerance=tolerance)
        result.update(key=asdict(key.dynamic_key()), partition=partition, case_ids=[case["id"] for case in cases])
        if result["accepted"]:
            winner = result["winner"]["config"]
            config = runtime.AttentionConfig(**winner) if winner is not None else None
            if config is not None and not runtime.validate_config(config, key, identity["target"]["backend"]):
                raise ValueError("Selected config is incompatible with the runtime contract")
            accepted[partition].append((key.dynamic_key(), config))
        report.append(result)
    provenance = {
        "measurement_sha256": digest(data),
        "manifest_sha256": data["manifest_hash"],
        "candidate_set_sha256": data["candidate_set_hash"],
        "runtime_config_sha256": data["runtime_config_sha256"],
        "tuner_sha256": data.get("tuner_sha256"),
        "selection": {"objective": "minimax_regret", "tie_tolerance": tolerance},
        "timing_scope": "production_wrapper_eager",
        "development_only": True,
    }
    selection = {
        "profiles": {},
        "selection_reports": {},
        "accepted_keys": sum(map(len, accepted.values())),
        "fallback_keys": sum(row["accepted"] and row["winner"]["id"] == "fallback" for row in report),
        "total_keys": len(report),
        "keys": report,
        "failed_cases": [case for case in data["cases"] if case.get("key") is None],
    }
    for partition, static in sorted(partitions.items()):
        destination = Path(output) / partition
        runtime.write_profile(destination, identity, accepted[partition], static_key=static, provenance=provenance)
        report_path = destination.with_name(destination.name.replace("unified_attention_", "selection-report_", 1))
        partition_report = [row for row in report if row["partition"] == partition]
        atomic_json(
            report_path, {
                "profile": str(destination),
                "identity": identity,
                "static_key": asdict(static),
                "measurement_sha256": provenance["measurement_sha256"],
                "accepted_keys": len(accepted[partition]),
                "fallback_keys": sum(row["accepted"] and row["winner"]["id"] == "fallback" for row in partition_report),
                "total_keys": len(partition_report),
                "keys": partition_report,
            })
        selection["profiles"][partition] = str(destination)
        selection["selection_reports"][partition] = str(report_path)

    return selection


def torch_environment():  # pylint: disable=import-outside-toplevel
    # Keep selection and CLI parsing independent of GPU packages.
    # pylint: disable=import-outside-toplevel
    import torch
    import triton
    import triton.testing

    runtime = runtime_module()
    attention = importlib.import_module("vllm.v1.attention.ops.triton_unified_attention")
    return torch, triton, runtime, attention


def allocate_case(case, torch, device, seed):
    torch.manual_seed(seed)
    dtype, out_dtype = getattr(torch, case["dtype"]), getattr(torch, case["out_dtype"])
    q_lens, kv_lens = case["query_lens"], case["kv_lens"]
    block, dim, q_heads, kv_heads = (case[name] for name in ("block_size", "head_size", "q_heads", "kv_heads"))
    counts = [(length + block - 1) // block for length in kv_lens]
    pages = torch.randperm(sum(counts), device=device, dtype=torch.int64)
    table = torch.zeros((case["batch"], max(counts)), device=device, dtype=torch.int32)
    offset = 0
    for sequence, count in enumerate(counts):
        table[sequence, :count] = pages[offset:offset + count]
        offset += count
    storage_heads = q_heads + 2 * kv_heads if case["q_layout"] == "qkv" else q_heads
    q_storage = (torch.randn((sum(q_lens), storage_heads, dim), device=device, dtype=torch.float32) * 0.5).to(dtype)
    q = q_storage[:, :q_heads, :]
    if case["kv_layout"] == "interleaved":
        kv_storage = (torch.randn(
            (sum(counts), block, kv_heads, 2 * dim), device=device, dtype=torch.float32) * 0.5).to(dtype)
        k, v = kv_storage.split(dim, dim=-1)
    else:
        k = (torch.randn((sum(counts), block, kv_heads, dim), device=device, dtype=torch.float32) * 0.5).to(dtype)
        v = (torch.randn(k.shape, device=device, dtype=torch.float32) * 0.5).to(dtype)
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
        "max_seqlen_k": max(kv_lens), "softmax_scale": dim**-0.5, "causal": True, "window_size":
        (case["sliding_window"] - 1, 0) if case["sliding_window"] else (-1, -1), "block_table": table, "softcap":
        case["softcap"], "q_descale": None, "k_descale": None, "v_descale": None, "seq_threshold_3D":
        case["seq_threshold_3d"], "num_par_softmax_segments": segments, "use_td": True, **buffers
    }
    reference = torch.empty_like(out)
    ratio = q_heads // kv_heads
    offset = 0
    for sequence, (q_len, kv_len, count) in enumerate(zip(q_lens, kv_lens, counts)):
        keys = k[table[sequence, :count].long()].reshape(-1, kv_heads, dim)[:kv_len].float()
        values = v[table[sequence, :count].long()].reshape(-1, kv_heads, dim)[:kv_len].float()
        positions = torch.arange(kv_len, device=device)
        for head in range(kv_heads):
            for start in range(0, q_len, 64):
                end = min(q_len, start + 64)
                query = q[offset + start:offset + end, head * ratio:(head + 1) * ratio].float().transpose(0, 1)
                scores = torch.matmul(query, keys[:, head].T) * dim**-0.5
                if case["softcap"]:
                    scores = case["softcap"] * torch.tanh(scores / case["softcap"])
                query_positions = kv_len - q_len + torch.arange(start, end, device=device)
                mask = positions[None, :] <= query_positions[:, None]
                if case["sliding_window"]:
                    mask &= positions[None, :] > query_positions[:, None] - case["sliding_window"]
                probabilities = torch.softmax(scores.masked_fill(~mask[None], float("-inf")), dim=-1)
                result = torch.matmul(probabilities, values[:, head]).transpose(0, 1)
                reference[offset + start:offset + end, head * ratio:(head + 1) * ratio] = result
        offset += q_len
    return kwargs, reference


def jit_kernel(module):
    kernel = module.kernel_unified_attention
    while hasattr(kernel, "configs") or hasattr(kernel, "values"):
        kernel = kernel.fn
    return kernel


@contextmanager
def observe_launch(module):
    kernel = jit_kernel(module)
    original = kernel.run
    launches = []

    def observed(*args, **kwargs):
        named = dict(zip(kernel.arg_names, args)) | kwargs
        compiled = original(*args, **kwargs)
        metadata = compiled.metadata
        launches.append({
            "config": {
                "block_m": named["BLOCK_M"], "tile_size": named["TILE_SIZE"], "num_warps": metadata.num_warps,
                "num_stages": metadata.num_stages, "grf_mode": getattr(metadata, "grf_mode", "default")
            },
            "block_q": named.get("BLOCK_Q"),
            "is_3d": named["IS_3D"],
            "use_td": named["USE_TD"],
            "use_td_qo": named.get("USE_TD_QO", False),
            "kernel_hash": compiled.hash,
        })
        return compiled

    kernel.run = observed
    try:
        yield launches
    finally:
        kernel.run = original


def check_output(torch, case, actual, reference):
    if case["dtype"].startswith("float8_"):
        atol, rtol = 0.025, 0.10
    elif case["out_dtype"] == "float16":
        atol, rtol = 0.003, 0.002
    else:
        atol, rtol = 0.025, 0.01
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    error = actual.float() - reference.float()
    relative_rmse = float((error.square().mean() / reference.float().square().mean().clamp_min(1e-20)).sqrt().item())
    if not math.isfinite(relative_rmse) or relative_rmse > 0.10:
        raise AssertionError(f"Relative RMS error {relative_rmse} exceeds 0.10")
    return {
        "atol": atol,
        "rtol": rtol,
        "relative_rmse": relative_rmse,
        "max_abs_error": float(error.abs().max().item()),
    }


def synchronize(torch, device):
    getattr(torch, device.type).synchronize(device)


def verify_launched_config(actual, expected):
    if actual != expected:
        raise RuntimeError(f"Selected config differs from actual launch: expected {expected}, got {actual}")


def prepare_option(option, case, inputs, reference, env, device):
    torch, _, runtime, module = env
    config = runtime.AttentionConfig(**option["config"]) if option["config"] is not None else None
    phase = "launch"
    inputs["out"].fill_(float("nan"))
    try:
        with runtime.override_config(config), observe_launch(module) as launches:
            module.unified_attention(**inputs)
            synchronize(torch, device)
            resolved = runtime.last_resolution()
        if not launches or resolved is None:
            raise RuntimeError("Wrapper did not expose its resolved key and actual launch")
        option["actual_launch"] = launches[-1]
        option["key"] = asdict(resolved[0])
        if not launches[-1]["use_td"]:
            raise RuntimeError("Requested TD benchmark launched pointer attention")
        if config is not None:
            verify_launched_config(launches[-1]["config"], asdict(config))
        phase = "correctness"
        option["accuracy"] = check_output(torch, case, inputs["out"], reference)
        option["status"] = "ok"
    except Exception as error:  # pylint: disable=broad-exception-caught
        option.update(status=f"{phase}_error", error=f"{type(error).__name__}: {error}")
        synchronize(torch, device)


def time_options(options, _case, inputs, env, device, rounds, warmup_ms, rep_ms, seed, field):
    torch, triton, runtime, module = env
    generator = random.Random(seed)
    for _ in range(rounds):
        order = [option for option in options if option["status"] == "ok"]
        generator.shuffle(order)
        for option in order:
            config = runtime.AttentionConfig(**option["config"]) if option["config"] is not None else None
            try:
                with runtime.override_config(config):
                    value = triton.testing.do_bench(lambda: module.unified_attention(**inputs), warmup=warmup_ms,
                                                    rep=rep_ms, return_mode="median")
                if not math.isfinite(value) or value <= 0:
                    raise RuntimeError(f"Invalid latency {value}")
                option.setdefault(field, []).append(value)
            except Exception as error:  # pylint: disable=broad-exception-caught
                option.update(status="timing_error", error=f"{type(error).__name__}: {error}")
                synchronize(torch, device)


def measure(args):
    env = torch_environment()
    torch, _, runtime, module = env
    device = torch.device(args.device)
    if device.type != "xpu":
        raise ValueError("The initial candidate pool and profiles target XPU")
    identity = runtime.get_profile_identity(device)
    if identity is None:
        raise ValueError("Cannot identify the device/compiler/source; refusing unversioned measurements")
    manifest = normalize_workloads(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    data = {
        "format_version": FORMAT_VERSION, "mode": "measure", "complete": False, "identity": identity, "manifest":
        manifest, "manifest_hash": digest(manifest), "candidate_set_hash": digest(candidate_configs()), "settings":
        vars(args), "timing_scope": "production_wrapper_eager", "cases": [], "started_at": time.time(), "torch_version":
        torch.__version__, "imported_source": module.__file__
    }
    data["settings"] = {key: value for key, value in vars(args).items() if key != "func"}
    data["tuner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data["runtime_config_sha256"] = hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()
    data["reference"] = "PyTorch float32 on actual quantized inputs; causal/window/softcap"
    getattr(torch, device.type).reset_peak_memory_stats(device)
    atomic_json(args.output, data)
    try:
        with torch.inference_mode():
            for index, case in enumerate(manifest):
                inputs, reference = allocate_case(case, torch, device, args.seed + index)
                fallback = {"id": "fallback", "config": None, "status": "pending", "samples_ms": []}
                row = {
                    "id": case["id"], "workload": case, "selection_features": selection_features(case), "key": None,
                    "options": [fallback]
                }
                data["cases"].append(row)
                prepare_option(fallback, case, inputs, reference, env, device)
                row["key"] = fallback.get("key")
                if row["key"] is not None:
                    key = runtime.AttentionKey(**row["key"])  # pylint: disable=not-a-mapping
                    for config in candidate_configs():
                        option = {"id": config_id(config), "config": config, "status": "pending", "samples_ms": []}
                        row["options"].append(option)
                        if not runtime.validate_config(runtime.AttentionConfig(**config), key, device.type):
                            option["status"] = "structurally_invalid"
                        else:
                            prepare_option(option, case, inputs, reference, env, device)
                    time_options(
                        row["options"],
                        case,
                        inputs,
                        env,
                        device,
                        args.rounds,
                        args.warmup_ms,
                        args.rep_ms,
                        args.seed + index,
                        "samples_ms",
                    )
                atomic_json(args.output, data)
                print(f"Measured {index + 1}/{len(manifest)}: {case['id']}", flush=True)
                del inputs, reference
            for group_index, cases in enumerate(grouped_cases(data["cases"]).values()):
                selected = finalists(cases, args.finalists)
                for row in cases:
                    index = next(i for i, case in enumerate(manifest) if case["id"] == row["id"])
                    inputs, reference = allocate_case(row["workload"], torch, device, args.seed + index)
                    options = [option for option in row["options"] if option["id"] in selected]
                    for option in options:
                        prepare_option(option, row["workload"], inputs, reference, env, device)
                    time_options(
                        options,
                        row["workload"],
                        inputs,
                        env,
                        device,
                        args.confirmation_rounds,
                        args.warmup_ms,
                        args.rep_ms,
                        args.seed + 10000 + index,
                        "confirmation_ms",
                    )
                    del inputs, reference
                atomic_json(args.output, data)
                print(f"Confirmed key {group_index + 1}", flush=True)
        data["complete"] = True
    except BaseException as error:
        data["session_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        data["finished_at"] = time.time()
        data["peak_allocated_bytes"] = getattr(torch, device.type).max_memory_allocated(device)
        atomic_json(args.output, data)


def summarize_ratios(values):
    ordered = sorted(values)
    if not values:
        return None
    return {
        "count": len(values), "geomean": math.exp(statistics.mean(map(math.log, values))), "median":
        statistics.median(values), "p95": ordered[math.ceil(0.95 * len(ordered)) - 1], "maximum": max(values),
        "regressions_gt_5pct": sum(value > 1.05 for value in values), "regressions_gt_10pct":
        sum(value > 1.10 for value in values), "regressions_gt_20pct": sum(value > 1.20 for value in values)
    }


# pylint: disable-next=too-many-branches
def compare(args):
    if args.profile_folder:
        os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(Path(args.profile_folder).resolve())
    env = torch_environment()
    torch, triton, runtime, module = env
    device = torch.device(args.device)
    manifest = normalize_workloads(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    baseline = load_by_path(args.autotune_source, "_ua_cached_autotune_reference") if args.autotune_source else None
    if baseline is not None and not hasattr(baseline.kernel_unified_attention, "cache"):
        raise ValueError("--autotune-source must provide the ua-FINAL autotuned wrapper")
    data = {
        "format_version": FORMAT_VERSION, "mode": "compare", "complete": False, "identity":
        runtime.get_profile_identity(device), "manifest": manifest, "manifest_hash": digest(manifest), "timing_scope":
        "production_wrapper_eager", "baseline_policy": "normal cached autotuning", "autotune_source_sha256":
        hashlib.sha256(Path(args.autotune_source).read_bytes()).hexdigest() if baseline else None, "settings":
        {key: value
         for key, value in vars(args).items()
         if key != "func"}, "cases": []
    }
    order = list(range(len(manifest)))
    if args.warmup_order == "reverse":
        order.reverse()
    elif args.warmup_order == "shuffle":
        random.Random(args.seed).shuffle(order)
    data["warmup_case_ids"] = [manifest[index]["id"] for index in order]
    data["tuner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data["runtime_config_sha256"] = hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()
    if args.profile_folder:
        data["profile_files_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(args.profile_folder).glob("*.json"))
        }
    atomic_json(args.output, data)
    arms = ["fallback", "profile"] + (["cached_autotune"] if baseline else [])

    def context(arm):
        if arm == "fallback":
            return runtime.override_config(None)
        return runtime.offline_mode() if arm == "profile" else nullcontext()

    def attention(arm):
        return baseline if arm == "cached_autotune" else module

    try:
        with torch.inference_mode():
            warmup_started = time.perf_counter()
            for index in order:
                inputs, reference = allocate_case(manifest[index], torch, device, args.seed + index)
                for arm in arms:
                    with context(arm):
                        attention(arm).unified_attention(**inputs)
                    synchronize(torch, device)
                    check_output(torch, manifest[index], inputs["out"], reference)
                del inputs, reference
            data["warmup_seconds"] = time.perf_counter() - warmup_started
            for index, case in enumerate(manifest):
                inputs, reference = allocate_case(case, torch, device, args.seed + index)
                row = {"id": case["id"], "workload": case, "selection_features": selection_features(case), "arms": {}}
                data["cases"].append(row)
                for arm in arms:
                    with context(arm), observe_launch(attention(arm)) as launches:
                        attention(arm).unified_attention(**inputs)
                        synchronize(torch, device)
                        resolution = runtime.last_resolution() if arm != "cached_autotune" else None
                        match = runtime.last_profile_match() if arm == "profile" else None
                    if not launches or not launches[-1]["use_td"]:
                        raise RuntimeError(f"{arm} did not launch TD attention")
                    if arm != "cached_autotune" and resolution is None:
                        raise RuntimeError(f"{arm} did not expose the runtime key")
                    if resolution is not None and resolution[1] is not None:
                        verify_launched_config(launches[-1]["config"], asdict(resolution[1]))
                    row["arms"][arm] = {
                        "actual_launch": launches[-1], "samples_ms": [], "accuracy":
                        check_output(torch, case, inputs["out"], reference)
                    }
                    if resolution is not None:
                        row["arms"][arm]["key"] = asdict(resolution[0])
                        row["arms"][arm]["profile_hit"] = match is not None
                        row["arms"][arm]["profile_match"] = match
                for round_index in range(args.rounds):
                    rotated = arms[round_index % len(arms):] + arms[:round_index % len(arms)]
                    for arm in rotated + list(reversed(rotated)):
                        with context(arm):
                            value = triton.testing.do_bench(
                                lambda arm=arm, inputs=inputs: attention(arm).unified_attention(**inputs),
                                warmup=args.warmup_ms,
                                rep=args.rep_ms,
                                return_mode="median",
                            )
                        if not math.isfinite(value) or value <= 0:
                            raise RuntimeError(f"Invalid latency {value}")
                        row["arms"][arm]["samples_ms"].append(value)
                atomic_json(args.output, data)
                print(f"Compared {index + 1}/{len(manifest)}: {case['id']}", flush=True)
                del inputs, reference
        data["profile_hits"] = sum(row["arms"]["profile"]["profile_hit"] for row in data["cases"])
        data["summaries"] = {}
        for reference_arm in ("fallback", "cached_autotune"):
            if reference_arm not in arms:
                continue
            for group in ("all", "hit", "miss", "bfloat16", "float16", "float8_e4m3fn", "2d", "3d"):
                rows = [
                    row for row in data["cases"]
                    if group == "all" or (group in ("hit", "miss") and row["arms"]["profile"]["profile_hit"] ==
                                          (group == "hit")) or row["workload"]["dtype"] == group or
                    (group in ("2d", "3d") and row["arms"]["profile"]["actual_launch"]["is_3d"] == (group == "3d"))
                ]
                ratios = [
                    statistics.median(row["arms"]["profile"]["samples_ms"]) /
                    statistics.median(row["arms"][reference_arm]["samples_ms"]) for row in rows
                ]
                data["summaries"][f"profile_over_{reference_arm}:{group}"] = summarize_ratios(ratios)
        data["complete"] = True
    except BaseException as error:
        data["session_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        atomic_json(args.output, data)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name, function in (("measure", measure), ("compare", compare)):
        command = modes.add_parser(name)
        command.add_argument("--manifest", required=True,
                             help="JSON list of development or frozen validation workloads")
        command.add_argument("--output", required=True, help="Raw measurement JSON; checkpointed atomically")
        command.add_argument("--device", default="xpu:0")
        command.add_argument("--rounds", type=int, default=5)
        command.add_argument("--warmup-ms", type=float, default=10)
        command.add_argument("--rep-ms", type=float, default=50)
        command.add_argument("--seed", type=int, default=1729)
        command.set_defaults(func=function)
        if name == "measure":
            command.add_argument("--confirmation-rounds", type=int, default=5)
            command.add_argument("--finalists", type=int, default=3)
        else:
            command.add_argument("--profile-folder")
            command.add_argument("--autotune-source", help="ua-FINAL attention source; its winner cache is retained")
            command.add_argument("--warmup-order", choices=("input", "reverse", "shuffle"), default="input")
    selection = modes.add_parser("select")
    selection.add_argument("--measurements", required=True)
    selection.add_argument("--output-folder", required=True)
    selection.add_argument("--runtime-module", help="Config module file for selection without importing vLLM")
    selection.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Worst-regret tie tolerance before geometric-mean tie-breaking",
    )
    args = parser.parse_args(argv)
    if args.mode in ("measure", "compare"):
        for name in ("rounds", "warmup_ms", "rep_ms", "confirmation_rounds", "finalists"):
            if hasattr(args, name) and getattr(args, name) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be positive")
    else:
        if not math.isfinite(args.tolerance) or args.tolerance < 0:
            parser.error("--tolerance must be finite and nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "select":
        result = select_results(
            json.loads(Path(args.measurements).read_text(encoding="utf-8")),
            runtime_module(args.runtime_module),
            args.output_folder,
            tolerance=args.tolerance,
        )
        print(f"Exported {result['accepted_keys']}/{result['total_keys']} keys")
        for path in result["profiles"].values():
            print(path)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
