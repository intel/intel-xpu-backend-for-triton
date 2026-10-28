"""Compare independent full-group and expanded head-group searches in one CI job."""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time


ARMS = ("full", "expanded")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def config_id(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


class Capture:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def record(*args, **kwargs):
            assert not args
            self.calls.append((grid, kwargs))
        return record


def allocate(torch, triton, case, seed):
    torch.manual_seed(seed)
    qh, kh, dim, block = (case[k] for k in ("q_heads", "k_heads", "head_size", "block_size"))
    pairs = case["seq_lens"]
    qlens, klens = zip(*pairs)
    q = torch.randn(sum(qlens), qh, dim, dtype=torch.bfloat16, device="xpu")
    k = torch.randn(case["num_blocks"], block, kh, dim, dtype=torch.bfloat16, device="xpu")
    v = torch.randn_like(k)
    if case["precision"] == "FP8":
        q, k, v = (tensor.to(torch.float8_e4m3fn) for tensor in (q, k, v))
    cumulative = [0]
    for length in qlens:
        cumulative.append(cumulative[-1] + length)
    table = torch.randint(case["num_blocks"], (len(pairs), triton.cdiv(max(klens), block)),
                          dtype=torch.int32, device="xpu")
    inputs = dict(q=q, k=k, v=v, cu_seqlens_q=torch.tensor(cumulative, dtype=torch.int32, device="xpu"),
                  seqused_k=torch.tensor(klens, dtype=torch.int32, device="xpu"),
                  max_seqlen_q=max(qlens), max_seqlen_k=max(klens), softmax_scale=dim**-.5,
                  causal=True, window_size=(case["sliding_window"] - 1, 0)
                  if case["sliding_window"] else (-1, -1), block_table=table,
                  softcap=case["soft_cap"] or 0, q_descale=None, k_descale=None, v_descale=None,
                  seq_threshold_3D=32, num_par_softmax_segments=16, use_td=True)
    if case["precision"] == "FP8":
        inputs.update(k_descale=torch.rand(len(pairs), kh, device="xpu"),
                      v_descale=torch.rand(len(pairs), kh, device="xpu"))
    shape = (sum(qlens), qh, dim)
    guards = []
    for _ in ARMS:
        storage = torch.full((math.prod(shape) + 512,), 123., dtype=torch.bfloat16, device="xpu")
        guards.append((storage, storage[256:-256].view(shape)))
    return inputs, guards


def reference(torch, case, inputs):
    pairs = case["seq_lens"]
    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    qh, kh, dim, block = (case[k] for k in ("q_heads", "k_heads", "head_size", "block_size"))
    table = inputs["block_table"].cpu().long()
    sequences = sorted({0, len(pairs) // 2, len(pairs) - 1})
    offsets, indices, outputs = [0], [], []
    for qlen, _ in pairs:
        offsets.append(offsets[-1] + qlen)

    def gather(tensor, *indices):
        if case["precision"] == "FP8":
            return tensor.view(torch.uint8)[indices].view(tensor.dtype).float().cpu()
        return tensor[indices].float().cpu()

    for seq in sequences:
        qlen, klen = pairs[seq]
        positions = sorted({0, min(15, qlen - 1), min(16, qlen - 1), qlen // 2, qlen - 1})
        tokens = torch.arange(klen)
        pages, slots = table[seq, tokens // block].to("xpu"), (tokens % block).to("xpu")
        keys, values = (gather(tensor, pages, slots).repeat_interleave(qh // kh, dim=1)
                        for tensor in (k, v))
        absolute = torch.tensor([offsets[seq] + i for i in positions], device="xpu")
        queries = gather(q, absolute)
        scores = torch.einsum("qhd,khd->hqk", queries, keys) * dim**-.5
        if case["soft_cap"]:
            scores = case["soft_cap"] * torch.tanh(scores / case["soft_cap"])
        query_positions = torch.tensor(positions) + klen - qlen
        valid = tokens[None, :] <= query_positions[:, None]
        if case["sliding_window"]:
            valid &= tokens[None, :] > query_positions[:, None] - case["sliding_window"]
        scores.masked_fill_(~valid[None], -float("inf"))
        outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), values))
        indices.extend(offsets[seq] + i for i in positions)
    return torch.tensor(indices, device="xpu"), torch.cat(outputs)


def calls_for(torch, ua, inputs, output):
    main, reduction = Capture(), Capture()
    original_main, original_reduction = ua.kernel_unified_attention, ua.reduce_segments
    try:
        ua.kernel_unified_attention, ua.reduce_segments = main, reduction
        segments = dict(softmax_segm_output=torch.empty(32, output.shape[1], 16, output.shape[2],
                                                       dtype=torch.float32, device="xpu"),
                        softmax_segm_max=torch.empty(32, output.shape[1], 16, device="xpu"),
                        softmax_segm_expsum=torch.empty(32, output.shape[1], 16, device="xpu"))
        ua.unified_attention(**inputs, out=output, **segments)
    finally:
        ua.kernel_unified_attention, ua.reduce_segments = original_main, original_reduction
    assert len(main.calls) == 1 and len(reduction.calls) <= 1
    return main.calls[0], reduction.calls, segments


def launch_for(ua, calls, config):
    (grid, kwargs), reductions, _ = calls
    merged = {**kwargs, **config}

    def launch():
        compiled = ua.kernel_unified_attention.fn[grid](**merged)
        for reduction_grid, reduction_kwargs in reductions:
            ua.reduce_segments[reduction_grid](**reduction_kwargs)
        return compiled
    return launch


def graph_for(torch, launch):
    compiled = launch()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        launch()
    for _ in range(3):
        graph.replay()
    torch.xpu.synchronize()
    return graph, compiled


def measure(torch, graph, replays):
    start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / replays


def samples(torch, graph, count):
    estimate = measure(torch, graph, 2)
    replays = max(1, min(64, math.ceil(3 / max(estimate, 1e-6))))
    values = [measure(torch, graph, replays) for _ in range(count)]
    assert all(math.isfinite(value) and value > 0 for value in values)
    return values


def check(torch, case, output, indices, expected, storage):
    atol, rtol = (.375001, .15) if case["precision"] == "FP8" else (.025, .01)
    actual = output[indices].float().cpu()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    assert bool(torch.all(storage[:256] == 123)) and bool(torch.all(storage[-256:] == 123))
    return float((actual - expected).abs().max())


def tune(torch, ua, case, inputs, guards, rng, directory):
    calls = calls_for(torch, ua, inputs, guards[0][1])
    named_args = calls[0][1]
    candidates = ua.prune_attention_configs(ua.ATTENTION_AUTOTUNE_CONFIGS, named_args)
    configs = {config_id(c.all_kwargs()): c.all_kwargs() for c in candidates}
    indices, expected = reference(torch, case, inputs)
    screening = {arm: {} for arm in ARMS}
    graphs, failures, correctness = {}, [], []
    ordered = list(configs)
    rng.shuffle(ordered)
    resource_errors = (torch.OutOfMemoryError,)
    from triton.runtime.errors import OutOfResources
    from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure
    resource_errors += (OutOfResources, CompilationError, CompileTimeAssertionFailure)
    try:
        for index, ident in enumerate(ordered):
            config = configs[ident]
            try:
                graph, _ = graph_for(torch, launch_for(ua, calls, config))
            except resource_errors as exc:
                failures.append(dict(config=config, error=type(exc).__name__ + ": " + str(exc)[:1000]))
                continue
            graphs[ident] = graph
            correctness.append(check(torch, case, guards[0][1], indices, expected, guards[0][0]))
            arms = list(ARMS) if config["HEADS_PER_PROGRAM"] == 0 else ["expanded"]
            rng.shuffle(arms)
            for arm in arms:
                screening[arm][ident] = samples(torch, graph, 3)
            if index % 16 == 0:
                dump(directory / "progress.json", dict(stage="screening", tried=index + 1,
                                                         candidates=len(configs), failures=len(failures)))
                print(case["id"], "screened", index + 1, "/", len(configs), flush=True)
        shortlists = {arm: sorted(screening[arm], key=lambda k: statistics.median(screening[arm][k]))[:6]
                      for arm in ARMS}
        assert all(shortlists.values()), "No valid candidates for an arm"
        refinement = {arm: {ident: [] for ident in shortlists[arm]} for arm in ARMS}
        for _ in range(8):
            trials = [(arm, ident) for arm in ARMS for ident in shortlists[arm]]
            rng.shuffle(trials)
            for arm, ident in trials:
                refinement[arm][ident].extend(samples(torch, graphs[ident], 1))
        winners = {arm: min(refinement[arm], key=lambda ident: statistics.median(refinement[arm][ident]))
                   for arm in ARMS}
        result = dict(candidate_counts={arm: len(screening[arm]) for arm in ARMS},
                      generated_candidates=len(configs), compile_failures=failures, screening_ms=screening,
                      refinement_ms=refinement, winners={arm: configs[winners[arm]] for arm in ARMS},
                      max_screening_abs_error=max(correctness), is_3d=named_args["IS_3D"],
                      complete=True)
        dump(directory / "tuning.json", result)
        return result
    finally:
        for graph in graphs.values():
            graph.reset()


def confirm(torch, triton, ua, case, winners, cycle, rng):
    inputs, guards = allocate(torch, triton, case, 20261028 + cycle * 104729)
    indices, expected = reference(torch, case, inputs)
    calls = {arm: calls_for(torch, ua, inputs, guards[i][1]) for i, arm in enumerate(ARMS)}
    graphs, binaries, errors = {}, {}, {}
    try:
        for i, arm in enumerate(ARMS):
            graphs[arm], compiled = graph_for(torch, launch_for(ua, calls[arm], winners[arm]))
            errors[arm] = check(torch, case, guards[i][1], indices, expected, guards[i][0])
            binaries[arm] = {name: hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()
                             for name, value in compiled.asm.items() if name in ("spv", "zebin", "cubin")}
        replays = {arm: max(1, min(64, math.ceil(3 / max(measure(torch, graphs[arm], 2), 1e-6))))
                   for arm in ARMS}
        warmup_until = time.monotonic() + .5
        while time.monotonic() < warmup_until:
            for arm in ARMS:
                graphs[arm].replay()
            torch.xpu.synchronize()
        values = {arm: [] for arm in ARMS}
        for round_index in range(16):
            order = list(ARMS if round_index % 2 == 0 else reversed(ARMS))
            if cycle % 2:
                order.reverse()
            for arm in order:
                values[arm].append(measure(torch, graphs[arm], replays[arm]))
        for i, arm in enumerate(ARMS):
            errors[arm] = check(torch, case, guards[i][1], indices, expected, guards[i][0])
        torch.testing.assert_close(guards[0][1], guards[1][1],
                                   atol=.375001 if case["precision"] == "FP8" else .025,
                                   rtol=.15 if case["precision"] == "FP8" else .01)
        return dict(seed=20261028 + cycle * 104729, timings_ms=values,
                    max_sampled_abs_error=errors, binary_sha256=binaries, replays=replays)
    finally:
        for graph in graphs.values():
            graph.reset()


def summarize(output):
    rows = []
    for path in sorted(output.glob("cases/*/result.json")):
        data = json.loads(path.read_text())
        if not data.get("complete"):
            continue
        ratios = [[math.log(b / a) for a, b in zip(c["timings_ms"]["full"], c["timings_ms"]["expanded"])]
                  for c in data["confirmation"]]
        center = statistics.mean(statistics.mean(cycle) for cycle in ratios)
        rng = random.Random(71911)
        estimates = []
        for _ in range(2000):
            cycles = [rng.choice(ratios) for cycle in ratios]
            estimates.append(statistics.mean(statistics.mean(rng.choices(c, k=len(c))) for c in cycles))
        estimates.sort()
        case = data["case"]
        seqs = case["seq_lens"]
        category = "decode" if all(q == 1 for q, k in seqs) else "prefill" if all(q == k for q, k in seqs) else "mixed"
        rows.append(dict(id=case["id"], precision=case["precision"], category=category,
                         q_heads=case["q_heads"], k_heads=case["k_heads"], block_size=case["block_size"],
                         num_blocks=case["num_blocks"], log_latency_ratio=center,
                         latency_reduction_pct=100 * (1 - math.exp(center)),
                         ci95_pct=[100 * (1 - math.exp(estimates[1949])), 100 * (1 - math.exp(estimates[49]))],
                         winners=data["tuning"]["winners"]))
    summaries = []
    for precision in ("BF16", "FP8"):
        for category in ("all", "prefill", "decode", "mixed"):
            subset = [r for r in rows if r["precision"] == precision and (category == "all" or r["category"] == category)]
            if subset:
                summaries.append(dict(precision=precision, category=category, cases=len(subset),
                                      geomean_latency_reduction_pct=100 * (1 - math.exp(statistics.mean(
                                          r["log_latency_ratio"] for r in subset)))))
    dump(output / "summary.json", dict(completed_cases=len(rows), groups=summaries, cases=rows))
    lines = ["# Paired head-group ablation", "", "Same PR source and output-store implementation in both arms; only the permitted head-group search differs.",
             "Each pool uses independent screening/refinement samples; frozen winners are compared on three fresh allocations with 16 balanced timing rounds each.",
             "", "| Precision | Workloads | Cases | Geometric-mean latency reduction |", "|---|---|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['precision']} | {row['category']} | {row['cases']} | {row['geomean_latency_reduction_pct']:+.2f}% |")
    lines += ["", "Positive reductions favor the expanded search. Confidence intervals in summary.json resample allocations and timing rounds, not devices or independent tuning campaigns.",
              "This includes the descriptor eligibility enabled by smaller groups; the flattening rule is identical in both arms. Cold compilation and tuning costs are excluded from warm timings.",
              "", f"Completed cases: {len(rows)} / 97."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def run(args, inventory):
    import torch
    import triton
    torch.set_num_threads(4)
    source = args.vllm / "vllm/v1/attention/ops/triton_unified_attention.py"
    helper = source.with_name("triton_attention_helpers.py")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha == inventory["source_sha256"], "Applied PR source differs from the reviewed source"
    load_module("vllm.v1.attention.ops.triton_attention_helpers", helper)
    ua = load_module("vllm.v1.attention.ops.triton_unified_attention", source)
    runtime = dict(source_sha256=source_sha, torch=str(torch.__version__), triton=triton.__version__,
                   device=torch.xpu.get_device_name(0), run_id=os.getenv("GITHUB_RUN_ID"),
                   commit=os.getenv("GITHUB_SHA"), runner=os.getenv("RUNNER_NAME"),
                   inputs=inventory, started=time.time(), complete=False)
    dump(args.output / "runtime.json", runtime)
    rng = random.Random(629711)
    selected = [case for case in inventory["cases"] if not args.case or args.case in case["id"]]
    for case in selected:
        directory = args.output / "cases" / case["id"]
        if (directory / "result.json").exists():
            continue
        print("Starting", case["id"], flush=True)
        inputs, guards = allocate(torch, triton, case, 20)
        tuning = tune(torch, ua, case, inputs, guards, rng, directory)
        del inputs, guards
        gc.collect()
        torch.xpu.empty_cache()
        confirmation = []
        for cycle in range(3):
            confirmation.append(confirm(torch, triton, ua, case, tuning["winners"], cycle, rng))
            gc.collect()
            torch.xpu.empty_cache()
        dump(directory / "result.json", dict(case=case, tuning=tuning, confirmation=confirmation, complete=True))
        summarize(args.output)
        print("Completed", case["id"], tuning["winners"], flush=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    runtime.update(complete=True, finished=time.time())
    dump(args.output / "runtime.json", runtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, default=Path(__file__).with_name("head_group_ablation_workloads.json"))
    parser.add_argument("--vllm", type=Path, default=Path("vllm"))
    parser.add_argument("--output", type=Path, default=Path("reports/head-group-ablation"))
    parser.add_argument("--case", default="")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    inventory = json.loads(args.workloads.read_text())
    for case in inventory["cases"]:
        case["seq_lens"] = inventory["sequence_patterns"][case["sequence_pattern"]]
    assert len(inventory["cases"]) == 97
    assert len({case["id"] for case in inventory["cases"]}) == 97
    if args.plan_only:
        print(json.dumps({"cases": len(inventory["cases"]), "source_sha256": inventory["source_sha256"]}))
    elif args.report_only:
        summarize(args.output)
    else:
        try:
            run(args, inventory)
        finally:
            summarize(args.output)
            summary = args.output / "summary.md"
            if os.getenv("GITHUB_STEP_SUMMARY") and summary.exists():
                with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
                    stream.write(summary.read_text())


if __name__ == "__main__":
    main()
