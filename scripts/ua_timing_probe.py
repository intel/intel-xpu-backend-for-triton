"""Local XPU timing diagnostic; never dispatches CI or changes installed code."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

CASES = {'ua_decode': 'ci-accuracy-0077', 'ua_prefill': 'ci-accuracy-0087',
         'ua_mixed': 'ci-accuracy-0091'}


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def scenarios(args):
    return [dict(workload=w, execution=e, eviction=c, events=t, profiler=p, trial=i)
            for w, e, c, t, p, i in itertools.product(
                args.workload, args.execution, args.eviction, args.events,
                args.profiler, range(args.trials))]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def workload(args, name, torch):
    if name.startswith('matmul'):
        torch.manual_seed(20)
        a = torch.randn(2048, 2048, device='xpu', dtype=torch.bfloat16)
        b = torch.randn_like(a)
        out = torch.empty_like(a)
        count = 1 if name == 'matmul_short' else 64
        def call():
            for _ in range(count):
                torch.mm(a, b, out=out)
        return call, out, dict(matmul_calls=count, dimension=2048)

    # Import patched source supplied by the caller, without applying patches.
    ops = args.ua_source / 'vllm/v1/attention/ops'
    import vllm.v1.attention.ops
    load(ops / 'triton_attention_helpers.py', 'vllm.v1.attention.ops.triton_attention_helpers')
    module = load(ops / 'triton_unified_attention.py', '_timing_main_attention')
    kernel = module.kernel_unified_attention
    # Freeze main's launch config before warmup: no online search during the probe.
    block_m = args.block_m or (32 if name == 'ua_decode' else 64)
    configs = [c for c in kernel.configs if c.kwargs['BLOCK_M'] == block_m]
    if len(configs) != 1:
        raise ValueError('Expected main autotuner with one matching BLOCK_M config')
    kernel.configs, kernel.cache = configs, {}
    cases = []
    for dtype in ('bfloat16', 'float8_e4m3fn'):
        cases += json.loads((args.repo / f'scripts/ua-ci-inputs-{dtype}.json').read_text())
    case = next(c for c in cases if c['id'] == CASES[name])
    source = args.repo / 'benchmarks/triton_kernels_benchmark/vllm/unified_attention/unified_attention_benchmark.py'
    tree = ast.parse(source.read_text())
    fixture = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'benchmark')
    fixture.decorator_list = []
    class Captured(Exception):
        pass
    def capture(**kwargs):
        raise Captured(kwargs)
    namespace = dict(torch=torch, BENCHMARKING_CONFIG={'verify': False}, is_td_patched=True,
                     unified_attention=capture,
                     benchmark_suite=SimpleNamespace(assert_close=lambda fn, *a, **kw: fn()))
    exec(compile(ast.Module(body=[fixture], type_ignores=[]), str(source), 'exec'), namespace)
    try:
        namespace['benchmark'](
            q_heads=case['q_heads'], k_heads=case['kv_heads'], head_size=case['head_size'],
            qdtype=getattr(torch, case['dtype']) if case['dtype'].startswith('float8') else None,
            seq_lens=list(zip(case['query_lens'], case['kv_lens'])),
            sliding_window=case['sliding_window'] or None, soft_cap=case['softcap'] or None,
            num_blocks=case['num_blocks'], block_size=case['block_size'], provider='triton-td')
    except Captured as result:
        inputs = result.args[0]
    else:
        raise RuntimeError('Fixture did not reach attention')
    output = inputs['out']
    metadata = dict(case=case, config=dict(configs[0].kwargs), num_warps=configs[0].num_warps,
                    num_stages=configs[0].num_stages,
                    source_sha256=hashlib.sha256((ops / 'triton_unified_attention.py').read_bytes()).hexdigest())
    return lambda: module.unified_attention(**inputs), output, metadata


def measure(call, evict, backend, repeats, scope):
    backend.synchronize()
    pairs = [(backend.Event(enable_timing=True), backend.Event(enable_timing=True))
             for _ in range(repeats if scope == 'per_replay' else 1)]
    start_wall = time.perf_counter()
    if scope == 'batch':
        pairs[0][0].record()
    for i in range(repeats):
        evict()
        if scope == 'per_replay':
            pairs[i][0].record()
        call()
        if scope == 'per_replay':
            pairs[i][1].record()
    if scope == 'batch':
        pairs[0][1].record()
    backend.synchronize()
    wall = (time.perf_counter() - start_wall) * 1000 / repeats
    samples = [a.elapsed_time(b) for a, b in pairs]
    if not all(math.isfinite(x) and x > 0 for x in samples):
        raise RuntimeError(f'Invalid event samples: {samples}')
    event = statistics.median(samples) if scope == 'per_replay' else samples[0] / repeats
    return dict(event_ms_per_call=event, raw_event_ms=samples, wall_ms_per_call=wall)


def wall_reference(call, evict, backend, repeats):
    backend.synchronize()
    begin = time.perf_counter()
    for _ in range(repeats):
        evict()
        call()
    backend.synchronize()
    return (time.perf_counter() - begin) * 1000 / repeats


def worker(args, scenario, destination):
    import torch
    backend = torch.xpu
    if not backend.is_available():
        raise RuntimeError('An XPU-enabled Torch installation and GPU are required')
    backend.set_device(0)
    result = dict(scenario=scenario, complete=False, rounds=[], environment=dict(
        torch=torch.__version__, torch_file=torch.__file__, torch_build=torch.__config__.show(),
        device=backend.get_device_name(), properties=str(backend.get_device_properties(0)),
        python=sys.version, runner=os.getenv('RUNNER_NAME'),
        environment={k: v for k, v in os.environ.items()
                     if k.startswith(('SYCL_', 'UR_', 'ZE_', 'ONEAPI_', 'DLE_', 'COMPILER_'))}))
    save(destination, result)
    with torch.inference_mode():
        fn, output, metadata = workload(args, scenario['workload'], torch)
        result['workload'] = metadata
        for _ in range(3):
            fn()
        backend.synchronize()
        expected = output.clone()
        backend.synchronize()
        cache = torch.empty(args.cache_mb * 1024 * 1024 // 4, device='xpu', dtype=torch.int32)
        cache.zero_()
        backend.synchronize()
        graphs = []
        def capture(call):
            graph = backend.XPUGraph()
            with backend.graph(graph):
                call()
            graphs.append(graph)
            return graph.replay
        call = capture(fn) if scenario['execution'] == 'graph' else fn
        if scenario['eviction'] == 'on':
            evict = capture(cache.zero_) if scenario['execution'] == 'graph' else cache.zero_
        else:
            evict = lambda: None
        for _ in range(3):
            evict()
            call()
        backend.synchronize()
        torch.testing.assert_close(output, expected)
        result['replay_matches_eager'] = True
        save(destination, result)
        for rnd in range(args.rounds):
            if scenario['profiler'] == 'before_round':
                # Graphs already exist: isolate the effect of entering/exiting a profiler session.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.XPU]) as prof:
                    for _ in range(3):
                        evict()
                        call()
                    backend.synchronize()
                prof.export_chrome_trace(str(destination.with_name(destination.stem + f'-round{rnd}-trace.json')))
            # Alternate reference order to reduce systematic drift bias.
            reference = None
            if rnd % 2 == 0:
                reference = wall_reference(call, evict, backend, args.repeats)
            row = measure(call, evict, backend, args.repeats, scenario['events'])
            if reference is None:
                reference = wall_reference(call, evict, backend, args.repeats)
            row['no_events_wall_ms_per_call'] = reference
            result['rounds'].append(row)
            save(destination, result)
        backend.synchronize()
        torch.testing.assert_close(output, expected)
        for graph in reversed(graphs):
            graph.reset()
    result['complete'] = True
    save(destination, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True, help='CI tuning worktree containing fixture and manifests')
    parser.add_argument('--ua-source', type=Path, help='Prepared main TD source root containing vllm/v1/attention/ops')
    parser.add_argument('--workload', nargs='+', choices=['matmul_short', 'matmul_long', *CASES],
                        default=['matmul_short', 'ua_decode', 'ua_prefill'])
    parser.add_argument('--execution', nargs='+', choices=['eager', 'graph'], default=['eager', 'graph'])
    parser.add_argument('--eviction', nargs='+', choices=['off', 'on'], default=['off', 'on'])
    parser.add_argument('--events', nargs='+', choices=['per_replay', 'batch'], default=['per_replay', 'batch'])
    parser.add_argument('--profiler', nargs='+', choices=['none', 'before_round'], default=['none', 'before_round'])
    parser.add_argument('--block-m', type=int, choices=[16, 32, 64], help='Override fixed main config; default decode=32, prefill/mixed=64')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--cache-mb', type=int, default=256)
    parser.add_argument('--timeout', type=int, default=600, help='Per-process timeout in seconds')
    parser.add_argument('--dry-run', action='store_true', help='Print process matrix without importing Torch')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    for field in ('trials', 'rounds', 'repeats', 'cache_mb', 'timeout'):
        if getattr(args, field) < 1:
            parser.error(f'--{field.replace("_", "-")} must be positive')
    if args.worker:
        worker(args, json.loads(args.worker), args.output)
        return
    matrix = scenarios(args)
    if args.dry_run:
        print(json.dumps(dict(processes=len(matrix), scenarios=matrix), indent=2))
        return
    if any(w.startswith('ua_') for w in args.workload) and args.ua_source is None:
        parser.error('--ua-source is required for attention workloads')
    args.output.mkdir(parents=True, exist_ok=False)
    random.Random(1729).shuffle(matrix)
    save(args.output / 'plan.json', dict(scenarios=matrix, rounds=args.rounds, repeats=args.repeats,
                                       cache_mb=args.cache_mb))
    results = []
    summaries = []
    for i, scenario in enumerate(matrix):
        dest = args.output / f'{i:03d}.json'
        command = [sys.executable, str(Path(__file__).resolve()), '--repo', str(args.repo.resolve()),
                   '--output', str(dest.resolve()), '--worker', json.dumps(scenario),
                   '--rounds', str(args.rounds), '--repeats', str(args.repeats), '--cache-mb', str(args.cache_mb)]
        if args.block_m is not None:
            command += ['--block-m', str(args.block_m)]
        if args.ua_source:
            command += ['--ua-source', str(args.ua_source.resolve())]
        print(f'{i+1}/{len(matrix)} {scenario}', flush=True)
        try:
            with dest.with_suffix('.log').open('w') as log:
                run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            row = dict(scenario=scenario, file=dest.name, returncode=run.returncode)
        except subprocess.TimeoutExpired:
            row = dict(scenario=scenario, file=dest.name, error='timeout')
        results.append(row)
        save(args.output / 'status.json', dict(complete=False, results=results))
        if row.get('returncode') != 0:
            raise SystemExit(f'Worker failed; see {dest.with_suffix(".log")}')
        data = json.loads(dest.read_text())
        values = [r['event_ms_per_call'] for r in data['rounds']]
        summaries.append(dict(scenario, file=dest.name, event_median_ms=statistics.median(values),
                              event_min_ms=min(values), event_max_ms=max(values),
                              round_spread=max(values) / min(values),
                              wall_median_ms=statistics.median(r['wall_ms_per_call'] for r in data['rounds']),
                              no_events_wall_median_ms=statistics.median(
                                  r['no_events_wall_ms_per_call'] for r in data['rounds'])))
        with (args.output / 'summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    save(args.output / 'status.json', dict(complete=True, results=results))


if __name__ == '__main__':
    main()
