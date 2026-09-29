"""Temporary same-runner comparison of online and offline unified attention."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import torch
import triton
import vllm.v1.attention.ops as ops
from torch.profiler import profile, ProfilerActivity, record_function

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / 'benchmarks/triton_kernels_benchmark/vllm/unified_attention'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--vllm-source', type=Path, required=True)
parser.add_argument('--main-ref', default='0849431d24201f4a249636fda4c4eac559406bb2')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--rounds', type=int, default=3)
parser.add_argument('--case', action='append', help='Limit local smoke measurements to these case IDs')
args = parser.parse_args()
P = args.output.resolve()
P.mkdir(parents=True, exist_ok=False)

def git(*arguments, cwd=ROOT):
    return subprocess.check_output(['git', *arguments], cwd=cwd, text=True).strip()

def prepare_source(label, patch):
    dest = P / label
    ops_path = Path('vllm/v1/attention/ops')
    shutil.copytree(args.vllm_source / ops_path, dest / ops_path)
    git('init', '-q', cwd=dest)
    patch_path = P / (label + '.patch')
    patch_path.write_text(patch)
    subprocess.run(['git', 'apply', '--check', '--include=vllm/v1/attention/ops/*', str(patch_path)], cwd=dest, check=True)
    subprocess.run(['git', 'apply', '--include=vllm/v1/attention/ops/*', str(patch_path)], cwd=dest, check=True)
    return dest / ops_path

patch_relative = 'benchmarks/triton_kernels_benchmark/vllm/unified_attention/unified_attention.patch'
main_ref = git('rev-parse', args.main_ref + '^{commit}')
OLD = prepare_source('main-source', git('show', main_ref + ':' + patch_relative) + '\n')
NEW = prepare_source('offline-source', (W / 'unified_attention.patch').read_text())

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

prefix = 'vllm.v1.attention.ops.'
load(OLD / 'triton_attention_helpers.py', prefix + 'triton_attention_helpers')
main = load(OLD / 'triton_unified_attention.py', '_diagnostic_main_ua')
for name in ['triton_attention_helpers', 'triton_unified_attention_config', 'triton_unified_attention']:
    setattr(ops, name, load(NEW / (name + '.py'), prefix + name))
r = ops.triton_unified_attention_config
new = ops.triton_unified_attention
t = load(W / 'tune_unified_attention.py', '_diagnostic_tuner')
os.environ['VLLM_TUNED_CONFIG_FOLDER'] = str(W / 'profiles')
device = torch.device('xpu:0')
torch.xpu.set_device(device)
assert len(main.kernel_unified_attention.configs) == 3, 'Review diagnostic assumptions if main pool changes'

# Execute CI's fixture and stop at the attention call to preserve its tensor layouts.
tree = ast.parse((W / 'unified_attention_benchmark.py').read_text())
node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'benchmark')
node.decorator_list = []
base = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('ref_paged_attn', '_dtype_size', 'is_enough_memory')]
glob = {'torch': torch, 'BENCHMARKING_CONFIG': {'verify': False}, 'is_td_patched': True}
exec(compile(ast.Module(body=base + [node], type_ignores=[]), str(W / 'unified_attention_benchmark.py'), 'exec'), glob)

# Use CI's timing loop; extract device events directly for both Torch versions.
profiler_source = W.parents[1] / 'benchmark_testing.py'
profiler_ast = ast.parse(profiler_source.read_text())
profiler_globals = dict(torch=torch, itertools=itertools, os=os, time=time, profile=profile, record_function=record_function, ProfilerActivity=ProfilerActivity, DEVICE='xpu', DEVICE_PROFILER_ACTIVITY=ProfilerActivity.XPU, DEVICE_PROFILER_TIME_KEY='xpu_time', synchronize=torch.xpu.synchronize)
profiler_functions = [n for n in profiler_ast.body if isinstance(n, ast.FunctionDef) and n.name == '_summarize_statistics']
exec(compile(ast.Module(body=profiler_functions, type_ignores=[]), str(profiler_source), 'exec'), profiler_globals)

class Captured(Exception):
    pass


def capture(**kwargs):
    raise Captured(kwargs)


def allocate(case):
    glob.update(unified_attention=capture,
                benchmark_suite=SimpleNamespace(assert_close=lambda fn, *a, **kw: fn()))
    try:
        glob['benchmark'](
            q_heads=case['q_heads'], k_heads=case['kv_heads'], head_size=case['head_size'],
            qdtype=getattr(torch, case['dtype']) if case['dtype'].startswith('float8') else None,
            seq_lens=list(zip(case['query_lens'], case['kv_lens'])),
            sliding_window=case['sliding_window'] or None, soft_cap=case['softcap'] or None,
            num_blocks=case['num_blocks'], block_size=case['block_size'], provider='triton-td')
    except Captured as error:
        return error.args[0]
    raise AssertionError('Fixture did not call attention')


def core(module):
    kernel = module.kernel_unified_attention
    while not isinstance(kernel, triton.runtime.jit.JITFunction):
        kernel = kernel.fn
    return kernel


@contextmanager
def observe(module):
    kernel = core(module)
    original = kernel.run
    logs = []

    def run(*args, **kwargs):
        named = dict(zip(kernel.arg_names, args)) | kwargs
        compiled = original(*args, **kwargs)
        metadata = compiled.metadata
        logs.append(dict(block_m=named['BLOCK_M'], tile_size=named['TILE_SIZE'],
                         num_warps=metadata.num_warps, num_stages=metadata.num_stages,
                         grf_mode=getattr(metadata, 'grf_mode', 'default'),
                         is_3d=named['IS_3D'], use_td=named['USE_TD'],
                         use_td_qo=named['USE_TD_QO'], kernel_hash=compiled.hash))
        return compiled

    kernel.run = run
    try:
        yield logs
    finally:
        kernel.run = original


def prof(fn):
 # Preserve CI's warmup, cache clearing and submission loop; adapt event extraction.
 source=ast.get_source_segment(profiler_source.read_text(),next(n for n in profiler_ast.body if isinstance(n,ast.FunctionDef) and n.name=='do_bench_upstream_pytorch_profiler'))
 offset=source.index('    profiling_func_filter =')
 source=source[:offset]+"""    trace = P / 'last-trace.json'
    prof.export_chrome_trace(str(trace))
    events = sorted([e for e in json.loads(trace.read_text())['traceEvents'] if e.get('ph') == 'X' and e.get('cat') == 'kernel' and e['name'] in ('kernel_unified_attention', 'reduce_segments')], key=lambda e: e['ts'])
    count = len(events) // n_repeat
    assert count in (1, 2) and len(events) == n_repeat * count, (len(events), n_repeat)
    samples = []
    for i in range(n_repeat):
        group = events[i * count:(i + 1) * count]
        assert group[0]['name'] == 'kernel_unified_attention'
        if count == 2:
            assert group[1]['name'] == 'reduce_segments'
        samples.append(sum(e['dur'] for e in group) / 1000)
    report.setdefault('profiler_samples', []).append(dict(status=json.loads((P/'status.json').read_text()), samples_ms=samples, kernels_per_call=count))
    return _summarize_statistics(torch.tensor(samples), quantiles, return_mode)
"""
 glob=profiler_globals|dict(P=P,json=json,report=report)
 exec(compile(source,str(profiler_source),'exec'),glob)
 return float(glob['do_bench_upstream_pytorch_profiler'](fn,n_warmup=25,n_repeat=10,quantiles=[.5,0,1])[3])


@contextmanager
def force_main(config):
    tuner = main.kernel_unified_attention
    previous_configs, previous_cache = tuner.configs, tuner.cache
    tuner.configs = [config]
    tuner.cache = {}
    try:
        yield
    finally:
        tuner.configs, tuner.cache = previous_configs, previous_cache


def save():
    t.atomic_json(P / 'results.json', report)


def status(**kw):
    t.atomic_json(P / 'status.json', dict(pid=os.getpid(), updated_at=time.time(), **kw))


def summarize():
    rows = []
    lines = ['# UA same-runner diagnostic', '', f"Compiler checkout: `{report['compiler_checkout']}`", f"Main UA patch: `{main_ref}`", '', '| Case | Arm | Config M/T/W/S/GRF | Graph µs | Cold profiler µs |', '|---|---|---|---:|---:|']
    for case in report['cases']:
        for arm, measurements in case['arms'].items():
            launch = measurements['launch']
            config = '/'.join(str(launch[k]) for k in ('block_m', 'tile_size', 'num_warps', 'num_stages', 'grf_mode'))
            graph = statistics.median(measurements['graph_ms']) * 1000
            profiler = statistics.median(measurements['profiler_ms']) * 1000
            lines.append(f"| {case['id']} | {arm} | {config} | {graph:.3f} | {profiler:.3f} |")
            rows.append(dict(case=case['id'], arm=arm, graph_us=graph, profiler_us=profiler, **launch))
    (P / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    if rows:
        with (P / 'timings.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

inputs_manifest = json.loads((ROOT / 'scripts/ua-ci-diagnostic-inputs.json').read_text())
manifest = inputs_manifest['cases']
if args.case:
    manifest = [c for c in manifest if c['id'] in args.case]
assert manifest and args.rounds > 0
report = dict(complete=False, compiler_checkout=git('rev-parse', 'HEAD'), main_ref=main_ref,
              vllm_pin=git('rev-parse', 'HEAD', cwd=args.vllm_source),
              torch=torch.__version__, triton=triton.__version__, triton_path=triton.__file__,
              device=torch.xpu.get_device_name(), device_properties=str(torch.xpu.get_device_properties(device)),
              runner=os.environ.get('RUNNER_NAME'), rounds=args.rounds,
              timing=dict(graph_warmup_ms=5, graph_rep_ms=30, profiler_warmup_ms=25, profiler_repeats=10),
              seeds=[], cases=[])
try:
    with torch.inference_mode():
        for case in inputs_manifest['seeds']:
            status(stage='prime_main', case=case['id'])
            inputs = allocate(case)
            with observe(main) as logs:
                main.unified_attention(**inputs)
                torch.xpu.synchronize()
            report['seeds'].append(dict(id=case['id'], launch=logs[-1]))
            save()
            del inputs
        for index, case in enumerate(manifest):
            status(stage='prepare', case=case['id'])
            inputs = allocate(case)
            with r.override_config(None):
                new.unified_attention(**inputs)
                key = r.last_resolution()[0]
            selected = r.resolve_config(key, device)
            filename = r.profile_filename(r.get_profile_identity(device), key.static_key())
            entries = json.loads((W / 'profiles' / filename).read_text())['entries']
            entry = next(e for e in entries if e['key'] == asdict(key.dynamic_key()))
            assert selected is not None and entry['config'] == asdict(selected), 'Expected exact shipped profile hit'
            with observe(main) as logs:
                main.unified_attention(**inputs)
                torch.xpu.synchronize()
            launch = logs[-1]
            main_config = r.AttentionConfig(**{k: launch[k] for k in ('block_m', 'tile_size', 'num_warps', 'num_stages', 'grf_mode')})
            arms = ['profile', 'main', 'main_config_on_offline'] + ['main_M' + str(c.kwargs['BLOCK_M']) for c in main.kernel_unified_attention.configs]
            row = dict(id=case['id'], workload=case, key=asdict(key), profile_file=filename,
                       profile_sha256=hashlib.sha256((W / 'profiles' / filename).read_bytes()).hexdigest(),
                       exact_profile_hit=True, selected=asdict(selected), main_config=asdict(main_config), arms={})
            def context(arm):
                if arm == 'main_config_on_offline':
                    return r.override_config(main_config)
                if arm.startswith('main_M'):
                    return force_main(next(c for c in main.kernel_unified_attention.configs if c.kwargs['BLOCK_M'] == int(arm[6:])))
                return nullcontext()
            def module_for(arm):
                return new if arm in ('profile', 'main_config_on_offline') else main
            for arm in arms:
                module = module_for(arm)
                with context(arm), observe(module) as logs:
                    module.unified_attention(**inputs)
                    torch.xpu.synchronize()
                row['arms'][arm] = dict(launch=logs[-1], graph_ms=[], profiler_ms=[])
            for round_index in range(args.rounds):
                order = arms.copy()
                random.Random(index * 100 + round_index).shuffle(order)
                for arm in order:
                    status(stage='measure', case=case['id'], round=round_index, arm=arm)
                    with context(arm):
                        fn = lambda: module_for(arm).unified_attention(**inputs)
                        metrics = [('graph_ms', lambda: t.benchmark_callable(fn, device, 5, 30)), ('profiler_ms', lambda: prof(fn))]
                        if round_index % 2:
                            metrics.reverse()
                        for metric, measure in metrics:
                            row['arms'][arm][metric].append(measure())
            report['cases'].append(row)
            save()
            summarize()
            print('DONE', case['id'], flush=True)
            del inputs
    report['complete'] = True
    save()
    status(stage='complete')
except BaseException as error:
    report['error'] = repr(error)
    save()
    status(stage='error', error=repr(error))
    raise
