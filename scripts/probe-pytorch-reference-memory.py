#!/usr/bin/env python3
"""Run PR #8595's unchanged PyTorch reference without its memory guard."""

import argparse
import ast
from collections import Counter
import csv
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

SOURCE = (Path(__file__).resolve().parents[1] / 'benchmarks/triton_kernels_benchmark/vllm/'
          'unified_attention/unified_attention_benchmark.py')


def load_reference(fp8):
    import torch

    if not torch.xpu.is_available():
        raise RuntimeError('This diagnostic requires an XPU device')
    properties = torch.xpu.get_device_properties()
    namespace = {
        'torch': torch,
        'product': product,
        'IS_FP8': fp8,
        'TOTAL_MEMORY_BYTES': properties.total_memory,
        'BENCHMARKING_CONFIG': {'verify': True},
    }
    functions = {'ref_paged_attn', '_dtype_size', 'is_enough_memory', '_build_attention_configs'}
    constants = {
        'MMAP_BLOCK_SIZES',
        'NUM_BLOCKS',
        'SEQ_LENS_LIGHT',
        'SEQ_LENS_HEAVY',
        'SEQ_LENS',
        'MODELS_BF16',
        'MODELS_FP8',
        'ATTENTION_CONFIGS_BF16',
        'ATTENTION_CONFIGS_FP8',
    }
    tree = ast.parse(SOURCE.read_text())
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions) or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants for target in node.targets))
    ]
    # Load the exact reference and configurations without importing vLLM or SYCL.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    estimator = next(node for node in ast.parse(SOURCE.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name == 'is_enough_memory')
    estimator.name = 'estimated_bytes'
    # Expose the existing estimator's byte total without changing its arithmetic.
    for node in ast.walk(estimator):
        if isinstance(node, ast.Return):
            node.value = ast.Name(id='total_memory', ctx=ast.Load())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[estimator], type_ignores=[])), str(SOURCE), 'exec'),
         namespace)
    return torch, namespace


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def manifest(path):
    cases = []
    for family in ('bf16', 'fp8'):
        torch, reference = load_reference(family == 'fp8')
        for index, config in enumerate(reference[f'ATTENTION_CONFIGS_{family.upper()}']):
            estimated = reference['estimated_bytes'](config, provider='pytorch')
            cases.append({
                'case': f'{family}:{index}',
                'configuration': repr(config),
                'estimated_bytes': estimated,
                'estimated_fits_80_percent': reference['is_enough_memory'](config, provider='pytorch'),
            })
    write_json(
        path, {
            'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'device': torch.xpu.get_device_name(),
            'total_device_bytes': reference['TOTAL_MEMORY_BYTES'],
            'torch_version': torch.__version__,
            'cases': cases,
        })


def worker(case, output):
    family, index = case.split(':')
    torch, reference = load_reference(family == 'fp8')
    config = reference[f'ATTENTION_CONFIGS_{family.upper()}'][int(index)]
    q_heads, k_heads, head_size, _, seq_lens, sliding_window, soft_cap, num_blocks, block_size = config
    result = {'case': case, 'status': 'error', 'stage': 'inputs'}
    started = time.monotonic()
    torch.manual_seed(20)
    torch.set_default_device('xpu')
    torch.xpu.empty_cache()
    torch.xpu.reset_peak_memory_stats()
    try:
        query_lens = [entry[0] for entry in seq_lens]
        kv_lens = [entry[1] for entry in seq_lens]
        query = torch.randn(sum(query_lens), q_heads, head_size, dtype=torch.bfloat16)
        key_cache = torch.randn(num_blocks, block_size, k_heads, head_size, dtype=torch.bfloat16)
        value_cache = torch.randn_like(key_cache)
        max_blocks = (max(kv_lens) + block_size - 1) // block_size
        block_tables = torch.randint(0, num_blocks, (len(seq_lens), max_blocks), dtype=torch.int32)
        torch.xpu.synchronize()
        result['input_allocated_bytes'] = torch.xpu.memory_allocated()
        result['stage'] = 'reference'
        # Match benchmark inputs and call the original function, bypassing only the guard.
        with torch.inference_mode():
            for _ in range(2):
                actual = reference['ref_paged_attn'](query, key_cache, value_cache, query_lens, kv_lens, block_tables,
                                                     head_size**-0.5, sliding_window, soft_cap)
                torch.xpu.synchronize()
                result['output_shape'] = list(actual.shape)
                del actual
        result['status'] = 'fit'
    except Exception as error:
        message = str(error)
        result['status'] = ('oom' if isinstance(error, torch.OutOfMemoryError) or 'out of memory' in message.lower()
                            else 'error')
        result['error'] = f'{type(error).__name__}: {message}'
        traceback.print_exc()
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        result['peak_allocated_bytes'] = torch.xpu.max_memory_allocated()
        result['peak_reserved_bytes'] = torch.xpu.max_memory_reserved()
        write_json(output, result)
        print(json.dumps(result), flush=True)


def summarize(report, reports):
    rows = report['results']
    counts = Counter(row['status'] for row in rows)
    rejected = [row for row in rows if not row['estimated_fits_80_percent']]
    lines = [
        f"# PyTorch reference memory: {report['device']}",
        '',
        f"Device memory: {report['total_device_bytes'] / 2**30:.2f} GiB. PyTorch: {report['torch_version']}.",
        f"Attempted {len(rows)}/{report['total_cases']} configurations. Outcomes: {dict(counts)}.",
        f"Of {len(rejected)} cases rejected by the 80% estimate, "
        f"{sum(row['status'] == 'fit' for row in rejected)} actually completed.",
        '',
        'Each case runs in a fresh process. The unchanged reference is called twice, '
        'releasing its output between calls. No estimate-based skips or SYCL fallback are used.',
        'Peak allocated/reserved memory covers inputs and reference execution. '
        'An OOM peak records allocations before failure, not the memory needed to complete.',
        '',
        '| Case | Guard allows | Outcome | Estimate GiB | Peak allocated GiB | Peak reserved GiB |',
        '|---|---|---|---:|---:|---:|',
    ]
    for row in rows:
        peaks = [
            f"{row[key] / 2**30:.3f}" if key in row else '—' for key in ('peak_allocated_bytes', 'peak_reserved_bytes')
        ]
        lines.append(f"| {row['case']} | {row['estimated_fits_80_percent']} | {row['status']} | "
                     f"{row['estimated_bytes'] / 2**30:.3f} | {peaks[0]} | {peaks[1]} |")
    summary = '\n'.join(lines) + '\n'
    (reports / 'summary.md').write_text(summary)
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as stream:
            stream.write(summary)
    with (reports / 'results.csv').open('w', newline='') as stream:
        fields = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print('\n'.join(lines[:6]), flush=True)


def sweep(args):
    reports = Path(args.reports).resolve()
    reports.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, __file__, '--manifest', str(reports / 'manifest.json')], check=True, timeout=180)
    report = json.loads((reports / 'manifest.json').read_text())
    report['total_cases'] = len(report['cases'])
    report['results'] = []
    cases = report.pop('cases')
    if args.case_limit:
        cases = cases[:args.case_limit]
    for case in cases:
        case_id = case['case'].replace(':', '-')
        output = reports / f'{case_id}.json'
        started = time.monotonic()
        with (reports / f'{case_id}.log').open('w') as log:
            try:
                process = subprocess.run([sys.executable, __file__, '--worker', case['case'], '--output',
                                          str(output)], stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout,
                                         check=False)
                result = json.loads(output.read_text()) if output.exists() else {
                    'status': 'process_error',
                    'returncode': process.returncode,
                }
                if process.returncode != 0 and result.get('status') == 'fit':
                    result.update(status='process_error', returncode=process.returncode)
            except subprocess.TimeoutExpired:
                result = {'status': 'timeout', 'timeout_seconds': args.timeout}
        row = {**case, **result, 'wall_seconds': time.monotonic() - started}
        report['results'].append(row)
        write_json(reports / 'results.json', report)
        print(
            f"{case['case']}: {row['status']}; estimated fit={case['estimated_fits_80_percent']}; "
            f"peak allocated={row.get('peak_allocated_bytes', 'unavailable')}", flush=True)
    summarize(report, reports)
    # OOMs are diagnostic results; infrastructure errors make the run incomplete.
    return int(any(row['status'] not in ('fit', 'oom') for row in report['results']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', default='reports/pytorch-reference-memory')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--case-limit', type=int, default=0)
    parser.add_argument('--manifest')
    parser.add_argument('--worker')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.manifest:
        manifest(args.manifest)
        return 0
    if args.worker:
        worker(args.worker, args.output)
        return 0
    return sweep(args)


if __name__ == '__main__':
    sys.exit(main())
