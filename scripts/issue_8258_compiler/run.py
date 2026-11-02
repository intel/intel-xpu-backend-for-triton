"""Scoped baseline/candidate compilation and matched-input CI measurements."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SCOPE = json.loads((HERE / 'scope.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def group_shapes(group):
    if group == 'slow':
        return SCOPE['slow_shapes']
    return [row[1:] if row[0] == 1 else row for row in SCOPE['all_CI_shapes']]


def describe():
    print(json.dumps(SCOPE, indent=2))
    print('One B580 job; baseline and patched compiler built from the same pinned commit.')
    print('3 original shapes, normal B; all 39 original CI shapes, normal and transposed B.')
    print('Two CI shapes exceed B580 capacity and are explicitly recorded as skips in each layout.')
    print('Expected: 77 shape/layout comparisons, 3 allocation histories each = 231 comparisons.')
    print('16 stock configurations per unbatched shape, 9 per batched shape; independent selection.')


def snapshot(reports, policy):
    import triton
    source = Path(triton.__file__).parent
    destination = reports / 'runtimes' / policy / 'triton'
    shutil.copytree(source, destination, symlinks=False,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    library = destination / '_C' / 'libtriton.so'
    save(reports / f'{policy}-build.json', {
        'base_commit': SCOPE['base_commit'], 'policy': policy,
        'patch_sha256': sha(HERE / 'candidate.patch') if policy == 'candidate' else None,
        'libtriton_sha256': sha(library), 'snapshot': str(destination),
        'installed_source': str(source),
    })


@contextmanager
def free_gpu(reports):
    import timing
    lock_path = Path('/tmp/xe-forge-issue-8258-compiler-gpu.lock')
    checks = []
    with lock_path.open('a') as lock:
        deadline = time.monotonic() + 600
        while True:
            held = False
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                for _ in range(2):
                    current = timing.owners()
                    checks.append({'time': time.time(), 'owners': current})
                    save(reports / f'gpu-checks-{os.getpid()}.json', checks)
                    if current:
                        break
                    time.sleep(0.5)
                else:
                    break
            except BlockingIOError:
                pass
            if held:
                fcntl.flock(lock, fcntl.LOCK_UN)
            if time.monotonic() >= deadline:
                raise RuntimeError('GPU did not become free; no benchmark started')
            print('Waiting for an unused GPU', flush=True)
            time.sleep(10)
        yield


def capture(reports):
    with free_gpu(reports):
        import torch
        import triton
        if torch.xpu.device_count() != 1 or 'B580' not in torch.xpu.get_device_name(0):
            raise RuntimeError('Expected exactly one B580 device')
        target = triton.runtime.driver.active.get_current_target()
        save(reports / 'target.json', {'target': asdict(target)})
        save(reports / 'environment.json', {
            'torch': torch.__version__, 'triton': triton.__version__,
            'device': str(torch.xpu.get_device_properties(0)),
            'workflow_commit': os.environ.get('GITHUB_SHA'),
        })
    manifest = dict(SCOPE, shapes=[], source=str(HERE / 'gemm_benchmark.py'),
                    saved_target=str(reports / 'target.json'))
    if sha(HERE / 'gemm_benchmark.py') != SCOPE['source_sha256']:
        raise RuntimeError('Pinned benchmark source changed')
    if sha(HERE / 'candidate.patch') != SCOPE['patch_sha256']:
        raise RuntimeError('Candidate patch changed')
    save(reports / 'manifest.json', manifest)


def prepare(reports, group):
    folder = reports / group
    filename_group = 'slow' if group == 'slow' else 'all'
    variants = {policy: json.loads((folder / f'{filename_group}-{policy}.json').read_text())
                for policy in ('baseline', 'candidate')}
    baseline, candidate = variants.values()
    if not all(v['complete'] for v in variants.values()):
        raise RuntimeError('Incomplete compilation')
    for field in ('cases', 'memory_skips', 'configurations', 'batched_configurations'):
        left, right = baseline[field], candidate[field]
        if field == 'cases':
            left, right = left.keys(), right.keys()
        if left != right:
            raise RuntimeError(f'Compiler scopes differ: {field}')
    if len(baseline['cases']) + len(baseline['memory_skips']) != len(group_shapes(group)):
        raise RuntimeError('Missing input shapes')
    protocol = dict(shapes=[case['shape'] if case['batch'] == 1 else [case['batch']] + case['shape']
                           for case in baseline['cases'].values()], policies=['baseline', 'candidate'],
                    transpose_b=group == 'ci-bt', seed=0, atol=0.0001, rtol=0.001,
                    allocation_arrangements=SCOPE['allocation_arrangements'],
                    selection_rounds=SCOPE['selection_rounds'], confirmation_rounds=SCOPE['confirmation_rounds'],
                    selection_launches_per_round=None, confirmation_launches_per_round=None,
                    compute_warmup_ms=50, prune_factor=4, memory_skips=baseline['memory_skips'])
    save(folder / 'protocol.json', protocol)
    prepared = dict(protocol_sha256=sha(folder / 'protocol.json'), shapes=protocol['shapes'],
                    configs=baseline['configurations'], complete=True, limited_smoke=False, cases={})
    for key, case in baseline['cases'].items():
        prepared['cases'][key] = {}
        for index, config in case['configs'].items():
            images = {}
            for policy, record in variants.items():
                entry = record['cases'][key]['configs'][index]
                if entry['status'] != 'compiled':
                    raise RuntimeError(f'Compile failed: {group}/{key}/{index}/{policy}: {entry}')
                images[policy] = dict(folder=str(Path(entry['metadata']).parent), sha256=entry['sha256'])
            prepared['cases'][key][index] = dict(status='compiled', config=config['config'], variants=images)
    save(folder / 'prepared.json', prepared)


def measure(reports, group):
    import timing
    folder = reports / group
    protocol = json.loads((folder / 'protocol.json').read_text())
    prepared = json.loads((folder / 'prepared.json').read_text())
    if prepared['protocol_sha256'] != sha(folder / 'protocol.json'):
        raise RuntimeError('Protocol changed after preparation')
    output = folder / 'stock-tuning'
    output.mkdir()
    result = dict(complete=False, shapes={})
    save(output / 'result.json', result)
    with free_gpu(reports):
        import torch
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        if torch.xpu.device_count() != 1 or 'B580' not in torch.xpu.get_device_name(0):
            raise RuntimeError('Expected exactly one B580')
        for shape in protocol['shapes']:
            key = 'x'.join(map(str, shape)) if len(shape) == 3 else f'B{shape[0]}_' + 'x'.join(map(str, shape[1:]))
            row = result['shapes'][key] = {'allocations': {}}
            winners = None
            for skew in protocol['allocation_arrangements']:
                observed = timing.allocation(torch, prepared, protocol, shape, skew,
                                             output / f'{key}-skew{skew}.json', winners)
                winners = observed['winners']
                row['allocations'][str(skew)] = dict(complete=True, winners=winners)
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
                save(output / 'result.json', result)
            foreign = [owner for owner in timing.owners() if owner['pid'] != os.getpid()]
            if foreign:
                raise RuntimeError(f'Foreign GPU owner appeared: {foreign}')
    result['complete'] = True
    save(output / 'result.json', result)


def summarize(reports):
    lines = ['# Compiler V4 CI comparison', '', 'Positive percentages indicate lower candidate latency.', '',
             '| Group | Shape | Latency reduction, geometric mean | Worst candidate/baseline ratio |',
             '|---|---|---:|---:|']
    records, failures, skip_lines = [], [], []
    for group in ('slow', 'ci', 'ci-bt'):
        folder = reports / group
        protocol = json.loads((folder / 'protocol.json').read_text())
        if not json.loads((folder / 'stock-tuning/result.json').read_text())['complete']:
            raise RuntimeError(f'Incomplete measurement group: {group}')
        for shape in protocol['shapes']:
            key = 'x'.join(map(str, shape)) if len(shape) == 3 else f'B{shape[0]}_' + 'x'.join(map(str, shape[1:]))
            allocations = []
            for skew in protocol['allocation_arrangements']:
                row = json.loads((folder / f'stock-tuning/{key}-skew{skew}.json').read_text())
                if not row['complete']:
                    raise RuntimeError('Incomplete allocation comparison')
                def median(policy):
                    mapped = row['mapping'][row['selected'][policy]]
                    return statistics.median(statistics.median(round_) for round_ in row['confirmation_ms'][mapped])
                baseline, candidate = median('baseline'), median('candidate')
                allocations.append(dict(skew_mib=skew, baseline_ms=baseline, candidate_ms=candidate,
                                        ratio=candidate / baseline, winners=row['winners'], addresses=row['addresses']))
            ratio = math.exp(statistics.mean(math.log(a['ratio']) for a in allocations))
            worst = max(a['ratio'] for a in allocations)
            records.append(dict(group=group, shape=shape, geomean_ratio=ratio, worst_ratio=worst, allocations=allocations))
            lines.append(f'| {group} | {key} | {100*(1-ratio):+.2f}% | {worst:.4f} |')
            if ratio > 1.02 or worst > 1.02:
                failures.append(dict(group=group, shape=shape, geomean_ratio=ratio, worst_ratio=worst))
        skip_lines += ['', f'{group} memory skips: `{json.dumps(protocol["memory_skips"])}`', '']
    lines += skip_lines
    lines += ['', 'A >2% slowdown in the geometric mean or any allocation history flags review; this is not a statistical significance test.']
    save(reports / 'comparison.json', dict(complete=True, comparisons=records, regressions_over_2_percent=failures))
    (reports / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if failures:
        raise SystemExit('Performance review required: at least one >2% regression')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['describe', 'snapshot', 'capture', 'prepare', 'measure', 'summarize'])
    parser.add_argument('--reports', type=Path, default=Path(os.environ.get('GEMM_CI_REPORTS', 'reports')))
    parser.add_argument('--policy', choices=['baseline', 'candidate'])
    parser.add_argument('--group', choices=['slow', 'ci', 'ci-bt'])
    args = parser.parse_args()
    reports = args.reports.resolve()
    os.environ['GEMM_CI_REPORTS'] = str(reports)
    reports.mkdir(parents=True, exist_ok=True)
    if args.action == 'describe': describe()
    elif args.action == 'snapshot': snapshot(reports, args.policy)
    elif args.action == 'capture': capture(reports)
    elif args.action == 'prepare': prepare(reports, args.group)
    elif args.action == 'measure': measure(reports, args.group)
    else: summarize(reports)


if __name__ == '__main__':
    main()
