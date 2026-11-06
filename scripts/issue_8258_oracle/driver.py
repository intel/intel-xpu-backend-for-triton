"""Portable three-policy compilation, retuning and confirmation on CI."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

from native_image import fingerprint
from oracle import confirmed_ms, require_matching_availability

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCOPE = json.loads((HERE / 'scope.json').read_text())
POLICIES = SCOPE['policies']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def capture(reports):
    from gpu_guard import acquire
    with acquire(reports / 'capture-availability.json'):
        import torch
        import triton
        if torch.xpu.device_count() != 1 or 'B580' not in torch.xpu.get_device_name(0):
            raise RuntimeError('Expected one B580 GPU')
        save(reports / 'target.json', {'target': asdict(triton.runtime.driver.active.get_current_target())})
        save(reports / 'environment.json', {
            'torch': torch.__version__, 'triton': triton.__version__,
            'device': str(torch.xpu.get_device_properties(0)),
            'workflow_commit': os.environ.get('GITHUB_SHA'),
            'compiler_base': SCOPE['compiler_base'],
            'libtriton_sha256': sha(Path(triton.__file__).parent / '_C/libtriton.so'),
        })
        library = Path(triton.__file__).parent / '_C/libtriton.so'
    save(reports / 'compiler-manifest.json', {
        'base_commit': SCOPE['compiler_base'],
        'compilers': {policy: {'path': str(library), 'sha256': sha(library), 'mode': policy}
                      for policy in POLICIES},
    })


def prepare_case(reports, case):
    folder = reports / 'cases' / case['id']
    folder.mkdir(parents=True, exist_ok=True)
    families = {}
    for name, spec in SCOPE['families'].items():
        families[name] = dict(spec, source=str(REPO / spec['source']),
                              kernel_source=str(HERE / spec['kernel_source']))
    protocol = {key: SCOPE[key] for key in ('allocation_arrangements', 'selection_rounds',
                'confirmation_rounds', 'seed', 'atol', 'rtol', 'prune_factor')}
    protocol.update(families=families, policies=POLICIES,
                    paired_build_manifest=str(reports / 'compiler-manifest.json'),
                    target_source=str(reports / 'target.json'))
    save(folder / 'protocol.json', protocol)
    save(folder / 'suite.json', {'cases': [case]})
    shutil.copy2(reports / 'compiler-manifest.json', folder / 'compiler-manifest.json')
    shutil.copy2(HERE / 'stock_configs.py', folder / 'stock_configs.py')
    return folder


def command(folder, label, args, seconds):
    environment = dict(os.environ, GEMM_CI_REPORTS=str(folder), PYTHONUNBUFFERED='1')
    with (folder / (label + '.log')).open('w') as log:
        process = subprocess.Popen([sys.executable, *args], env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(f'{label} exceeded {seconds} seconds')
    if code:
        raise RuntimeError(f'{label} failed with exit code {code}; inspect {label}.log')


def image(record):
    return [(fingerprint(Path(kernel['folder']) / 'kernel.zebin'), kernel['launch'],
             kernel['runtime_signature'], kernel['pointer_names'], kernel['execution_metadata']) for kernel in record['kernels']]


def audit(folder, case):
    records = {policy: json.loads((folder / f'compiled-{policy}.json').read_text())
               for policy in POLICIES}
    if not all(record['complete'] for record in records.values()):
        raise RuntimeError('Incomplete compilation')
    configurations = {policy: record['cases'][case['id']]['configs']
                      for policy, record in records.items()}
    if len({tuple(values) for values in configurations.values()}) != 1:
        raise RuntimeError('Policy configuration pools differ')
    changed, unchanged, valid, unavailable, details = [], [], [], [], {}
    for index in configurations['baseline']:
        entries = {policy: values[index] for policy, values in configurations.items()}
        if len({json.dumps(entry['config'], sort_keys=True) for entry in entries.values()}) != 1:
            raise RuntimeError('Policy tuning parameters differ')
        if not require_matching_availability({policy: entry['status'] for policy, entry in entries.items()}):
            unavailable.append(index)
            continue
        images = {policy: image(entry) for policy, entry in entries.items()}
        if images['learned'] not in (images['baseline'], images['intervention']):
            raise RuntimeError('Learned native program is neither baseline nor forced')
        (changed if images['baseline'] != images['intervention'] else unchanged).append(index)
        valid.append(index)
        details[index] = {'config': entries['baseline']['config'],
                          'forced_changed_native_program': images['baseline'] != images['intervention'],
                          'learned_matches': [policy for policy in ('baseline', 'intervention')
                                              if images['learned'] == images[policy]]}
    save(folder / 'audit.json', {'cases': {case['id']: {
        'changed_configs': changed, 'unchanged_configs': unchanged,
        'common_compile_configs': valid, 'unavailable_in_all_policies': unavailable,
        'details': details}}})


def summarize(reports):
    records = []
    fixed = []
    for case in SCOPE['cases']:
        folder = reports / 'cases' / case['id']
        histories = []
        for skew in SCOPE['allocation_arrangements']:
            path = folder / f'skew-{skew}.json'
            if not path.exists():
                continue
            observation = json.loads(path.read_text())
            if not observation.get('complete'):
                continue
            values = {policy: confirmed_ms(observation, observation['selected'][policy])
                      for policy in [*POLICIES, 'oracle']}
            histories.append({'skew_mib': skew, 'latencies_ms': values,
                              'winners': observation['winners'], 'addresses': observation['addresses']})
            for index in observation['fixed_config_indices']:
                times = {policy: confirmed_ms(observation, f'{index}:{policy}') for policy in POLICIES}
                fixed.append({'case': case['id'], 'skew_mib': skew,
                              'config': index, 'latencies_ms': times})
        if histories:
            gain = {policy: 100 * (1 - math.exp(statistics.mean(
                math.log(h['latencies_ms'][policy] / h['latencies_ms']['baseline']) for h in histories)))
                for policy in [*POLICIES, 'oracle']}
            audit_path = folder / 'audit.json'
            audited = json.loads(audit_path.read_text())['cases'][case['id']] if audit_path.exists() else {}
            records.append({'case': case, 'histories': histories, 'latency_reduction_pct': gain,
                            'complete': len(histories) == len(SCOPE['allocation_arrangements']),
                            'changed_configs': audited.get('changed_configs', [])})
    save(reports / 'comparison.json', {'expected_cases': SCOPE['count'], 'records': records,
                                     'fixed_configurations': fixed})
    lines = ['# TD GEMM forced-policy opportunity study', '',
             'Positive percentages mean lower latency than baseline. Oracle selection uses tuning rounds; reported timings use separate confirmation rounds.', '',
             '| Case | Histories | Forced gain | Learned gain | Oracle gain | Changed configs |',
             '|---|---:|---:|---:|---:|---:|']
    for record in records:
        gain = record['latency_reduction_pct']
        lines.append(f'| {record["case"]["id"]} | {len(record["histories"])} | {gain["intervention"]:+.2f}% | {gain["learned"]:+.2f}% | {gain["oracle"]:+.2f}% | {len(record["changed_configs"])} |')
    lines.extend(['', f'Completed cases: {sum(record["complete"] for record in records)}/{SCOPE["count"]}.',
                  'Forced regressions are expected diagnostic observations. Compile asymmetries and numerical failures stop the experiment.'])
    (reports / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')


def run(reports):
    progress = {'complete': False, 'cases': {}, 'started': time.time()}
    for case in SCOPE['cases']:
        folder = prepare_case(reports, case)
        entry = progress['cases'][case['id']] = {'complete': False}
        save(reports / 'progress.json', progress)
        try:
            for policy in POLICIES:
                command(folder, 'compile-' + policy,
                        [str(HERE / 'compile.py'), policy, '--case', case['id'],
                         '--families', case['family'], '--cohorts', 'ci,local,anchors'], 600)
            audit(folder, case)
            winners = None
            for skew in SCOPE['allocation_arrangements']:
                args = [str(HERE / 'measure.py'), '--case', case['id'], '--skew', str(skew),
                        '--output', str(folder / f'skew-{skew}.json')]
                if winners is not None:
                    args.extend(['--winners', str(folder / 'winners.json')])
                command(folder, f'measure-{skew}', args, 300)
                observation = json.loads((folder / f'skew-{skew}.json').read_text())
                winners = observation['winners']
                save(folder / 'winners.json', winners)
            entry['complete'] = True
            print('COMPLETE', case['id'], flush=True)
        except Exception as error:
            entry['error'] = repr(error)
            save(reports / 'progress.json', progress)
            summarize(reports)
            raise
        save(reports / 'progress.json', progress)
        summarize(reports)
    progress['complete'] = True
    save(reports / 'progress.json', progress)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['describe', 'capture', 'run', 'summarize'])
    args = parser.parse_args()
    if args.command == 'describe':
        print(json.dumps({key: SCOPE[key] for key in ('count', 'cohort_counts', 'policies', 'limits')}, indent=2))
        return
    reports = Path(os.environ['GEMM_CI_REPORTS']).resolve()
    reports.mkdir(parents=True, exist_ok=True)
    {'capture': capture, 'run': run, 'summarize': summarize}[args.command](reports)


if __name__ == '__main__':
    main()
