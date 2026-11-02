"""Authorized, exclusive-device candidate selection and held-out confirmation."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import traceback
from types import SimpleNamespace

ROOT = Path(os.environ['GEMM_CI_REPORTS']).resolve()
SCAN = r'''
import json, os
from pathlib import Path
hits = []
for process in Path('/proc').glob('[0-9]*'):
    try:
        for fd in (process/'fd').iterdir():
            target = os.readlink(fd)
            if target.startswith('/dev/dri/'):
                hits.append({'pid': int(process.name), 'device': target,
                             'command': (process/'comm').read_text().strip()})
    except PermissionError:
        raise RuntimeError('Root-visible GPU ownership scan was incomplete')
    except (FileNotFoundError, ProcessLookupError):
        pass
print(json.dumps(hits))
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def owners():
    return json.loads(subprocess.check_output(['sudo', '-n', 'python3', '-c', SCAN], text=True))


def median_rounds(rounds):
    return statistics.median(statistics.median(r) for r in rounds)


def select_winners(valid, timings, mapping, policies):
    if not valid:
        raise RuntimeError('No common correctness-valid configuration')
    return {policy: min(valid, key=lambda index: (median_rounds(timings[mapping[f'{index}:{policy}']]), int(index)))
            for policy in policies}


def allocation(torch, prepared, protocol, shape, skew, output, winners=None):
    from triton.compiler.compiler import CompiledKernel
    torch.manual_seed(protocol['seed'])
    batch, m, n, k = ([1] + shape) if len(shape) == 3 else shape
    shape_key = f'{m}x{n}x{k}' if batch == 1 else f'B{batch}_{m}x{n}x{k}'
    candidates = prepared['cases'][shape_key]
    guard = torch.zeros(skew * 1024 * 1024, device='xpu', dtype=torch.uint8)
    a = torch.rand((batch, m, k), device='xpu', dtype=torch.bfloat16)
    bt = protocol.get('transpose_b', False)
    b_storage = torch.rand((batch, n, k) if bt else (batch, k, n), device='xpu', dtype=torch.bfloat16)
    b = b_storage.transpose(1, 2) if bt else b_storage
    c = torch.empty((batch, m, n), device='xpu', dtype=torch.float32)
    reference = torch.bmm(a, b, out_dtype=torch.float32)
    rows = torch.linspace(0, m-1, 16, device='xpu').long()
    cols = torch.linspace(0, n-1, 16, device='xpu').long()
    sampled_a, sampled_b = a[:, rows, :].cpu().double(), b[:, :, cols].cpu().double()
    cpu_reference = sampled_a @ sampled_b
    flush = torch.empty(256 * 1024 * 1024 // 4, device='xpu', dtype=torch.int32)
    rng = random.Random(protocol['seed'] + skew)
    result = {'shape': shape, 'skew_mib': skew, 'addresses': {'guard': guard.data_ptr(),
              'A': a.data_ptr(), 'B': b.data_ptr(), 'C': c.data_ptr()}, 'candidates': {},
              'mapping': {}, 'valid_configs': [], 'selection_ms': {}, 'selection_orders': [],
              'confirmation_ms': {}, 'confirmation_orders': [], 'winners': winners,
              'input_sample_sha256': sha(sampled_a.numpy().tobytes() + sampled_b.numpy().tobytes())}
    objects, functions, binaries = {}, {}, {}
    mapping = result['mapping']
    policies = protocol['policies']
    if winners is None:
        indices = list(candidates)
    else:
        indices = sorted(set(winners.values()), key=int)

    def persist():
        save(output, result)

    def validate(key):
        c.fill_(float('nan'))
        functions[key]()
        torch.xpu.synchronize()
        for start in range(0, m, 128):
            torch.testing.assert_close(c[:, start:start+128], reference[:, start:start+128],
                                       atol=protocol['atol'], rtol=protocol['rtol'])
        actual = c[:, rows[:, None], cols[None, :]].cpu().double()
        torch.testing.assert_close(actual, cpu_reference, atol=protocol['atol'], rtol=protocol['rtol'])
        return {'full_output_close': True, 'cpu_fp64_sample_max_abs_error': (actual-cpu_reference).abs().max().item()}

    for index in indices:
        item = candidates[index]
        entry = result['candidates'][index] = {'config': item['config'], 'variants': {}}
        if item['status'] != 'compiled':
            entry.update(status='compile_failed', error=item.get('error'))
            persist()
            continue
        good = True
        for policy in policies:
            identifier = f'{index}:{policy}'
            variant = item['variants'][policy]
            if 'alias' in variant:
                variant = item['variants'][variant['alias']]
            binary_sha = variant['sha256']
            if binary_sha in binaries:
                mapping[identifier] = binaries[binary_sha]
                entry['variants'][policy] = {'alias': binaries[binary_sha]}
                continue
            mapping[identifier] = identifier
            folder = Path(variant['folder'])
            try:
                if sha((folder / 'kernel.zebin').read_bytes()) != binary_sha:
                    raise RuntimeError('Native image hash mismatch')
                source = SimpleNamespace(constants={}, signature={0: '*bf16', 1: '*bf16', 2: '*fp32'})
                compiled = CompiledKernel(source, {f'kernel.{ext}': str(folder/f'kernel.{ext}')
                                                  for ext in ('json', 'zebin')}, binary_sha)
                config = item['config']
                grid = (math.ceil(m/config['BLOCK_SIZE_M']) * math.ceil(n/config['BLOCK_SIZE_N']), batch, 1)
                launch = compiled[grid]
                objects[identifier] = compiled
                functions[identifier] = lambda launch=launch: launch(a, b, c)
                checked = validate(identifier)
                functions[identifier]()
                torch.xpu.synchronize()
                entry['variants'][policy] = {'status': 'valid', 'before': checked, 'sha256': binary_sha,
                                             'n_regs': compiled.n_regs, 'n_spills': compiled.n_spills}
                binaries[binary_sha] = identifier
            except Exception as error:
                good = False
                entry['variants'][policy] = {'status': 'invalid', 'error': repr(error)}
                print('INVALID', shape_key, index, policy, repr(error)[:400], flush=True)
        entry['status'] = 'valid' if good else 'excluded_common_pool'
        if good:
            result['valid_configs'].append(index)
        persist()
        print('VALIDATION', shape_key, skew, index, entry['status'], flush=True)

    warm = min(functions, key=lambda key: int(key.split(':')[0]))
    warm_start, warm_end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    warm_start.record(); functions[warm](); warm_end.record(); warm_end.synchronize()
    warm_ms = warm_start.elapsed_time(warm_end)
    warm_repeats = max(1, min(4096, math.ceil(protocol.get('compute_warmup_ms', 50) / max(warm_ms, .001))))
    result['compute_warmup'] = {'kernel': warm, 'estimate_ms': warm_ms, 'repeats': warm_repeats, 'frequency': []}

    def measure(keys, rounds, launches, field):
        result[field + '_ms'] = {key: [] for key in keys}
        for rnd in range(rounds):
            for _ in range(warm_repeats): functions[warm]()
            torch.xpu.synchronize()
            try: freq = Path('/sys/class/drm/card0/device/tile0/gt0/freq0/act_freq').read_text().strip()
            except OSError: freq = None
            result['compute_warmup']['frequency'].append({'phase': field, 'round': rnd, 'mhz': freq})
            order = list(keys)
            rng.shuffle(order)
            result[field + '_orders'].append(order)
            for key in order:
                if field == 'probe':
                    times = []
                    for _ in range(launches):
                        flush.zero_(); torch.xpu.synchronize()
                        begin, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
                        begin.record(); functions[key](); end.record(); end.synchronize()
                        times.append(begin.elapsed_time(end))
                else:
                    # Use exactly the default XPU autotuner benchmark implementation.
                    from triton.testing import do_bench
                    times = do_bench(functions[key], return_mode='all')
                if not times or any(not math.isfinite(t) or t <= 0 for t in times):
                    raise RuntimeError('Invalid timing from stock benchmark')
                result[field + '_ms'][key].append(times)
            foreign = [o for o in owners() if o['pid'] != os.getpid()]
            if foreign:
                raise RuntimeError('Foreign GPU owner during timing: ' + repr(foreign))
            persist()
            print(field.upper(), shape_key, skew, rnd + 1, flush=True)

    valid = result['valid_configs']
    if winners is None:
        keys = sorted({mapping[f'{index}:{policy}'] for index in valid for policy in policies})
        result['probe_orders'] = []
        measure(keys, 1, 2, 'probe')
        best = min(median_rounds(result['probe_ms'][key]) for key in keys)
        retained = [index for index in valid if any(median_rounds(result['probe_ms'][mapping[f'{index}:{policy}']]) <= protocol['prune_factor'] * best for policy in policies)]
        result['pruning'] = {'factor': protocol['prune_factor'], 'fastest_probe_ms': best, 'retained_common_configs': retained,
                             'excluded_common_configs': [i for i in valid if i not in retained]}
        keys = sorted({mapping[f'{index}:{policy}'] for index in retained for policy in policies})
        measure(keys, protocol['selection_rounds'], protocol['selection_launches_per_round'], 'selection')
        winners = select_winners(retained, result['selection_ms'], mapping, policies)
        result['winners'] = winners
        print('WINNERS', shape_key, winners, flush=True)
        persist()
    elif any(index not in valid for index in winners.values()):
        raise RuntimeError('A selected configuration failed on the second allocation')
    controls = {policy: f"{winners['baseline']}:{policy}" for policy in policies}
    selected = {policy: f'{winners[policy]}:{policy}' for policy in policies}
    result['selected'] = selected
    result['baseline_config_controls'] = controls
    keys = sorted({mapping[key] for key in [*controls.values(), *selected.values()]})
    measure(keys, protocol['confirmation_rounds'], protocol['confirmation_launches_per_round'], 'confirmation')
    for key in keys:
        result.setdefault('after_validation', {})[key] = validate(key)
    torch.testing.assert_close(a[:, rows, :].cpu().double(), sampled_a, rtol=0, atol=0)
    torch.testing.assert_close(b[:, :, cols].cpu().double(), sampled_b, rtol=0, atol=0)
    result.update(complete=True, unchanged_input_samples=True)
    persist()
    torch.xpu.synchronize()
    return result


