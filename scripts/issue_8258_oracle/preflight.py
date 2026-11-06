"""CPU-only checks for diagnostic selection and the portable source bundle."""
import ast
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from driver import HERE, REPO, SCOPE, prepare_case, sha
from freeze_inputs import capture_provenance
from oracle import confirmed_ms, require_matching_availability, select_oracle


def configuration_pool(family):
    class Config:
        def __init__(self, kwargs, num_warps=4, num_stages=2):
            self.kwargs = kwargs
            self.num_warps = num_warps
            self.num_stages = num_stages
    namespace = {'triton': SimpleNamespace(Config=Config), 'List': list, 'DEVICE': 'xpu'}
    if family in ('plain', 'transpose_a'):
        tree = ast.parse((HERE / 'stock_configs.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<stock-configs>', 'exec'), namespace)
        return namespace['get_matmul_autotune_configs']()
    info = SCOPE['families'][family]
    if family in ('splitk', 'streamk'):
        tree = ast.parse((REPO / info['source']).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'get_autotune_configs')
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<stock-configs>', 'exec'), namespace)
        return namespace['get_autotune_configs']()
    return eval(info['configs_expression'], namespace)


def check_measurement_startup():
    class GPUBoundaryReached(Exception):
        pass

    previous_env = os.environ.get('GEMM_CI_REPORTS')
    previous_argv = sys.argv
    previous_compile = sys.modules.get('compile')
    try:
        with tempfile.TemporaryDirectory() as temporary:
            reports = Path(temporary)
            (reports / 'compiler-manifest.json').write_text('{}')
            folder = prepare_case(reports, SCOPE['cases'][0])
            (folder / 'audit.json').write_text('{}')
            for policy in SCOPE['policies']:
                (folder / f'compiled-{policy}.json').write_text(json.dumps({
                    'complete': True, 'protocol_sha256': sha(folder / 'protocol.json')}))
            assert not (folder / 'run.py').exists()
            assert not (folder / 'measure.py').exists()
            os.environ['GEMM_CI_REPORTS'] = str(folder)
            spec = importlib.util.spec_from_file_location('oracle_measure_startup', HERE / 'measure.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            captured = {}

            def capture(root, runner):
                captured.update(capture_provenance(root, runner))
                return dict(captured)

            @contextmanager
            def gpu_boundary(path):
                assert set(captured) == {'protocol.json', 'suite.json',
                                         'compiler-manifest.json', 'audit.json', 'measure.py'}
                assert captured['measure.py'] == sha(HERE / 'measure.py')
                raise GPUBoundaryReached
                yield

            module.capture_provenance = capture
            module.acquire = gpu_boundary
            sys.argv = ['measure.py', '--case', SCOPE['cases'][0]['id'], '--skew', '0',
                        '--output', str(folder / 'skew-0.json')]
            try:
                module.main()
            except GPUBoundaryReached:
                pass
            else:
                raise AssertionError('Measurement startup did not reach the GPU boundary')
    finally:
        sys.argv = previous_argv
        if previous_env is None:
            os.environ.pop('GEMM_CI_REPORTS', None)
        else:
            os.environ['GEMM_CI_REPORTS'] = previous_env
        if previous_compile is None:
            sys.modules.pop('compile', None)
        else:
            sys.modules['compile'] = previous_compile


def main():
    for path in HERE.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    assert SCOPE['count'] == len(SCOPE['cases']) == 180
    assert len({case['id'] for case in SCOPE['cases']}) == 180
    for family, info in SCOPE['families'].items():
        for field, parent in [('source', REPO), ('kernel_source', HERE)]:
            assert sha(parent / info[field]) == info[field + '_sha256']
        tree = ast.parse((REPO / info['source']).read_text())
        source_functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        extracted = ast.parse((HERE / info['kernel_source']).read_text())
        extracted_functions = {node.name: node for node in extracted.body if isinstance(node, ast.FunctionDef)}
        for name in info['functions']:
            source_functions[name].decorator_list = []
            extracted_functions[name].decorator_list = []
            assert ast.dump(source_functions[name]) == ast.dump(extracted_functions[name])
    counts = {family: len(configuration_pool(family)) for family in SCOPE['families']}
    assert counts == {'add': 5, 'exp': 5, 'gelu': 5, 'plain': 16,
                      'splitk': 1, 'streamk': 1, 'transpose_a': 16}, counts
    constants = {'M', 'N', 'K', 'stride_am', 'stride_ak', 'stride_bk', 'stride_bn',
                 'stride_cm', 'stride_cn', 'stride_dm', 'stride_dn', 'transpose_a',
                 'transpose_b', 'ACCUMULATOR_DTYPE', 'acc_dtype'}
    for family, info in SCOPE['families'].items():
        tree = ast.parse((HERE / info['kernel_source']).read_text())
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        entries = info['functions'][-2:] if family == 'streamk' else [info['functions'][-1]]
        for config in configuration_pool(family):
            for name in entries:
                required = {arg.arg for arg in functions[name].args.args
                            if arg.annotation and ast.unparse(arg.annotation) == 'tl.constexpr'}
                assert not required - constants - set(config.kwargs), (family, name)
    assert require_matching_availability({'baseline': 'compiled', 'intervention': 'compiled'})
    assert not require_matching_availability({'baseline': 'failed', 'intervention': 'failed'})
    try:
        require_matching_availability({'baseline': 'compiled', 'intervention': 'failed'})
    except RuntimeError:
        pass
    else:
        raise AssertionError('Asymmetric compilation must not shrink the baseline pool')
    mapping = {f'{index}:{policy}': f'{index}:{policy}' for index in (0, 1)
               for policy in ('baseline', 'intervention')}
    timings = {'0:baseline': [[10]], '0:intervention': [[20]],
               '1:baseline': [[12]], '1:intervention': [[5]]}
    assert select_oracle(['0', '1'], timings, mapping) == '1:intervention'
    timings['1:intervention'] = [[30]]
    assert select_oracle(['0', '1'], timings, mapping) == '0:baseline'
    selected = select_oracle(['0', '1'], timings, mapping)
    observation = {'mapping': mapping, 'confirmation_ms': {key: [[value]] for key, value in
                   [('0:baseline', 11), ('0:intervention', 1), ('1:baseline', 12), ('1:intervention', 30)]}}
    assert confirmed_ms(observation, selected) == 11
    check_measurement_startup()
    print(json.dumps({'cases': 180, 'configurations_per_family': counts,
                      'configuration_policy_compilations': sum(counts[case['family']] * 3 for case in SCOPE['cases']),
                      'checks': ['Python syntax', 'exact kernel bodies', 'source hashes', 'full stock pools',
                                 'all compile-time arguments supplied',
                                 'measurement CLI startup and provenance before GPU initialization',
                                 'asymmetric compile rejection', 'oracle gain', 'oracle baseline fallback',
                                 'independent confirmation selection']}, indent=2))


if __name__ == '__main__':
    main()
