"""Run the offline tuner with the exact CI tensor fixture."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / 'benchmarks/triton_kernels_benchmark/vllm/unified_attention'
spec = importlib.util.spec_from_file_location('ua_tuner', BENCHMARK / 'tune_unified_attention.py')
tuner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuner)
tree = ast.parse((BENCHMARK / 'unified_attention_benchmark.py').read_text())
node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'benchmark')
node.decorator_list = []
helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('ref_paged_attn', '_dtype_size', 'is_enough_memory')]
namespace = {'torch': torch, 'BENCHMARKING_CONFIG': {'verify': False}, 'is_td_patched': True}
exec(compile(ast.Module(body=helpers + [node], type_ignores=[]), 'ci-fixture', 'exec'), namespace)


class Captured(Exception):
    pass


def capture(**kwargs):
    raise Captured(kwargs)


def allocate_ci(case, device, seed):
    # CI initializes its own fixed seed and creates the exact input strides.
    namespace.update(unified_attention=capture,
                     benchmark_suite=SimpleNamespace(assert_close=lambda fn, *a, **kw: fn()))
    try:
        namespace['benchmark'](
            q_heads=case['q_heads'], k_heads=case['kv_heads'], head_size=case['head_size'],
            qdtype=getattr(torch, case['dtype']) if case['dtype'].startswith('float8') else None,
            seq_lens=list(zip(case['query_lens'], case['kv_lens'])),
            sliding_window=case['sliding_window'] or None, soft_cap=case['softcap'] or None,
            num_blocks=case['num_blocks'], block_size=case['block_size'], provider='triton-td')
    except Captured as result:
        return result.args[0]
    raise RuntimeError('CI fixture did not reach the attention call')


if __name__ == '__main__':
    tuner.allocate_case = allocate_ci
    tuner.main()
