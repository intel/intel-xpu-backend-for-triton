"""Read live benchmark inputs and optionally split them without splitting profiles."""
import argparse
import hashlib
import json
import os
from pathlib import Path

DTYPES = ('bfloat16', 'float8_e4m3fn')


def benchmark_workloads():
    # Generate both complete lists regardless of the caller's benchmark shortcuts.
    os.environ['FP8'] = '0'
    os.environ['DEBUG_BENCH'] = '0'
    from triton_kernels_benchmark.vllm.unified_attention import unified_attention_benchmark as benchmark  # pylint: disable=import-outside-toplevel

    manifests = {}
    for dtype, configs in zip(DTYPES, (benchmark.ATTENTION_CONFIGS_BF16, benchmark.ATTENTION_CONFIGS_FP8)):
        cases = []
        for index, config in enumerate(configs):
            q_heads, kv_heads, head_size, _, seq_lens, window, softcap, num_blocks, block_size = config
            cases.append({
                'id': f'{dtype}-{index:04d}',
                'batch': len(seq_lens),
                'block_size': block_size,
                'dtype': dtype,
                'head_size': head_size,
                'kv_heads': kv_heads,
                'kv_layout': 'contiguous',
                'kv_lens': [kv for _, kv in seq_lens],
                'num_blocks': num_blocks,
                'num_segments': 16,
                'out_dtype': 'bfloat16',
                'q_heads': q_heads,
                'q_layout': 'contiguous',
                'query_lens': [q for q, _ in seq_lens],
                'seq_threshold_3d': 32,
                'sliding_window': window or 0,
                'softcap': softcap or 0.0,
            })
        manifests[dtype] = cases
    return manifests


def profile_group(case):
    # CI uses TD and unquantized/native-FP8 caches for every input.
    return (case['block_size'], case['head_size'], case['q_heads'] // case['kv_heads'], case['dtype'],
            case['out_dtype'], case['q_layout'], case['kv_layout'], bool(case['softcap']), case['sliding_window'])


def partition(manifests):
    result = {part: {dtype: [] for dtype in DTYPES} for part in ('1', '2')}
    for dtype in DTYPES:
        for case in manifests[dtype]:
            # Ownership must stay fixed when runners filter out different workloads.
            digest = hashlib.sha256(json.dumps(profile_group(case)).encode()).hexdigest()
            part = str(1 + int(digest[:8], 16) % 2)
            result[part][dtype].append(case)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('1', '2'), help='Omit to select all benchmark inputs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifests = benchmark_workloads()
    parts = partition(manifests)
    selected = parts[args.part] if args.part else manifests
    args.output.mkdir(parents=True, exist_ok=True)
    assignment = {part: [case['id'] for dtype in DTYPES for case in cases[dtype]] for part, cases in parts.items()}
    (args.output / 'parts.json').write_text(json.dumps(assignment, indent=2) + '\n')
    for dtype, cases in selected.items():
        (args.output / f'{dtype}.json').write_text(json.dumps(cases, indent=2) + '\n')
        print(f"{('Part ' + args.part) if args.part else 'All'}: {len(cases)} {dtype} inputs", flush=True)


if __name__ == '__main__':
    main()
