"""Read live benchmark inputs and optionally split them without splitting profiles."""
import argparse
import hashlib
import json
import os
from pathlib import Path

DTYPES = ('bfloat16', 'float8_e4m3fn')
PARTS = ('1', '2', '3', '4')


def benchmark_workloads():
    # Generate complete BF16 and FP8 workload lists.
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
                'q_heads': q_heads,
                'k_heads': kv_heads,
                'head_size': head_size,
                'qdtype': None if dtype == 'bfloat16' else dtype,
                'seq_lens': seq_lens,
                'sliding_window': window,
                'soft_cap': softcap,
                'num_blocks': num_blocks,
                'block_size': block_size,
            })
        manifests[dtype] = cases
    return manifests


def profile_group(case):
    dtype = case['qdtype'] or 'bfloat16'
    return (case['block_size'], case['head_size'], case['q_heads'] // case['k_heads'], dtype, 'bfloat16',
            bool(case['soft_cap']), case['sliding_window'] or 0)


def partition(manifests):
    result = {part: {dtype: [] for dtype in DTYPES} for part in PARTS}
    for dtype in DTYPES:
        for case in manifests[dtype]:
            # The same profile group always produces the same digest, so its workloads stay together in the same part
            digest = hashlib.sha256(json.dumps(profile_group(case)).encode()).hexdigest()
            part = PARTS[int(digest[:8], 16) % len(PARTS)]
            result[part][dtype].append(case)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='xpu:0', help='XPU used for benchmark memory filtering')
    parser.add_argument('--part', choices=PARTS, help='Omit to select all benchmark inputs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    import torch  # pylint: disable=import-outside-toplevel

    device = torch.device(args.device)
    torch.xpu.set_device(device)
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
