import argparse
import os
import uuid
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Parse compilation report folders into a CSV')
    parser.add_argument('source',
                        help='Path to the root compilation report folder (contains per-benchmark subdirectories)')
    parser.add_argument('target', help='Path to output CSV file')
    parser.add_argument('--tag', help='Tag for the benchmark run', default='')
    parser.add_argument('--benchmark_group', help='Benchmark group name, e.g. triton-benchmarks-pvc',
                        default='triton-benchmarks')
    return parser.parse_args()


def split_kernel_name_and_params(folder_name: str) -> tuple[str, str]:
    """Split a folder name into kernel name and params.

    E.g. 'scan_kernel_32_16_1' -> ('scan_kernel', '32_16_1')
         'matmul_kernel_with_tensor_descriptors_4_256_256_...' -> ('matmul_kernel_with_tensor_descriptors', '4_256_256_...')
    """
    parts = folder_name.split('_')
    for i, part in enumerate(parts):
        if part.lstrip('-').isnumeric() or part in ('True', 'False'):
            return '_'.join(parts[:i]), '_'.join(parts[i:])
    return folder_name, ''


def parse_results_rec(data: dict | float, comp_uuid: str, name: str, parent: str, acc):
    # check for leaf
    sample = {
        'comp_uuid': comp_uuid,
        'name': name,
        'full_name': f"{parent}/{name}" if parent else name,
        'parent': parent,
    }
    if isinstance(data, (int, float)):
        acc.append({**sample, 'time_s': float(data)})
        return

    for k, v in data.items():
        if k in ('time', 'Total'):
            acc.append({**sample, 'time_s': float(v)})
            continue
        if k == 'asm':
            continue
        k = k.replace('/', '_')
        parse_results_rec(v, comp_uuid, name=k, parent=sample['full_name'], acc=acc)
    return


def parse_flat_results(data: dict, comp_uuid: str) -> list[dict]:
    """Json contains hierarchical data about passes runtime, we parse it into flat
    structure that shows parent relationships for each row to allow easier queries based on prefixes
    """
    acc: list[dict] = []
    parse_results_rec(data, comp_uuid, name='root', parent='', acc=acc)
    return acc


def prepare_benchmark_reports(bench_dir, ts, tag, benchmark_group) -> tuple[list[dict], list[dict]]:  # pylint: disable=too-many-locals
    benchmark_uuid = uuid.uuid4().hex

    benchmark = bench_dir.name

    passes_rows: list[dict] = []
    kernel_rows: list[dict] = []

    for kernel_folder in sorted(bench_dir.iterdir()):
        if not kernel_folder.is_dir():
            continue
        kernel_uuid = uuid.uuid4().hex
        json_files = sorted(kernel_folder.glob('JITFunction._do_compile*.json'))
        if not json_files:
            print(f'Benchmark folder {kernel_folder} contains no JITFunction._do_compile*.json files, skipping')
            continue

        kernel_name, params = split_kernel_name_and_params(kernel_folder.name)

        for json_file in json_files:
            comp_uuid = uuid.uuid4().hex

            with open(json_file, encoding='utf-8') as f:
                data = json.load(f)

            loc = data['asm']

            kernel_rows.append({
                # Same for all kernels in the benchmark
                'ts': ts,
                'tag': tag,
                'benchmark_group': benchmark_group,
                'benchmark': benchmark,
                'benchmark_uuid': benchmark_uuid,
                # Kernel-specific
                'kernel_uuid': kernel_uuid,
                'comp_uuid': comp_uuid,
                'kernel_name': kernel_name,
                'params': params,
                # Aggregated results
                'time_s': data['time'],
                'loc_source': loc['source'],
                'loc_ttir': loc['ttir'],
                'loc_ttgir': loc['ttgir'],
                'loc_llir': loc['llir'],
                'loc_spv': loc['spv'],
            })

            passes_rows.extend(parse_flat_results(data, comp_uuid))
    return kernel_rows, passes_rows


def parse_reports(source_dir: Path, tag: str, benchmark_group: str) -> pd.DataFrame:
    ts = datetime.now().isoformat()

    kernel_rows: list[dict] = []
    passes_rows: list[dict] = []
    for bench_dir in sorted(source_dir.iterdir()):
        if not bench_dir.is_dir():
            continue
        bench_kernel_rows, bench_passes_rows = prepare_benchmark_reports(bench_dir, ts=ts, tag=tag,
                                                                         benchmark_group=benchmark_group)
        kernel_rows.extend(bench_kernel_rows)
        passes_rows.extend(bench_passes_rows)

    return pd.DataFrame.from_records(kernel_rows), pd.DataFrame.from_records(passes_rows)


def main():
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise ValueError(f'Error: source folder {source} not found')

    host_info = {
        n: os.getenv(n.upper(), default='')
        for n in [
            'libigc1_version',
            'level_zero_version',
            'gpu_device',
            'agama_version',
            'torch_version',
            'compiler_version',
        ]
    }
    if not host_info['gpu_device']:
        raise RuntimeError('Could not find GPU device description, was `capture-hw-details.sh` called?')

    kernel_df, passes_df = parse_reports(source, tag=args.tag, benchmark_group=args.benchmark_group)

    for col, val in host_info.items():
        kernel_df[col] = val

    print('Kernels shape', kernel_df.shape)
    print('Passes shape', passes_df.shape)

    # mkdir and save both csvs
    target_dir = Path(args.target)
    os.makedirs(target_dir, exist_ok=True)
    kernel_df.to_csv(target_dir / 'kernels-compile-stats.csv', index=False)
    passes_df.to_csv(target_dir / 'passes-compile-stats.csv', index=False)


if __name__ == '__main__':
    main()
