import argparse
import os
import uuid
import json
from datetime import datetime

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Parse MoE benchmark CSV')
    parser.add_argument('source', help='Path to the MoE benchmark CSV file')
    parser.add_argument('target', help='Path to output CSV file')
    parser.add_argument(
        '--param_cols',
        help='Names of parameter columns, separated by commas.',
        required=True,
    )
    parser.add_argument('--tag', help='Tag for the benchmark run', default='')
    parser.add_argument('--benchmark', help='moe-benchmark', required=True)
    parser.add_argument('--bgroup', help='Benchmark group', required=True)

    return parser.parse_args()


def parse_csv(csv_file_path, tag, bench_group, benchmark, param_cols):
    """Parse the benchmark CSV and extract performance metrics."""

    df = pd.read_csv(csv_file_path)

    run_uuid = uuid.uuid4().hex
    current_datetime = datetime.now().isoformat()

    def serialize_params(row):
        param2val = {}
        for p in param_cols:
            try:
                param2val[p] = int(row[p])
            except ValueError:
                param2val[p] = str(row[p])
        return json.dumps(param2val)

    df['params'] = df.apply(serialize_params, axis=1)

    compilers = ['pytorch', 'triton', 'triton-td']

    dfs = []
    for compiler_name in compilers:
        for value_name in ['TFlops', 'GB/s']:
            col = f'{compiler_name}-{value_name}'
            if col not in df.columns:
                continue
            # Filter out NaN values
            valid_rows = df[df[col].notna()].copy()
            if len(valid_rows) > 0:
                valid_rows['run_uuid'] = run_uuid
                valid_rows['ts'] = current_datetime
                valid_rows['benchmark_group'] = bench_group
                valid_rows['benchmark'] = benchmark
                valid_rows['compiler'] = compiler_name
                # GB/s -> gbps
                valid_rows['value_name'] = value_name.lower().replace('/', 'p')
                valid_rows['value'] = valid_rows[col].astype(float)
                valid_rows['tag'] = tag

                result_df = valid_rows[[
                    'run_uuid', 'ts', 'benchmark_group', 'benchmark', 'compiler', 'value_name', 'value', 'params', 'tag'
                ]]
                dfs.append(result_df)

    # Concatenate all compiler results
    df_results = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    host_info = {
        n: os.getenv(n.upper(), default='')
        for n in [
            'libigc1_version',
            'level_zero_version',
            'gpu_device',
            'agama_version',
            'torch_version',
            'compiler_version',
            'benchmarking_method',
        ]
    }
    if not host_info['gpu_device']:
        raise RuntimeError('Could not find GPU device description, was `capture-hw-details.sh` called?')

    for name, val in host_info.items():
        df_results[name] = val

    print(f'DataFrame shape: {df_results.shape}')

    return df_results


def main():
    args = parse_args()
    if not os.path.exists(args.source):
        raise ValueError(f'Error: CSV file {args.source} not found')

    param_cols = args.param_cols.split(',')
    df_results = parse_csv(args.source, args.tag, args.bgroup, args.benchmark, param_cols)
    df_results.to_csv(args.target, index=False)


if __name__ == '__main__':
    main()
