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
    parser.add_argument('--tag', help='Tag for the benchmark run', default='')
    parser.add_argument('--benchmark', help='moe-benchmark', default='')

    return parser.parse_args()


def parse_moe_csv(csv_file_path, tag, benchmark):
    """Parse the MoE benchmark CSV and extract performance metrics."""

    df = pd.read_csv(csv_file_path)

    run_uuid = uuid.uuid4().hex
    current_datetime = datetime.now().isoformat()

    # Create params for all rows vectorized
    df['params'] = df.apply(
        lambda row: json.dumps({
            'num_experts': int(row['num_experts']),
            'max_tokens_per_expert': int(row['max_tokens_per_expert']),
            'K': int(row['K']),
            'N': int(row['N']),
        }), axis=1)

    # Define compiler columns
    compilers = [('triton', 'triton-TFlops'), ('pytorch', 'pytorch-TFlops'), ('triton-td', 'triton-td-TFlops')]

    # Create list of dataframes for each compiler
    dfs = []
    for compiler_name, tflops_col in compilers:
        if tflops_col in df.columns:
            # Filter out NaN values
            valid_rows = df[df[tflops_col].notna()].copy()
            if len(valid_rows) > 0:
                valid_rows['run_uuid'] = run_uuid
                valid_rows['ts'] = current_datetime
                valid_rows['benchmark_group'] = 'moe-benchmark'
                valid_rows['benchmark'] = benchmark
                valid_rows['compiler'] = compiler_name
                valid_rows['value_name'] = 'tflops'
                valid_rows['value'] = valid_rows[tflops_col].astype(float)
                valid_rows['tag'] = tag

                # Select only needed columns
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

    df_results = parse_moe_csv(args.source, args.tag, args.benchmark)
    df_results.to_csv(args.target, index=False)


if __name__ == '__main__':
    main()
