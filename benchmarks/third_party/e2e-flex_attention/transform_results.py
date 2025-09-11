import argparse
import re
import os
import uuid
import json
from datetime import datetime

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Parse LLM profiling log')
    parser.add_argument('log_file', help='Path to the LLM profiling log file')
    parser.add_argument('output_csv', help='Path to output CSV file')
    parser.add_argument('--tag', help='Tag for the benchmark run', default='')
    parser.add_argument('--model', help='Model name', default='unknown-model')
    parser.add_argument('--max-new-tokens', type=int, help='Maximum new tokens', default=128)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=1)

    return parser.parse_args()


def parse_llm_log(log_file_path, tag, model, max_new_tokens, batch_size):
    """Parse the LLM profiling log and extract performance metrics."""

    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics = {}

    inference_match = re.search(r'inference-latency:\s+([\d.]+)\s+sec\.', content)
    if inference_match:
        metrics['inference_latency'] = float(inference_match.group(1))

    first_token_match = re.search(r'first-token-latency:\s+([\d.]+)\s+sec\.', content)
    if first_token_match:
        metrics['first_token_latency'] = float(first_token_match.group(1))

    rest_token_match = re.search(r'rest-token-latency:\s+([\d.]+)\s+sec\.', content)
    if rest_token_match:
        metrics['rest_token_latency'] = float(rest_token_match.group(1))

    p90_match = re.search(r'P90-rest-token-latency:\s+([\d.]+)\s+sec\.', content)
    if p90_match:
        metrics['p90_rest_token_latency'] = float(p90_match.group(1))

    prompt_match = re.search(r'Prompt size:\s+(\d+)', content)
    prompt_size = int(prompt_match.group(1)) if prompt_match else 1024

    params = {
        'model': model,
        'input_tokens': prompt_size,
        'max_new_tokens': max_new_tokens,
        'batch_size': batch_size,
    }
    params_json = json.dumps(params)

    rows = []
    run_uuid = uuid.uuid4().hex
    current_datetime = datetime.now().isoformat()

    # Create one row for each metric
    for metric_name, metric_value in metrics.items():
        row = {
            'run_uuid': run_uuid,
            'ts': current_datetime,
            'benchmark_group': 'e2e-flex-attention',
            'benchmark': 'e2e-flex-attention',
            'compiler': 'triton',
            'value_name': metric_name + '_s',
            'value': metric_value,
            'params': params_json,
            'tag': tag,
        }
        rows.append(row)

    df_results = pd.DataFrame(rows)

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

    print(f'Extracted metrics: {json.dumps(metrics, indent=2)}')
    print(f'DataFrame shape: {df_results.shape}')

    return df_results


def main():
    args = parse_args()
    if not os.path.exists(args.log_file):
        print(f'Error: Log file {args.log_file} not found')
        return 1

    df_results = parse_llm_log(args.log_file, args.tag, args.model, args.max_new_tokens, args.batch_size)
    df_results.to_csv(args.output_csv, index=False)
    print(f'Transformed CSV saved to {args.output_csv}')
    return 0


if __name__ == '__main__':
    main()
