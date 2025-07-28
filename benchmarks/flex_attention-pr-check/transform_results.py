import argparse
import re
import os
import uuid
import json
from datetime import datetime

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Parse Llama 3.1 profiling log')
    parser.add_argument('log_file', help='Path to the llama profiling log file')
    parser.add_argument('output_csv', help='Path to output CSV file')
    parser.add_argument("--tag", help="Tag for the benchmark run", default="")

    return parser.parse_args()

def parse_llama_log(log_file_path, output_csv_path, tag):
    """Parse the Llama profiling log and extract performance metrics."""
    
    with open(log_file_path, 'r') as f:
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

    # Create DataFrame with results
    df_results = pd.DataFrame()
    
    # Add the metrics as DataFrame columns
    df_results = pd.DataFrame([{
        'benchmark': 'flex_attention-pr-check',
        'model': 'meta-llama/Llama-3.1-8B',
        'run_uuid': uuid.uuid4().hex,
        'datetime': datetime.now().isoformat(),
        'tag': tag,
        'input_tokens': prompt_size,
        'max_new_tokens': 128,  # From the script parameters
        'batch_size': 1,  # Default from script
        'inference_latency': metrics.get('inference_latency', 0),
        'first_token_latency': metrics.get('first_token_latency', 0),
        'rest_token_latency': metrics.get('rest_token_latency', 0),
        'p90_rest_token_latency': metrics.get('p90_rest_token_latency', 0),
    }])
    
    host_info = {
        n: os.getenv(n.upper(), default="")
        for n in [
            "libigc1_version",
            "level_zero_version",
            "gpu_device",
            "agama_version",
            "torch_version",
            "compiler_version",
            "benchmarking_method",
        ]
    }
    if not host_info["gpu_device"]:
        raise RuntimeError("Could not find GPU device description, was `capture-hw-details.sh` called?")
    
    for name, val in host_info.items():
        df_results[name] = val
    
    df_results.to_csv(output_csv_path, index=False)
    
    print(f"Successfully parsed log and created CSV: {output_csv_path}")
    print(f"Extracted metrics: {json.dumps(metrics, indent=2)}")
    print(f"DataFrame shape: {df_results.shape}")
    
    return df_results


def main():
    args = parse_args()
    if not os.path.exists(args.log_file):
        print(f"Error: Log file {args.log_file} not found")
        return 1
    
    df_results = parse_llama_log(args.log_file, args.output_csv, args.tag)
    print(f"Transformed CSV saved to {args.output_csv}")
    return 0

if __name__ == "__main__":
    main()
