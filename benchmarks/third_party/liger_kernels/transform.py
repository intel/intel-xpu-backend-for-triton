import os
import uuid
import json
import argparse
import datetime

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform microbenchmark CSV for database upload."
    )
    parser.add_argument("source", help="Path to source CSV file with benchmark results")
    parser.add_argument("target", help="Path to target CSV file for database upload")
    parser.add_argument("--tag", help="Tag for the benchmark run", default="")
    return parser.parse_args()


def transform_df(df, tag):
    df_results = pd.DataFrame()

    df_results = df_results[
        df_results["metric_name"].eq("speed") & ~df_results["gpu_name"].str.contains("NVIDIA")
    ]
    df_results["run_uuid"] = uuid.uuid4().hex  # Generate a unique run ID
    df_results["datetime"] = datetime.datetime.now()
    df_results["benchmark"] = df["kernel_name"] + "-" + df["kernel_operation_mode"]
    df_results["compiler"] = df["kernel_provider"]
    # Use the 50th percentile value.
    df_results["time_ms"] = df["y_value_50"]
    df_results["comment"] = ""  # Empty comment

    # Create the parameters JSON, handling different x_value types correctly.
    def make_params(row):
        x_value = int(row["x_value"])
        params = {row["x_name"]: x_value}

        # Add extra_benchmark_config_str as dict
        # params.update(json.loads(row["extra_benchmark_config_str"]))

        return json.dumps(params)

    df_results["params"] = df.apply(make_params, axis=1)

    df_results["agama_version"] = df["liger_version"]  # Map liger_version to agama_version
    df_results["tag"] = tag  # Use provided tag, if any

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
        raise RuntimeError(
            "Could not find GPU device description, was `capture-hw-details.sh` called?"
        )
    for name, val in host_info.items():
        df_results[name] = val

    return df_results


def main():
    args = parse_args()
    df = pd.read_csv(args.source)
    result_df = transform_df(df, args.tag)
    result_df.to_csv(args.target, index=False)
    print(f"Transformed CSV saved to {args.target}")


if __name__ == "__main__":
    main()
