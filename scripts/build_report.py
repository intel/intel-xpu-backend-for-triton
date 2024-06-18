import argparse
import os
import uuid
import json
import datetime

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build report based on triton-benchmark run")
    parser.add_argument("source", help="Path to source csv file with benchmark results")
    parser.add_argument(
        "target",
        help="Path to result csv file with benchmark results including host info and dates",
    )
    parser.add_argument("--param_cols", help="Names of parameter columns, separated by commas.", required=True)
    parser.add_argument("--benchmark", help="Name of the benchmark.", required=True)
    parser.add_argument("--compiler", help="Name of the compiler, like `triton`.", required=True)
    parser.add_argument("--tflops_col", help="Column name with tflops.", required=True)
    parser.add_argument("--hbm_col", help="Column name with HBM results.", required=False, default=None)
    return parser.parse_args()


def check_cols(target_cols, all_cols):
    diff = set(target_cols).difference(all_cols)
    assert (len(diff) == 0), f"Couldn't find required columns: '{diff}' among available '{all_cols}'"


def transform_df(df, param_cols, tflops_col, hbm_col, benchmark, compiler):
    check_cols(param_cols, df.columns)
    check_cols([tflops_col] + [] if hbm_col is None else [hbm_col], df.columns)
    # Build json with parameters
    df_results = pd.DataFrame()
    df_results["params"] = [json.dumps(j) for j in df[param_cols].astype(int).to_dict("records")]
    df_results['tflops'] = df[tflops_col]
    if hbm_col is not None:
        df_results['hbm_gbs'] = df[hbm_col]

    df_results["run_uuid"] = uuid.uuid4().hex
    df_results["datetime"] = datetime.datetime.now()
    df_results["benchmark"] = benchmark
    df_results["compiler"] = compiler

    host_info = {
        n: os.getenv(n.upper(), default="")
        for n in ["libigc1_version", "level_zero_version", "gpu_device", "agama_version"]
    }
    assert host_info['gpu_device'], "Could not find GPU device description, was capture_device.sh called?"
    for name, val in host_info.items():
        df_results[name] = val

    return df_results


def main():
    args = parse_args()
    param_cols = args.param_cols.split(",")
    df = pd.read_csv(args.source)
    result_df = transform_df(df, param_cols=param_cols, tflops_col=args.tflops_col, hbm_col=args.hbm_col,
                             benchmark=args.benchmark, compiler=args.compiler)
    result_df.to_csv(args.target, index=False)


if __name__ == "__main__":
    main()
