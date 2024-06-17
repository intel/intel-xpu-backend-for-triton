import argparse
import os
import uuid
import json
import datetime

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build report based on triton-benchmark run")
    parser.add_argument("source", help="Path to source csv file with benchmark results", required=True)
    parser.add_argument(
        "target",
        required=True,
        help="Path to result csv file with benchmark results including host info and dates",
    )
    parser.add_argument("--param_cols", help="Names of parameter columns, separated by commas.", required=True)
    parser.add_argument("--name", help="Name of the benchmark.", required=True)
    parser.add_argument("--result_cols", help="Names of the result columns, separated by commas.", required=True)
    return parser.parse_args()


def check_cols(target_cols, all_cols):
    diff = set(target_cols).difference(all_cols)
    assert (len(diff) == 0), f"Couldn't find required columns: '{diff}' among available '{all_cols}'"


def parse_result(name):
    name = name.lower()
    if "triton" in name:
        return "triton"
    elif "xetla" in name:
        return "xetla"
    else:
        return name


def transform_df(df, param_cols, result_cols, bench_name):

    check_cols(param_cols, df.columns)
    check_cols(result_cols, df.columns)
    # Build json with parameters
    df_results = df[result_cols].copy()
    df_results["params"] = [json.dumps(j) for j in df[param_cols].astype(int).to_dict("records")]

    df = pd.melt(
        df_results,
        id_vars=["params"],
        value_vars=result_cols,
        value_name="tflops",
        var_name="comment",
    )

    df["compiler"] = df["comment"].apply(parse_result)
    df["run_uuid"] = uuid.uuid4().hex
    df["datetime"] = datetime.datetime.now()
    df["benchmark"] = bench_name

    host_info = {
        n: os.getenv(n.upper(), default="")
        for n in ["libigc1_version", "level_zero_version", "gpu_device", "agama_version"]
    }
    assert host_info['gpu_device'], "Could not find GPU device description, was capture_device.sh called?"
    for name, val in host_info.items():
        df[name] = val

    return df


def main():
    args = parse_args()
    param_cols = args.param_cols.split(",")
    result_cols = args.result_cols.split(",")
    df = pd.read_csv(args.source)
    result_df = transform_df(df, param_cols=param_cols, result_cols=result_cols, bench_name=args.name)
    result_df.to_csv(args.target, index=False)


if __name__ == "__main__":
    main()
