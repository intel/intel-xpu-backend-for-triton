from typing import Optional
from dataclasses import asdict, dataclass, fields

import argparse
import warnings
import os
import uuid
import json
import datetime

import pandas as pd


@dataclass
class PassedArgs:  # pylint: disable=too-many-instance-attributes
    source: str
    target: str
    param_cols: str
    benchmark: str
    compiler: str
    tflops_col: str
    hbm_col: Optional[str] = None
    tag: str = ""
    mask: bool = False


@dataclass
class HostInfo:
    libigc1_version: str = ""
    level_zero_version: str = ""
    gpu_device: str = ""
    agama_version: str = ""
    torch_version: str = ""
    compiler_version: str = ""
    benchmarking_method: str = ""

    @classmethod
    def from_env(cls) -> "HostInfo":
        kwargs = {f.name: os.getenv(f.name.upper(), "") for f in fields(cls)}
        host_info_obj = cls(**kwargs)
        if not host_info_obj.gpu_device:
            raise ValueError("Could not find GPU device description, was `capture-hw-details.sh` called?")
        return host_info_obj


def parse_args() -> PassedArgs:
    parser = argparse.ArgumentParser(description="Build report based on triton-benchmark run")
    parser.add_argument("source", help="Path to source csv file with benchmark results")
    parser.add_argument(
        "target",
        help="Path to result csv file with benchmark results including host info and dates",
    )
    parser.add_argument(
        "--param_cols",
        help="Names of parameter columns, separated by commas.",
        required=True,
    )
    parser.add_argument("--benchmark", help="Name of the benchmark.", required=True)
    parser.add_argument("--compiler", help="Name of the compiler, like `triton`.", required=True)
    parser.add_argument("--tflops_col", help="Column name with tflops.", required=True)
    parser.add_argument("--hbm_col", help="Column name with HBM results.", required=False, default=None)
    parser.add_argument("--tag", help="How to tag results", required=False, default="")
    parser.add_argument("--mask", help="Mask identifiers among the params", required=False, action="store_true")
    parsed_args = parser.parse_args()
    return PassedArgs(**vars(parsed_args))


def check_cols(target_cols, all_cols):
    diff = set(target_cols).difference(all_cols)
    if len(diff) != 0:
        raise ValueError(f"Couldn't find required columns: '{diff}' among available '{all_cols}'")


def build_report(args: PassedArgs, results_df: Optional[pd.DataFrame] = None):
    df = pd.read_csv(args.source) if results_df is None else results_df
    param_cols = args.param_cols.split(",")
    hbm_col = args.hbm_col
    check_cols(param_cols, df.columns)
    check_cols([args.tflops_col] + [] if hbm_col is None else [hbm_col], df.columns)
    # Build json with parameters
    df_results = pd.DataFrame()
    # Type conversion to int is important here, because dashboards expect
    # int values.
    # Changing it without changing dashboards and database will
    # break comparison of old and new results
    if args.mask:
        df_results["MASK"] = df[param_cols[-1]]
        param_cols = param_cols[:-1]
        for p in param_cols:
            df[p] = df[p].astype(int)
            df_results["params"] = [json.dumps(j) for j in df[[*param_cols, "MASK"]].to_dict("records")]
    else:
        df_results["params"] = [json.dumps(j) for j in df[param_cols].astype(int).to_dict("records")]
    df_results["tflops"] = df[args.tflops_col]
    if hbm_col is not None:
        df_results["hbm_gbs"] = df[hbm_col]

    if "run_counter" in df.columns:
        # We are currently using `run_counter` as a way to separate runs inside of a one benchmark run.
        max_counter = df["run_counter"].max()
        mapping = {i: uuid.uuid4().hex for i in range(1, max_counter + 1)}
        df_results["run_uuid"] = df["run_counter"].map(mapping)
    else:
        df_results["run_uuid"] = uuid.uuid4().hex

    # All incoming benchmarks should have datetime now
    if "datetime" not in df.columns:
        warnings.warn("No datetime column found in the input file, using current time")
        df_results["datetime"] = datetime.datetime.now()
    else:
        df_results["datetime"] = df["datetime"]
    df_results["benchmark"] = args.benchmark
    df_results["compiler"] = args.compiler
    df_results["tag"] = args.tag

    for name, val in asdict(HostInfo.from_env()).items():
        df_results[name] = val

    df_results.to_csv(args.target, index=False)


def main():
    args = parse_args()
    build_report(args)


if __name__ == "__main__":
    main()
