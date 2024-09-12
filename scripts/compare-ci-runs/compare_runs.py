#! /usr/bin/env python3

import argparse
import shutil
import subprocess
import os
import sys

from pathlib import Path
from typing import Optional

import pandas as pd


def get_config(ident: str) -> str:
    """Retrieve configuration name."""
    if ":" in ident:
        return ident.split(":")[0]

    return ident


def download(ident: str) -> bool:
    """Download artifacts for given configuration and CI run ID from GitHub."""
    if not shutil.which("gh"):
        print("Could not find 'gh' executable on the '$PATH'")
        return False

    if ":" not in ident:
        print("Invalid format, expecting 'name:Github CI Run ID' for download")
        return False

    name, run = ident.split(":", 1)

    ret = subprocess.run(["gh", "run", "download", "-R", "intel/intel-xpu-backend-for-triton", "-D", name, run],
                         capture_output=True, check=False)

    if ret.returncode != 0:
        print("Downloading run artifacts with 'gh' CLI failed")
        if ret.stdout:
            print("Command stdout:")
            print(ret.stdout.decode("UTF-8"))
        if ret.stderr:
            print("Command stderr:")
            print(ret.stderr.decode("UTF-8"))
        return False

    return True


def get_raw_data(args: argparse.Namespace) -> tuple[Optional[Path], Optional[Path]]:
    """Discover or download the raw data for both configurations."""
    num_dir = Path(get_config(args.numerator))
    denom_dir = Path(get_config(args.denominator))

    if args.local:
        if ":" in args.numerator or ":" in args.denominator:
            print("Invalid format, expecting only 'name' for local run")
            return None, None

        if not num_dir.is_dir():
            print(f"Directory {num_dir} must exist if no download is happening.")
            return None, None

        if not denom_dir.is_dir():
            print(f"Directory {denom_dir} must exist if no download is happening.")
            return None, None
    else:
        if not download(args.numerator):
            return None, None
        if not download(args.denominator):
            return None, None

    return num_dir, denom_dir


def parse_pytorch_benchmark_data(config: str, df: pd.DataFrame, file: Path) -> pd.DataFrame:
    """Parse pytorch benchmark data from a single CSV file into the dataframe."""
    path = Path(file).absolute()
    datatype = path.parts[-2]
    suite = path.parts[-3]

    mode = "unknown"
    if "inference" in path.parts[-1]:
        mode = "inference"
    elif "training" in path.parts[-1]:
        mode = "training"

    raw_data = pd.read_csv(file, header=0, usecols=["dev", "name", "batch_size", "speedup"])

    raw_data["suite"] = suite
    raw_data["datatype"] = datatype
    raw_data["mode"] = mode
    raw_data.rename(columns={"speedup": f"speedup {config}"}, inplace=True)

    return pd.concat([df, raw_data], ignore_index=True)


def merge_triton_xetla_reports_data(config: str, triton_file: Path, xetla_file: Path) -> pd.DataFrame:
    """Merge triton and xetla raw data."""
    try:
        triton_raw_data, xetla_raw_data = [
            pd.read_csv(file, header=0, usecols=["params", "tflops", "benchmark"])
            for file in [triton_file, xetla_file]
        ]
        triton_raw_data.rename(columns={"tflops": f"Triton-TFlops-{config}"}, inplace=True)
        xetla_raw_data.rename(columns={"tflops": f"XeTLA-TFlops-{config}"}, inplace=True)
        return triton_raw_data.merge(xetla_raw_data, how="outer", on=["params", "benchmark"])
    except FileNotFoundError:
        print(f"Warning: One or both files not found: {triton_file} or {xetla_file}")
        return pd.DataFrame()


def build_triton_benchmark_reports_path(directory: Path, report_name: str) -> str:
    """Construct the full file path for a given report name."""
    return os.path.join(directory, "benchmark-reports", f"{report_name}-report.csv")


def parse_triton_benchmark_data(config: str, df: pd.DataFrame, directory: Path) -> pd.DataFrame:
    """Parse triton benchmark data from a merged dataframe into the dataframe.
        Now focus on dft path for softmax, gemm and attention
        which include both xetla and triton data with regular name."""

    reports_files = {
        "softmax": ("softmax-triton", "softmax-xetla"), "gemm": ("gemm-triton", "gemm-xetla"), "attn":
        ("attn-triton", "attn-xetla")
    }

    reports_list = [df]
    for (triton_file, xetla_file) in reports_files.values():
        triton_path = build_triton_benchmark_reports_path(Path(directory), triton_file)
        xetla_path = build_triton_benchmark_reports_path(Path(directory), xetla_file)
        reports_list.append(merge_triton_xetla_reports_data(config, triton_path, xetla_path))

    return pd.concat(reports_list, ignore_index=True)


def parse_directory(triton_benchmark: bool, config: str, previous: pd.DataFrame, directory: Path) -> pd.DataFrame:
    """Parse all CSV files for a configuration in a directory, merging with
        the previous dataframe if present."""
    if triton_benchmark:
        df = pd.DataFrame()
        df = parse_triton_benchmark_data(config, df, directory)
    else:
        df = pd.DataFrame(columns=["dev", "name", "batch_size", f"speedup {config}", "suite", "datatype", "mode"])
        for file in Path(directory).rglob("*performance.csv"):
            df = parse_pytorch_benchmark_data(config, df, file)

    if previous is not None:
        df = df.merge(previous, how="outer", on=["params", "benchmark"]) if triton_benchmark else df.merge(
            previous, how="outer", on=["suite", "datatype", "mode", "name", "dev"])
    return df


def summarize_diff(triton_benchmark: bool, perf_index: str, plot: bool, df: pd.DataFrame, num_col: str, denom_col: str,
                   numerator: str, denominator: str):
    """Summarize data difference of numerator and denominator."""
    both_failed = df.loc[(df[num_col] == 0.0) & (df[denom_col] == 0.0)]
    print(f"Both failed ({both_failed.shape[0]} configurations):")
    print(both_failed.to_string())
    print("\n" * 2)

    num_failed = df.loc[(df[num_col] == 0.0) & (df[denom_col] != 0.0)]
    print(f"Only {numerator} failed ({num_failed.shape[0]} configurations):")
    print(num_failed.to_string())
    print("\n" * 2)

    denom_failed = df.loc[(df[num_col] != 0.0) & (df[denom_col] == 0.0)]
    print(f"Only {denominator} failed ({denom_failed.shape[0]} configurations):")
    print(denom_failed.to_string())
    print("\n" * 2)

    nan_entries = df[df[[num_col, denom_col]].isnull().any(axis=1)]
    print("NaN entries present:")
    print(nan_entries.to_string())
    print("\n" * 2)

    # Filter out NaN and zero values
    df = df[df[[num_col, denom_col]].notnull().all(1)]
    df = df.loc[(df[num_col] != 0.0) & (df[denom_col] != 0.0)]

    df["relative difference"] = (df[num_col] - df[denom_col]) / df[denom_col]

    print(f"Overview of relative difference in {perf_index}.\n"
          "Relative difference 0.0 means both perform identically,"
          f" relative difference > 0.0 means {numerator} performs better,"
          f" relative difference < 0.0 means {denominator} performs better")

    print(df["relative difference"].describe())
    print(f"Mean {perf_index} for denominator: {df[denom_col].mean()}")
    print("\n" * 2)

    df.sort_values(by=["relative difference"], inplace=True, ignore_index=True, ascending=True)
    print_cfgs = 10
    print(f"{print_cfgs} best configurations ({denominator} better than "
          f"{numerator}, showing relative difference in {perf_index})")
    print(df.head(print_cfgs))
    print("\n" * 2)
    df.sort_values(by=["relative difference"], inplace=True, ignore_index=True, ascending=False)
    print(f"{print_cfgs} worst configurations ({denominator} worse than "
          f"{numerator}, showing relative difference in {perf_index})")
    print(df.head(print_cfgs))
    print("\n" * 2)

    if plot:
        # pylint: disable=import-outside-toplevel
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        df["xlabel"] = df[["params", "benchmark"]].agg(
            ", ".join, axis=1) if triton_benchmark else df[["suite", "mode", "datatype"]].agg(", ".join, axis=1)

        # Sort by configuration
        order = list(df["xlabel"].unique())
        order.sort()
        filename = f"performance-plot-{num_col}-{denom_col}.pdf".lower()
        with PdfPages(filename) as pdf:
            fig = plt.figure()
            plt.xticks(rotation=85)

            title = ("Relative difference 0.0 means both perform identically,\n"
                     f"relative difference > 0.0 means {numerator} performs better,\n"
                     f"relative difference < 0.0 means {denominator} performs better")
            plt.title(f"Comparison {numerator} vs {denominator}.")

            plt.figtext(1, 0.5, title)

            ax = sns.boxplot(df, x="xlabel", y="relative difference", order=order)

            ax.set(xlabel=None, ylabel=f"Relative difference in {perf_index}")

            pdf.savefig(fig, bbox_inches="tight")
            print(f"Saved performance plot to {filename}")


def eval_data(triton_benchmark: bool, plot: bool, df: pd.DataFrame, numerator: str, denominator: str):
    """Evaluate the data, print a summary and plot if enabled."""
    if triton_benchmark:
        num_tri2xe_col = f"Tri2Xe-{numerator}"
        dem_tri2xe_col = f"Tri2Xe-{denominator}"

        df_ratio = df[["params", "benchmark", num_tri2xe_col, dem_tri2xe_col]]
        summarize_diff(triton_benchmark, "tri2xe", plot, df_ratio, num_tri2xe_col, dem_tri2xe_col, numerator,
                       denominator)
    else:
        num_col = f"speedup {numerator}"
        denom_col = f"speedup {denominator}"

        df.drop(columns=["batch_size_x", "batch_size_y"], inplace=True)
        summarize_diff(triton_benchmark, "speedup", plot, df, num_col, denom_col, numerator, denominator)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(prog="compare-runs", description="Compare performance of two CI runs")
    parser.add_argument("-n", "--numerator", help="Numerator in the comparison. Format 'name[:Github CI Run ID]'.",
                        required=True)
    parser.add_argument("-d", "--denominator", help="Denominator in the comparison. Format 'name[:Github CI Run ID]'.",
                        required=True)
    parser.add_argument("-p", "--path", help="Directory to store raw data and output.", default=None)
    parser.add_argument("-l", "--local", help="Use existing raw data instead of downloading from Github.",
                        action="store_true")
    parser.add_argument("-e", "--eval-only", help="Use existing preprocessed data", action="store_true")
    parser.add_argument("--no-plot", help="Do not plot, no requirement on seaborn and matplotlib", action="store_true")
    parser.add_argument("--triton-benchmark", help="Compare triton benchmark performance of two CI runs",
                        action="store_true")

    args = parser.parse_args()

    if args.path:
        path = Path(args.path).absolute()
        path.mkdir(parents=True, exist_ok=True)
        os.chdir(path)

    num_cfg = get_config(args.numerator)
    denom_cfg = get_config(args.denominator)
    csv_file = f"preprocessed-data-{num_cfg}-{denom_cfg}.csv"

    if args.eval_only:
        if not Path(csv_file).is_file():
            print(f"Could not find preprocessed data file {csv_file}")
            sys.exit(1)
        df = pd.read_csv(csv_file, header=0)
    else:
        (num_dir, denom_dir) = get_raw_data(args)

        if not num_dir or not denom_dir:
            print("Failed to obtain raw data")
            sys.exit(1)

        df = parse_directory(args.triton_benchmark, num_cfg, None, num_dir)
        df = parse_directory(args.triton_benchmark, denom_cfg, df, denom_dir)

        if args.triton_benchmark:
            cols = [
                "params", "benchmark", f"Triton-TFlops-{num_cfg}", f"XeTLA-TFlops-{num_cfg}",
                f"Triton-TFlops-{denom_cfg}", f"XeTLA-TFlops-{denom_cfg}"
            ]
        else:
            cols = [
                "dev", "suite", "name", "mode", "datatype", "batch_size_x", "batch_size_y", f"speedup {num_cfg}",
                f"speedup {denom_cfg}"
            ]

        df = df[cols]
        if args.triton_benchmark:
            df[f"Tri2Xe-{num_cfg}"] = df[f"Triton-TFlops-{num_cfg}"] / df[f"XeTLA-TFlops-{num_cfg}"]
            df[f"Tri2Xe-{denom_cfg}"] = df[f"Triton-TFlops-{denom_cfg}"] / df[f"XeTLA-TFlops-{denom_cfg}"]

        print(f"Storing preprocessed data to {csv_file}")
        df.to_csv(csv_file, index=False)

    eval_data(args.triton_benchmark, (not args.no_plot), df, num_cfg, denom_cfg)


if __name__ == "__main__":
    main()
