#! /usr/bin/env python3

import argparse
import shutil
import subprocess
import os
import pandas as pd
from pathlib import Path
from typing import Optional


def get_config(ident: str) -> str:
    """Retrieve configuration name."""
    if ":" in ident:
        return ident.split(":")[0]

    return ident


def download(ident: str) -> bool:
    """Download artifacts for given configuration and CI run ID from Github."""
    if not shutil.which("gh"):
        print("Could not find 'gh' executable on the '$PATH'")
        return False

    if ":" not in ident:
        print("Invalid format, expecting 'name:Github CI Run ID' for download")
        return False

    name, run = ident.split(":", 1)

    ret = subprocess.run(["gh", "run", "download", "-R", "intel/intel-xpu-backend-for-triton", "-D", name, run],
                         capture_output=True)

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
    numDir = Path(get_config(args.numerator))
    denomDir = Path(get_config(args.denominator))

    if args.local:
        if ":" in args.numerator or ":" in args.denominator:
            print("Invalid format, expecting only 'name' for local run")
            return (None, None)

        if not numDir.is_dir():
            print(f"Directory {numDir} must exist if no download is happening.")
            return (None, None)

        if not denomDir.is_dir():
            print(f"Directory {denomDir} must exist if no download is happening.")
            return (None, None)
    else:
        if not download(args.numerator):
            return (None, None)
        if not download(args.denominator):
            return (None, None)

    return (numDir, denomDir)


def parse_data(config: str, df: pd.DataFrame, file: Path) -> pd.DataFrame:
    """Parse data from a single CSV file into the dataframe."""
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


def parse_directory(config: str, previous: pd.DataFrame, directory: Path) -> pd.DataFrame:
    """Parse all CSV files for a configuration in a directory, merging with
        the previous dataframe if present."""
    df = pd.DataFrame(columns=["dev", "name", "batch_size", f"speedup {config}", "suite", "datatype", "mode"])
    for file in Path(directory).rglob("*performance.csv"):
        df = parse_data(config, df, file)

    if previous is not None:
        df = df.merge(previous, how="outer", on=["suite", "datatype", "mode", "name", "dev"])
    return df


def eval_data(df: pd.DataFrame, numerator: str, denominator: str, plot: bool):
    """Evaluate the data, print a summary and plot if enabled."""
    numCol = f"speedup {numerator}"
    denomCol = f"speedup {denominator}"

    df.drop(columns=["batch_size_x", "batch_size_y"], inplace=True)

    bothFailed = df.loc[(df[numCol] == 0.0) & (df[denomCol] == 0.0)]
    print(f"Both failed ({bothFailed.shape[0]} configurations):")
    print(bothFailed.to_string())
    print("\n" * 2)

    numFailed = df.loc[(df[numCol] == 0.0) & (df[denomCol] != 0.0)]
    print(f"Only {numerator} failed ({numFailed.shape[0]} configurations):")
    print(numFailed.to_string())
    print("\n" * 2)

    denomFailed = df.loc[(df[numCol] != 0.0) & (df[denomCol] == 0.0)]
    print(f"Only {denominator} failed ({denomFailed.shape[0]} configurations):")
    print(denomFailed.to_string())
    print("\n" * 2)

    nanEntries = df[df[[numCol, denomCol]].isnull().any(axis=1)]
    print("NaN entries present:")
    print(nanEntries.to_string())
    print("\n" * 2)

    # Filter out NaN and zero values
    df = df[df[[numCol, denomCol]].notnull().all(1)]
    df = df.loc[(df[numCol] != 0.0) & (df[denomCol] != 0.0)]

    df["relative difference"] = ((df[numCol] - df[denomCol]) / df[denomCol])

    print("Overview of relative difference in speedup.\n"
          "Relative difference 0.0 means both perform identically,"
          f" relative difference > 0.0 means {numerator} performs better,"
          f" relative difference < 0.0 means {denominator} performs better")

    print(df["relative difference"].describe())
    print(f"Mean speedup for denominator: {df[denomCol].mean()}")
    print("\n" * 2)

    df.sort_values(by=["relative difference"], inplace=True, ignore_index=True, ascending=True)
    printCfgs = 10
    print(f"{printCfgs} fastest configurations ({denominator} faster than "
          "{numerator}, showing relative difference in speedup)")
    print(df.head(printCfgs))
    print("\n" * 2)
    df.sort_values(by=["relative difference"], inplace=True, ignore_index=True, ascending=False)
    print(f"{printCfgs} slowest configurations ({denominator} slower than "
          "{numerator}, showing relative difference in speedup)")
    print(df.head(printCfgs))
    print("\n" * 2)

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        df["xlabel"] = df[["suite", "mode", "datatype"]].agg(", ".join, axis=1)

        # Sort by configuration
        order = list(df["xlabel"].unique())
        order.sort()
        filename = f"performance-plot-{numerator}-{denominator}.pdf"
        with PdfPages(filename) as pdf:
            fig = plt.figure()
            plt.xticks(rotation=85)

            title = ("Relative difference 0.0 means both perform identically,\n"
                     f"relative difference > 0.0 means {numerator} performs better,\n"
                     f"relative difference < 0.0 means {denominator} performs better")
            plt.title(f"Comparison {numerator} vs {denominator}.")

            plt.figtext(1, 0.5, title)

            ax = sns.boxplot(df, x="xlabel", y="relative difference", order=order)

            ax.set(xlabel=None, ylabel="Relative difference in speedup")

            pdf.savefig(fig, bbox_inches="tight")
            print(f"Saved performance plot to {filename}")


if __name__ == "__main__":
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

    args = parser.parse_args()

    if args.path:
        path = Path(args.path).absolute()
        path.mkdir(parents=True, exist_ok=True)
        os.chdir(path)

    numCfg = get_config(args.numerator)
    denomCfg = get_config(args.denominator)
    csvFile = f"preprocessed-data-{numCfg}-{denomCfg}.csv"

    if args.eval_only:
        if not Path(csvFile).is_file():
            print(f"Could not find preprocessed data file {csvFile}")
            exit(-1)
        df = pd.read_csv(csvFile, header=0)
    else:
        (numDir, denomDir) = get_raw_data(args)

        if not numDir or not denomDir:
            print("Failed to obtain raw data")
            exit(-1)

        df = parse_directory(numCfg, None, numDir)
        df = parse_directory(denomCfg, df, denomDir)

        cols = [
            "dev", "suite", "name", "mode", "datatype", "batch_size_x", "batch_size_y", f"speedup {numCfg}",
            f"speedup {denomCfg}"
        ]
        df = df[cols]

        print(f"Storing preprocessed data to {csvFile}")
        df.to_csv(csvFile, index=False)

    eval_data(df, numCfg, denomCfg, (not args.no_plot))
