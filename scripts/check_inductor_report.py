#!/usr/bin/env python
import argparse
from pathlib import Path
import csv
import sys


def check_report(suite, dtype, mode, test_mode, device, models_file):
    inductor_log_dir = Path("torch_compile_debug") / suite / dtype
    inductor_report_filename = f"inductor_{suite}_{dtype}_{mode}_{device}_{test_mode}.csv"
    inductor_report_path = Path(inductor_log_dir / inductor_report_filename)

    subset = []
    report = []
    exitcode = 0

    with open(models_file, encoding="utf-8") as f:
        subset = f.read().strip().split("\n")

    with open(inductor_report_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        report_with_header = []
        for l in reader:
            report_with_header.append(l)
        for r in report_with_header[1:]:
            if r[0] == device:
                report.append(r)

    test_list = [r[1] for r in report]

    if test_mode == "performance":
        for m in subset:
            if m not in test_list:
                exitcode = 1
                print(f"Test is not found in report: {m}")

    if test_mode == "accuracy":
        test_statuses = [r[3] for r in report]
        for m in subset:
            try:
                idx = test_list.index(m)
            except ValueError:
                exitcode = 1
                print(f"Test is NOT FOUND: {m}")
                continue
            if test_statuses[idx] != "pass":
                exitcode = 1
                print(f"Test is NOT PASSED: {m}")
    return exitcode


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--suite", required=True)
    argparser.add_argument("--dtype", required=True)
    argparser.add_argument("--mode", required=True, choices=("training", "inference"))
    argparser.add_argument("--test_mode", required=True, choices=("performance", "accuracy"))
    argparser.add_argument("--device", help="i.e. xpu", required=True)
    argparser.add_argument("--models-file", help="Subset of models list", required=True)
    args = argparser.parse_args()
    exitcode = check_report(args.suite, args.dtype, args.mode, args.test_mode, args.device, args.models_file)
    print(f"Report check result: {'SUCCESS' if exitcode == 0 else 'FAIL'}")
    sys.exit(exitcode)


if __name__ == "__main__":
    main()
