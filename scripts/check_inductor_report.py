#!/usr/bin/env python
import argparse
from pathlib import Path
import csv
import sys
from dataclasses import dataclass


@dataclass
class PassedArgs:
    suite: str
    dtype: str
    mode: str
    test_mode: str
    device: str
    models_file: str
    inductor_log_dir: str


def get_inductor_report_path(args: PassedArgs) -> Path:
    inductor_log_dir_leaf = Path(args.inductor_log_dir) / args.suite / args.dtype
    inductor_report_filename = f"inductor_{args.suite}_{args.dtype}_{args.mode}_{args.device}_{args.test_mode}.csv"
    return Path(inductor_log_dir_leaf / inductor_report_filename)


def check_report(args: PassedArgs) -> int:
    test_mode = args.test_mode
    inductor_report_path = get_inductor_report_path(args)

    subset = []
    report = []
    exitcode = 0

    with open(args.models_file, encoding="utf-8") as f:
        subset = f.read().splitlines()

    with open(inductor_report_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        report_with_header = []
        for l in reader:
            report_with_header.append(l)
        for r in report_with_header[1:]:
            if r[0] == args.device:
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
    argparser.add_argument("--mode", required=True, choices=("inference", "training", "inference-with-freezing"))
    argparser.add_argument("--test_mode", required=True, choices=("performance", "accuracy"))
    argparser.add_argument("--device", help="i.e. xpu", required=True)
    argparser.add_argument("--models-file", help="Subset of models list", required=True)
    argparser.add_argument("--inductor-log-dir", help="Inductor test log directory", default="inductor_log")
    parsed_args = argparser.parse_args()
    passed_args = PassedArgs(**vars(parsed_args))
    exitcode = check_report(passed_args)
    print(f"Report check result: {'SUCCESS' if exitcode == 0 else 'FAIL'}")
    sys.exit(exitcode)


if __name__ == "__main__":
    main()
