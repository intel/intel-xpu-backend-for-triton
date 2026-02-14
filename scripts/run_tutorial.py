"""Script to run a Triton tutorial and collect generated csv files."""

import argparse
import datetime
import importlib.util
import pathlib
import shutil
import tempfile
import sys

import triton.testing


class CustomMark(triton.testing.Mark):  # pylint: disable=too-few-public-methods
    """Custom Mark to set save_path."""

    def __init__(self, fn, benchmarks, reports_path: pathlib.Path):
        self.fn = fn
        self.benchmarks = benchmarks
        self.reports_path = reports_path

    def run(self, **kwargs):
        """Runs a benchmark."""
        if 'save_path' in kwargs:
            return super().run(**kwargs)
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = super().run(save_path=tmp_dir, **kwargs)
            for file in pathlib.Path(tmp_dir).glob('*.csv'):
                print(f'Report file: {file.name}')
                shutil.move(file, self.reports_path)
            return result


def create_argument_parser() -> argparse.ArgumentParser:
    """Creates argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('tutorial', help='Tutorial to run')
    parser.add_argument('--reports', required=False, type=str, help='Directory to store tutorial CSV reports')
    return parser


def run_tutorial(path: pathlib.Path) -> float:
    """Runs tutorial."""
    spec = importlib.util.spec_from_file_location('__main__', path)
    if not spec or not spec.loader:
        raise AssertionError(f'Failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    # Reset sys.argv because some tutorials, such as 09, parse their command line arguments.
    sys.argv = [str(path)]
    start_time = datetime.datetime.now()
    spec.loader.exec_module(module)
    elapsed_time = datetime.datetime.now() - start_time
    return elapsed_time.total_seconds()


def main():
    """Main."""
    args = create_argument_parser().parse_args()
    tutorial_path = pathlib.Path(args.tutorial)

    reports_path = pathlib.Path(args.reports) if args.reports else None
    name = tutorial_path.stem

    if reports_path:
        report_path = reports_path / name
        report_path.mkdir(parents=True, exist_ok=True)

        def perf_report(benchmarks):
            """Marks a function for benchmarking."""
            return lambda fn: CustomMark(fn, benchmarks, report_path)

        triton.testing.perf_report = perf_report

    run_tutorial(tutorial_path)


if __name__ == '__main__':
    main()
