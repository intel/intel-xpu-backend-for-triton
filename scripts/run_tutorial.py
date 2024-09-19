"""Script to run a Triton tutorial and collect generated csv files."""

import argparse
import importlib.util
import pathlib
import shutil
import tempfile

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
    parser.add_argument('--reports', required=False, type=str, default='.',
                        help='Directory to store tutorial CSV reports, default: %(default)s')
    return parser


def run_tutorial(path: pathlib.Path):
    """Runs """
    spec = importlib.util.spec_from_file_location('__main__', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def main():
    """Main."""
    args = create_argument_parser().parse_args()
    tutorial_path = pathlib.Path(args.tutorial)
    reports_path = pathlib.Path(args.reports)
    name = tutorial_path.stem
    report_path = reports_path / name
    report_path.mkdir(parents=True, exist_ok=True)

    def perf_report(benchmarks):
        """Marks a function for benchmarking."""
        return lambda fn: CustomMark(fn, benchmarks, report_path)

    triton.testing.perf_report = perf_report
    run_tutorial(tutorial_path)


if __name__ == '__main__':
    main()
