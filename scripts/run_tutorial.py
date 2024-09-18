"""Script to run a Triton tutorial and collect csv files generated the tutorial."""

import argparse
import importlib.util
import pathlib


def create_argument_parser() -> argparse.ArgumentParser:
    """Creates argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('tutorial', help='Tutorial to run')
    return parser


def run_tutorial(path: pathlib.Path):
    """Runs """
    name = path.stem.replace('-', '_')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def main():
    """Main."""
    args = create_argument_parser().parse_args()
    run_tutorial(pathlib.Path(args.tutorial))


if __name__ == '__main__':
    main()

