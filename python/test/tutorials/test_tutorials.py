from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import sys
import tempfile

import pytest

import triton.testing

TUTORIALS = [
    "01-vector-add",
    "02-fused-softmax",
    "03-matrix-multiplication",
    "04-low-memory-dropout",
    "05-layer-norm",
    "06-fused-attention",
    "07-extern-functions",
    "08-grouped-gemm",
    "09-persistent-matmul",
    "10-experimental-block-pointer",
]

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
TUTORIALS_DIR = REPO_ROOT / "python" / "tutorials"


class CustomMark(triton.testing.Mark):
    """Custom Mark that redirects benchmark CSV reports to a given directory."""

    def __init__(self, fn, benchmarks, reports_path: pathlib.Path):
        self.fn = fn
        self.benchmarks = benchmarks
        self.reports_path = reports_path

    def run(self, **kwargs):
        """Run benchmarks, moving generated CSV files to reports_path."""
        if 'save_path' in kwargs:
            return super().run(**kwargs)
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = super().run(save_path=tmp_dir, **kwargs)
            for csv_file in pathlib.Path(tmp_dir).glob('*.csv'):
                print(f'Report file: {csv_file.name}')
                shutil.move(str(csv_file), str(self.reports_path))
            return result


@pytest.fixture(autouse=True)
def tutorial_environment():
    """Save and restore global state around each tutorial run.

    Handles:
    - sys.argv: some tutorials (e.g. 09) parse command line arguments.
    - triton.testing.perf_report: monkey-patched when CSV reports are enabled.
    """
    original_argv = sys.argv[:]
    original_perf_report = triton.testing.perf_report

    reports_dir = None
    if os.environ.get("TRITON_TEST_REPORTS", "false").lower() == "true":
        reports_dir = os.environ.get("TRITON_TEST_REPORTS_DIR", "") or None

    yield reports_dir

    sys.argv = original_argv
    triton.testing.perf_report = original_perf_report


@pytest.mark.parametrize("name", TUTORIALS, ids=TUTORIALS)
def test_tutorial(name: str, tutorial_environment):
    """Run a single Triton tutorial as a pytest test case."""
    tutorial_path = TUTORIALS_DIR / f"{name}.py"
    assert tutorial_path.exists(), f"Missing tutorial file: {tutorial_path}"

    reports_dir = tutorial_environment
    if reports_dir:
        report_path = pathlib.Path(reports_dir) / tutorial_path.stem
        report_path.mkdir(parents=True, exist_ok=True)

        def perf_report(benchmarks):
            """Marks a function for benchmarking with report redirection."""
            return lambda fn: CustomMark(fn, benchmarks, report_path)

        triton.testing.perf_report = perf_report

    spec = importlib.util.spec_from_file_location('__main__', tutorial_path)
    if not spec or not spec.loader:
        raise AssertionError(f'Failed to load module from {tutorial_path}')
    module = importlib.util.module_from_spec(spec)
    sys.argv = [str(tutorial_path)]
    spec.loader.exec_module(module)
