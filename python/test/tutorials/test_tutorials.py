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

_FA_TUTORIAL = "06-fused-attention"


class CustomMark(triton.testing.Mark):
    """Redirects benchmark CSV reports to a given directory instead of cwd."""

    def __init__(self, fn, benchmarks, reports_path: pathlib.Path):
        self.fn = fn
        self.benchmarks = benchmarks
        self.reports_path = reports_path

    def run(self, **kwargs):
        if 'save_path' in kwargs:
            return super().run(**kwargs)
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = super().run(save_path=tmp_dir, **kwargs)
            for csv_file in pathlib.Path(tmp_dir).glob('*.csv'):
                print(f'Report file: {csv_file.name}')
                shutil.move(str(csv_file), str(self.reports_path))
            return result


@pytest.fixture(autouse=True)
def tutorial_environment(monkeypatch) -> pathlib.Path | None:
    """Prevent tutorials from leaking global state between test runs."""
    monkeypatch.setattr(sys, "argv", sys.argv[:])

    # Save and restore the triton allocator so tutorials that call
    # triton.set_allocator() (06, 08, 09) don't leak into subsequent tests.
    from triton.runtime import _allocation
    saved_token = _allocation._allocator.set(_allocation._allocator.get())

    reports_dir = None
    if os.environ.get("TRITON_TEST_REPORTS", "false").lower() == "true":
        reports_dir = os.environ.get("TRITON_TEST_REPORTS_DIR", "") or None

    yield pathlib.Path(reports_dir) if reports_dir else None

    _allocation._allocator.reset(saved_token)


def _configure_fa(mode: str, monkeypatch) -> None:
    """Apply FA run-mode constraints (fp8_only / skip_fp8) via env vars."""
    if mode == "fp8_only":
        monkeypatch.setenv("HEAD_DIM", os.environ.get("HEAD_DIM", "128"))
        monkeypatch.setenv("FWD_FP8_ONLY", "1")
    elif mode == "skip_fp8":
        monkeypatch.setenv("HEAD_DIM", os.environ.get("HEAD_DIM", "128"))
        monkeypatch.setenv("FWD_FP8_SKIP", "1")


def _run_tutorial(name: str, monkeypatch, tutorial_environment):
    """Run a single Triton tutorial by name."""
    tutorial_path = TUTORIALS_DIR / f"{name}.py"
    assert tutorial_path.exists(), f"Missing tutorial file: {tutorial_path}"

    reports_dir = tutorial_environment
    if reports_dir:
        report_path = reports_dir / tutorial_path.stem
        report_path.mkdir(parents=True, exist_ok=True)

        def perf_report(benchmarks):
            return lambda fn: CustomMark(fn, benchmarks, report_path)

        monkeypatch.setattr(triton.testing, "perf_report", perf_report)

    spec = importlib.util.spec_from_file_location('__main__', tutorial_path)
    if not spec or not spec.loader:
        raise AssertionError(f'Failed to load module from {tutorial_path}')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "argv", [str(tutorial_path)])
    spec.loader.exec_module(module)


# Hyphens become underscores for valid Python identifiers.
# Tutorial 06 is parametrized via fa_config, e.g. test_06_fused_attention[default].
# Skiplist files must match these node IDs exactly.

for t in TUTORIALS:
    name = f"test_{t.replace('-', '_')}"
    if t == _FA_TUTORIAL:

        def test(fa_config, monkeypatch, tutorial_environment, t=t):
            # "default" = no env-var overrides; uses environment as-is.
            if fa_config in ("fp8_only", "skip_fp8"):
                _configure_fa(fa_config, monkeypatch)
            _run_tutorial(t, monkeypatch, tutorial_environment)

    else:

        def test(monkeypatch, tutorial_environment, t=t):
            _run_tutorial(t, monkeypatch, tutorial_environment)

    globals()[name] = test

del t, name, test  # Don't export from module.
