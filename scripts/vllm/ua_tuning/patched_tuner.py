# SPDX-License-Identifier: Apache-2.0
"""Apply CI adaptations to a temporary tuner copy; run directly to check the patch."""

import importlib.util
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmarks/triton_kernels_benchmark/vllm/unified_attention"
PATCH = Path(__file__).with_name("tuner-ci.patch")


@contextmanager
def prepared_tuner():
    with tempfile.TemporaryDirectory(prefix="ua-ci-tuner-") as folder:
        target = Path(folder) / "tune_unified_attention.py"
        shutil.copyfile(BENCHMARK_DIR / target.name, target)
        for options in (("--check", ), ()):
            subprocess.run(["git", "apply", *options, str(PATCH)], cwd=folder, check=True)
        yield target


def load_tuner():
    with prepared_tuner() as path:
        spec = importlib.util.spec_from_file_location("tune_unified_attention", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load patched tuner from {path}")
        tuner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tuner)
    return tuner


if __name__ == "__main__":
    with prepared_tuner():
        print("UA CI tuner patch applies cleanly")
