from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

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
RUN_TUTORIAL = REPO_ROOT / "scripts" / "run_tutorial.py"


@pytest.mark.parametrize("name", TUTORIALS, ids=TUTORIALS)
def test_tutorial(name: str):
    tutorial_path = TUTORIALS_DIR / f"{name}.py"
    assert tutorial_path.exists(), f"Missing tutorial file: {tutorial_path}"

    args = [sys.executable, str(RUN_TUTORIAL), str(tutorial_path)]

    if os.environ.get("TRITON_TEST_REPORTS", "false").lower() == "true":
        reports_dir = os.environ.get("TRITON_TEST_REPORTS_DIR", "")
        if reports_dir:
            args.append(f"--reports={reports_dir}")

    subprocess.run(args, check=True)
