# SPDX-License-Identifier: Apache-2.0
"""Tune or benchmark CI fixtures with --manifest inputs.json [--tune --clear-cache]."""

import ast
from functools import cache
from types import SimpleNamespace

import torch
import triton

import manifest


class _CapturedAttention(Exception):
    """Stop the CI fixture before attention executes."""


def _capture(**kwargs):
    raise _CapturedAttention(kwargs)


@cache
def _benchmark_fixture(device):
    # Extract the fixture without importing CI's provider and reporting setup.
    path = manifest.BENCHMARK_DIR / "unified_attention_benchmark.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "benchmark")
    node.decorator_list = []
    namespace = {
        "torch": torch,
        "triton": triton,
        "DEVICE": device,
        "BENCHMARKING_CONFIG": {"verify": False},
        "BENCHMARK_SKIPLIST": set(),
        "is_td_patched": True,
        "unified_attention": _capture,
        "benchmark_suite": SimpleNamespace(assert_close=lambda fn, *args, **kwargs: fn()),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # pylint: disable=exec-used
    return namespace["benchmark"]


def allocate_ci(case, device, seed):
    """Use CI's fixed seed, tensor layouts, scales and scratch-buffer sizes."""
    del seed  # The fixture sets its own seed, independently of tuning order.
    expected = {
        "q_layout": "contiguous",
        "kv_layout": "contiguous",
        "out_dtype": "bfloat16",
        "seq_threshold_3d": 32,
        "num_segments": 16,
    }
    for name, value in expected.items():
        if case[name] != value:
            raise ValueError(f"CI fixtures require {name}={value!r}")
    if case["dtype"] not in ("bfloat16", "float8_e4m3fn"):
        raise ValueError("CI fixtures support BF16 and FP8 inputs")
    if case.get("softmax_scale", case["head_size"]**-0.5) != case["head_size"]**-0.5:
        raise ValueError("CI fixtures require softmax_scale=head_size**-0.5")
    if "num_blocks" not in case or case["num_blocks"] <= 0:
        raise ValueError("CI fixtures require a positive num_blocks")
    previous_device = torch.get_default_device()
    try:
        _benchmark_fixture(str(device))(q_heads=case["q_heads"], k_heads=case["kv_heads"], head_size=case["head_size"],
                                        qdtype=torch.float8_e4m3fn if case["dtype"] == "float8_e4m3fn" else None,
                                        seq_lens=list(zip(case["query_lens"],
                                                          case["kv_lens"])), sliding_window=case["sliding_window"]
                                        or None, soft_cap=case["softcap"] or None, num_blocks=case["num_blocks"],
                                        block_size=case["block_size"], provider="triton-td")
    except _CapturedAttention as result:
        return result.args[0]
    finally:
        torch.set_default_device(previous_device)
    raise RuntimeError("CI fixture did not reach the attention call")


def main(argv=None):
    manifest.main(argv, allocate=allocate_ci, description=__doc__)


if __name__ == "__main__":
    main()
