# SPDX-License-Identifier: Apache-2.0
"""Tune or benchmark CI workload inputs with --manifest inputs.json [--tune --clear-cache]."""

# pylint: disable=import-outside-toplevel

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
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
        subprocess.run(["git", "apply", str(PATCH)], cwd=folder, check=True)
        yield target


def load_tuner():
    with prepared_tuner() as path:
        spec = importlib.util.spec_from_file_location("tune_unified_attention", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load patched tuner from {path}")
        tuner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tuner)
    return tuner


def allocate_ci(case, device, seed):
    """Create Q/K/V tensors, sequence metadata, and output/scratch buffers matching the CI benchmark."""
    import torch

    del seed
    # Match the benchmark's random inputs independently of tuning order.
    torch.manual_seed(20)
    query_lens = [q for q, _ in case["seq_lens"]]
    kv_lens = [kv for _, kv in case["seq_lens"]]
    assert case["q_heads"] % case["k_heads"] == 0
    q = torch.randn(sum(query_lens), case["q_heads"], case["head_size"], dtype=torch.bfloat16, device=device)
    k = torch.randn(case["num_blocks"], case["block_size"], case["k_heads"], case["head_size"], dtype=torch.bfloat16,
                    device=device)
    v = torch.randn_like(k)
    block_tables = torch.randint(0, case["num_blocks"],
                                 (len(query_lens), (max(kv_lens) + case["block_size"] - 1) // case["block_size"]),
                                 dtype=torch.int32, device=device)
    inputs = {
        "q": q,
        "k": k,
        "v": v,
        "cu_seqlens_q": torch.tensor([0] + query_lens, dtype=torch.int32,
                                     device=device).cumsum(dim=0, dtype=torch.int32),
        "seqused_k": torch.tensor(kv_lens, dtype=torch.int32, device=device),
        "out": torch.empty_like(q),
        "block_table": block_tables,
        "max_seqlen_q": max(query_lens),
        "max_seqlen_k": max(kv_lens),
        "softmax_scale": case["head_size"]**-0.5,
        "causal": True,
        "window_size": (case["sliding_window"] - 1, 0) if case["sliding_window"] is not None else (-1, -1),
        "softcap": case["soft_cap"] if case["soft_cap"] is not None else 0,
        "q_descale": None,
        "k_descale": None,
        "v_descale": None,
        "seq_threshold_3D": 32,
        "num_par_softmax_segments": 16,
        "use_td": True,
    }
    if case["qdtype"] is not None:
        dtype = getattr(torch, case["qdtype"])
        for name in ("q", "k", "v"):
            inputs[name] = inputs[name].to(dtype)
        for name in ("k_descale", "v_descale"):
            inputs[name] = torch.rand((len(query_lens), case["k_heads"]), dtype=torch.float32, device=device)
    shape = (32, case["q_heads"], 16)
    inputs.update(
        softmax_segm_output=torch.empty((*shape, 1 << (case["head_size"] - 1).bit_length()), dtype=torch.float32,
                                        device=device),
        softmax_segm_max=torch.empty(shape, dtype=torch.float32, device=device),
        softmax_segm_expsum=torch.empty(shape, dtype=torch.float32, device=device),
    )
    return inputs


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--check-patch", action="store_true",
                        help="Check tuner patch compatibility without importing PyTorch or vLLM")
    inputs.add_argument("--manifest")
    parser.add_argument("--tune", action="store_true", help="Search candidates and export winners")
    parser.add_argument("--save-dir", default=str(BENCHMARK_DIR / "configs"))
    parser.add_argument("--measurements", help="Optional detailed tuning checkpoint JSON")
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--confirmation-rounds", type=int, default=5)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--warmup-ms", type=float, default=10)
    parser.add_argument("--rep-ms", type=float, default=50)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Worst-regret tie tolerance before geometric-mean tie-breaking")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Evict 256 MB before each graph replay, excluding eviction from timing")
    args = parser.parse_args(argv)
    if args.check_patch:
        if len(argv) != 1:
            parser.error("--check-patch cannot be combined with tuning options")
        with prepared_tuner():
            print("UA CI tuner patch applies cleanly")
        return
    for name in ("rounds", "warmup_ms", "rep_ms", "confirmation_rounds", "finalists"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            option = name.replace("_", "-")
            parser.error(f"--{option} must be finite and positive")
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("--tolerance must be finite and nonnegative")
    if args.measurements and not args.tune:
        parser.error("--measurements requires --tune")
    workloads = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if workloads == []:
        print("No inputs for this part; skipping tuning or benchmarking")
        return

    tuner = load_tuner()
    timer = tuner.benchmark_callable
    if args.clear_cache:
        from timing import cold_graph_timer

        timer = cold_graph_timer
    scope = "attention_sequence_cold_graph_device" if args.clear_cache else "attention_sequence_graph_device"
    tuner.main(args, workloads, allocate=allocate_ci, timer=timer, timing_scope=scope)


if __name__ == "__main__":
    main()
