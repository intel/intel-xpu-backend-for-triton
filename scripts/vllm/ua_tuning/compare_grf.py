# SPDX-License-Identifier: Apache-2.0
"""Independently tune explicit 128/256 GRF modes and compare frozen winners."""

import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from tune_ci import allocate_ci, load_tuner

MODES = ("128", "256")
PAIRS = 5
TIE_FRACTION = 0.03


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def mode_candidates(tuner, key):
    # Preserve the official grid and replace default with explicit 128.
    return [{**config, "grf_mode": "128" if config["grf_mode"] == "default" else config["grf_mode"]}
            for config in tuner.candidate_configs(key)]


def select_winners(options):
    winners = {}
    for mode in MODES:
        valid = [
            option for option in options
            if option["config"]["grf_mode"] == mode and option["status"] == "ok" and option["samples_ms"]
        ]
        if not valid:
            raise RuntimeError(f"No correct, launchable {mode}-GRF candidate")
        winners[mode] = min(valid, key=lambda option: (statistics.median(option["samples_ms"]), option["id"]))
    return winners


def summarize(samples):
    medians = {mode: statistics.median(samples[mode]) for mode in MODES}
    if not all(math.isfinite(value) and value > 0 for value in medians.values()):
        raise RuntimeError("Invalid winner timing")
    # Require a >3% reduction relative to the slower mode.
    ratio = medians["128"] / medians["256"]
    winner = "128" if ratio < 1 - TIE_FRACTION else "256" if 1 / ratio < 1 - TIE_FRACTION else "tie"
    return {"median_ms": medians, "samples_ms": samples, "winner": winner}


def native_metadata(compiled):
    import yaml
    from elftools.elf.elffile import ELFFile

    binary = compiled.asm["zebin"]
    section = ELFFile(io.BytesIO(binary)).get_section_by_name(".ze_info")
    if section is None:
        raise RuntimeError("Missing final native .ze_info metadata")
    kernels = yaml.safe_load(section.data())["kernels"]
    return {
        "sha256":
        hashlib.sha256(binary).hexdigest(),
        "kernels": [{
            "name": kernel["name"], **{
                key: kernel["execution_env"].get(key, 0 if key in ("spill_size", "slm_size") else None)
                for key in ("grf_count", "spill_size", "simd_size", "slm_size")
            }
        }
                    for kernel in kernels],
    }


def reference(torch, case, inputs):
    """Chunk the paged PyTorch reference to bound memory on the existing CI inputs.

    CI passes KVQuantMode.NONE, including with FP8 storage, so the supplied
    per-tensor descales are unused by the kernel and by this reference.
    """
    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    expected = torch.empty_like(inputs["out"])
    offset = 0
    for sequence, (qlen, kvlen) in enumerate(case["seq_lens"]):
        count = (kvlen + case["block_size"] - 1) // case["block_size"]
        pages = inputs["block_table"][sequence, :count].long()
        # Gather FP8 storage as bytes, avoiding a float copy of the entire cache.
        keys = k[pages] if k.dtype == torch.bfloat16 else k.view(torch.uint8)[pages].view(k.dtype)
        values = v[pages] if v.dtype == torch.bfloat16 else v.view(torch.uint8)[pages].view(v.dtype)
        repeat = case["q_heads"] // case["k_heads"]
        keys = keys.flatten(0, 1)[:kvlen].repeat_interleave(repeat, dim=1).transpose(0, 1).float()
        values = values.flatten(0, 1)[:kvlen].repeat_interleave(repeat, dim=1).transpose(0, 1).float()
        positions = torch.arange(kvlen, device=q.device)
        for start in range(0, qlen, 128):
            stop = min(start + 128, qlen)
            query = q[offset + start:offset + stop].transpose(0, 1).float()
            scores = (query @ keys.transpose(1, 2)) * inputs["softmax_scale"]
            if inputs["softcap"]:
                scores = inputs["softcap"] * torch.tanh(scores / inputs["softcap"])
            query_positions = kvlen - qlen + torch.arange(start, stop, device=q.device)
            mask = positions[None, :] <= query_positions[:, None]
            if case["sliding_window"]:
                mask &= query_positions[:, None] - positions[None, :] < case["sliding_window"]
            scores.masked_fill_(~mask, float("-inf"))
            expected[offset + start:offset + stop] = (scores.softmax(-1) @ values).transpose(0, 1)
        offset += qlen
    return expected


class Runner:

    def __init__(self, device):
        import torch
        from timing import cold_graph_timer
        from triton import knobs

        knobs.intel.gen_native_code = True
        self.torch = torch
        self.device = torch.device(device)
        torch.xpu.set_device(self.device)
        self.tuner = load_tuner()
        self.timer = cold_graph_timer
        self.compiled = {}
        ua = importlib.import_module(self.tuner.unified_attention.__module__)
        for name in ("kernel_unified_attention", "reduce_segments"):
            kernel = getattr(ua, name)
            original = kernel.run

            def capture(*args, _run=original, _name=name, **kwargs):
                result = _run(*args, **kwargs)
                self.compiled[_name] = result
                return result

            kernel.run = capture

    def release(self):
        self.inputs = None
        self.expected = None
        self.compiled.clear()
        self.torch.xpu.empty_cache()

    def allocate(self, case, seed):
        self.release()
        self.inputs = allocate_ci(case, self.device, seed)
        self.expected = reference(self.torch, case, self.inputs)
        self.atol, self.rtol = (0.025, 0.01) if case["qdtype"] is None else (3 / 8 + 1e-6, 0.15)
        self.tuner.synchronize(self.device)

    def check(self):
        self.torch.testing.assert_close(self.inputs["out"], self.expected, atol=self.atol, rtol=self.rtol)

    def prepare(self, option, inputs, device, binary_dir=None):
        self.compiled.clear()
        inputs["out"].fill_(float("nan"))
        self.tuner.prepare_option(option, inputs, device)
        if option["status"] != "ok":
            return
        self.check()
        option["native"] = {name: native_metadata(code) for name, code in self.compiled.items()}
        kernels = option["native"]["kernel_unified_attention"]["kernels"]
        expected_grfs = int(option["config"]["grf_mode"])
        if not kernels or any(kernel["grf_count"] != expected_grfs for kernel in kernels):
            raise RuntimeError(f"Requested {expected_grfs} GRFs, native metadata is {kernels}")
        option["correctness"] = {"passed": True, "atol": self.atol, "rtol": self.rtol}
        if binary_dir is not None:
            for code in self.compiled.values():
                raw = code.asm["zebin"]
                (binary_dir / f"{hashlib.sha256(raw).hexdigest()}.zebin").write_bytes(raw)

    def time(self, fn, device):
        value = self.timer(fn, device, warmup_ms=self.tuner.WARMUP_MS, rep_ms=self.tuner.REP_MS)
        self.check()
        return value

    def compare(self, configs, directory):
        samples = {mode: [] for mode in MODES}
        native = {}
        for mode in MODES:
            option = {"id": mode, "config": configs[mode], "status": "pending", "samples_ms": []}
            self.prepare(option, self.inputs, self.device, binary_dir=directory)
            if option["status"] != "ok":
                raise RuntimeError(f"Frozen winner no longer launches: {option}")
            native[mode] = option["native"]
        for pair in range(PAIRS):
            for mode in MODES if pair % 2 == 0 else reversed(MODES):
                config = self.tuner.runtime.AttentionConfig(**configs[mode])
                with self.tuner.runtime.override_config(config):
                    samples[mode].append(self.time(lambda: self.tuner.unified_attention(**self.inputs), self.device))
        return {**summarize(samples), "native": native}


def run(args):
    runner = Runner(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    with runner.torch.inference_mode():
        if args.replay:
            frozen = json.loads(args.replay.read_text())
            runner.allocate(frozen["case"], args.seed)
            write_json(args.output / "confirmation.json", runner.compare(frozen["configs"], args.output))
            return
        workloads = json.loads(args.manifest.read_text())
        rows = []
        for index, case in enumerate(workloads):
            directory = args.output / case["id"]
            directory.mkdir()
            print(f"Workload {index + 1}/{len(workloads)}: {case['id']}", flush=True)
            runner.allocate(case, args.seed + index)
            key = runner.tuner.runtime.make_attention_key(**runner.inputs)
            options = []
            for config in mode_candidates(runner.tuner, key):
                if not runner.tuner.runtime.validate_config(runner.tuner.runtime.AttentionConfig(**config), key, "xpu"):
                    continue
                option = {"id": runner.tuner.config_id(config), "config": config, "status": "pending", "samples_ms": []}
                options.append(option)
                runner.prepare(option, runner.inputs, runner.device)
                if len(options) % 32 == 0:
                    write_json(directory / "search.json", options)
                    print(f"Prepared {len(options)} candidates", flush=True)
            write_json(directory / "search.json", options)
            runner.tuner.time_options(options, runner.inputs, runner.device, args.seed + index, rounds=args.rounds,
                                      timer=runner.time)
            write_json(directory / "search.json", options)
            winners = select_winners(options)
            configs = {mode: winners[mode]["config"] for mode in MODES}
            frozen = {"case": case, "key": asdict(key), "configs": configs}
            write_json(directory / "frozen.json", frozen)
            result = runner.compare(configs, directory)
            if result["winner"] == "128":
                # Let the child use the full memory budget of the original workload.
                runner.release()
                subprocess.run([
                    sys.executable,
                    str(Path(__file__).resolve()), "--replay",
                    str(directory / "frozen.json"), "--device", args.device, "--seed",
                    str(args.seed + index), "--output",
                    str(directory / "fresh")
                ], check=True)
                confirmation = json.loads((directory / "fresh/confirmation.json").read_text())
                if confirmation["native"] != result["native"]:
                    raise RuntimeError("Fresh-process native binaries differ from frozen winners")
                result["confirmation"] = confirmation
                if confirmation["winner"] != "128":
                    result["winner"] = "unconfirmed_128"
            write_json(directory / "comparison.json", result)
            row = {
                "workload": case["id"], **{f"config_{mode}": runner.tuner.canonical(configs[mode])
                                           for mode in MODES},
                **{f"latency_{mode}_us": result["median_ms"][mode] * 1000
                   for mode in MODES}, "winner": result["winner"]
            }
            rows.append(row)
            with (args.output / "comparison.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps(row), flush=True)
        counts = {label: sum(row["winner"] == label for row in rows) for label in (*MODES, "tie", "unconfirmed_128")}
        write_json(args.output / "summary.json",
                   {"counts": counts, "workloads": len(rows), "pairs": PAIRS, "tie_fraction": TIE_FRACTION})
        print(f"GRF comparison: {counts}", flush=True)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(f"\nGRF comparison ({args.manifest.name}): {counts}; {len(rows)} workloads.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    run(args)


if __name__ == "__main__":
    main()
