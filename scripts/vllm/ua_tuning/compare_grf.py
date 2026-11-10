# SPDX-License-Identifier: Apache-2.0
"""Compare compiler-default GRF selection with forced 256, before and after tuning."""

import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import os
import random
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from tune_ci import allocate_ci, load_tuner

MODES = ("default", "256")
PAIRS = 6
TIE_FRACTION = 0.03


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def mode_candidates(tuner, key):
    # None makes AttentionConfig.launch_kwargs omit the GRF override entirely.
    return [{**config, "grf_mode": None if config["grf_mode"] == "default" else config["grf_mode"]}
            for config in tuner.candidate_configs(key)]


def requested_mode(config):
    return config["grf_mode"] or "default"


def launch_settings(config):
    return {name: value for name, value in config.items() if name != "grf_mode"}


def select_winners(options):
    winners = {}
    for mode in MODES:
        valid = [
            option for option in options
            if requested_mode(option["config"]) == mode and option["status"] == "ok" and option["samples_ms"]
        ]
        if not valid:
            raise RuntimeError(f"No correct, launchable {mode}-GRF candidate")
        winners[mode] = min(valid, key=lambda option: (statistics.median(option["samples_ms"]), option["id"]))
    return winners


def summarize(samples):
    if any(len(samples[mode]) != PAIRS for mode in MODES):
        raise RuntimeError(f"Expected {PAIRS} paired samples per mode")
    if not all(math.isfinite(value) and value > 0 for mode in MODES for value in samples[mode]):
        raise RuntimeError("Invalid winner timing")
    medians = {mode: statistics.median(samples[mode]) for mode in MODES}
    winner = "tie"
    for mode, other in (MODES, tuple(reversed(MODES))):
        reductions = [1 - fast / slow for fast, slow in zip(samples[mode], samples[other])]
        if 1 - medians[mode] / medians[other] > TIE_FRACTION and sum(
                value > TIE_FRACTION for value in reductions) >= PAIRS - 1:
            winner = mode
    return {
        "median_ms": medians, "samples_ms": samples, "winner": winner,
        "latency_reduction_fraction_256": 1 - medians["256"] / medians["default"]
    }


def comparison_groups(winners, options):
    groups = {"independent_winners": {mode: winners[mode]["config"] for mode in MODES}}
    missing = {}
    if launch_settings(groups["independent_winners"]["default"]) == launch_settings(
            groups["independent_winners"]["256"]):
        return groups, missing
    for anchor in MODES:
        matches = [option for option in options
                   if launch_settings(option["config"]) == launch_settings(winners[anchor]["config"])]
        by_mode = {requested_mode(option["config"]): option for option in matches}
        name = f"matched_{anchor}_winner"
        if all(mode in by_mode and by_mode[mode]["status"] == "ok" for mode in MODES):
            groups[name] = {mode: by_mode[mode]["config"] for mode in MODES}
        else:
            missing[name] = {mode: by_mode[mode]["status"] if mode in by_mode else "not_in_grid" for mode in MODES}
    return groups, missing


def same_binaries(native):
    return {name: data["sha256"] for name, data in native["default"].items()} == {
        name: data["sha256"] for name, data in native["256"].items()
    }


def confirm_result(result, confirmation):
    if confirmation["native"] != result["native"]:
        raise RuntimeError("Fresh-process native binaries differ from frozen winners")
    result["confirmation"] = confirmation
    if result["winner"] in MODES and confirmation["winner"] != result["winner"]:
        result["winner"] = f"unconfirmed_{result['winner']}"
    return result


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
        from triton.runtime.autotuner import OutOfResources

        self.resource_error = OutOfResources
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
        option["native"] = {name: native_metadata(code) for name, code in self.compiled.items()}
        kernels = option["native"]["kernel_unified_attention"]["kernels"]
        mode = requested_mode(option["config"])
        if not kernels or any(not kernel["grf_count"] for kernel in kernels):
            raise RuntimeError(f"Missing native GRF count: {kernels}")
        if mode == "256" and any(kernel["grf_count"] != 256 for kernel in kernels):
            raise RuntimeError(f"Requested 256 GRFs, native metadata is {kernels}")
        launch_kwargs = self.tuner.runtime.AttentionConfig(**option["config"]).launch_kwargs("xpu")
        if mode == "default" and "grf_mode" in launch_kwargs:
            raise RuntimeError(f"Default arm must omit grf_mode: {launch_kwargs}")
        option["launch_kwargs"] = launch_kwargs
        try:
            self.check()
        except AssertionError as error:
            option.update(status="correctness_error", error=str(error))
            return
        option["correctness"] = {"passed": True, "atol": self.atol, "rtol": self.rtol}
        if binary_dir is not None:
            for code in self.compiled.values():
                raw = code.asm["zebin"]
                (binary_dir / f"{hashlib.sha256(raw).hexdigest()}.zebin").write_bytes(raw)

    def time(self, fn, device):
        value = self.timer(fn, device, warmup_ms=self.tuner.WARMUP_MS, rep_ms=self.tuner.REP_MS)
        self.check()
        return value

    def compare(self, configs, directory, reverse=False):
        directory.mkdir(parents=True, exist_ok=True)
        samples = {mode: [] for mode in MODES}
        native = {}
        for mode in MODES:
            option = {"id": mode, "config": configs[mode], "status": "pending", "samples_ms": []}
            self.prepare(option, self.inputs, self.device, binary_dir=directory)
            if option["status"] != "ok":
                raise RuntimeError(f"Frozen winner no longer launches: {option}")
            native[mode] = option["native"]
        for pair in range(PAIRS):
            for mode in MODES if (pair + reverse) % 2 == 0 else reversed(MODES):
                config = self.tuner.runtime.AttentionConfig(**configs[mode])
                with self.tuner.runtime.override_config(config):
                    samples[mode].append(self.time(lambda: self.tuner.unified_attention(**self.inputs), self.device))
        result = {
            **summarize(samples), "native": native, "identical_binaries": same_binaries(native),
            "same_launch_settings": launch_settings(configs["default"]) == launch_settings(configs["256"])
        }
        if result["identical_binaries"]:
            result.update(winner="tie", reason="identical_binaries")
        return result

    def search(self, options, directory, seed, rounds):
        generator = random.Random(seed)
        for round_index in range(rounds):
            order = [option for option in options if option["status"] == "ok"]
            generator.shuffle(order)
            for option in order:
                config = self.tuner.runtime.AttentionConfig(**option["config"])
                try:
                    with self.tuner.runtime.override_config(config):
                        value = self.time(lambda: self.tuner.unified_attention(**self.inputs), self.device)
                    option["samples_ms"].append(value)
                except AssertionError as error:
                    option.update(status="correctness_error", error=str(error))
                except self.resource_error as error:
                    option.update(status="resource_error", error=str(error))
                    self.tuner.synchronize(self.device)
            write_json(directory / "search.json", options)
            print(f"Search round {round_index + 1}/{rounds}: {len(order)} candidates", flush=True)


def write_reports(output, records, requested):
    rows = []
    for record in records:
        for name, result in record.get("comparisons", {}).items():
            rows.append({
                "workload": record["workload"], "comparison": name, "winner": result["winner"],
                "identical_binaries": result["identical_binaries"],
                "latency_default_us": result["median_ms"]["default"] * 1000,
                "latency_256_us": result["median_ms"]["256"] * 1000,
                "latency_reduction_fraction_256": result["latency_reduction_fraction_256"],
                "correctness_exclusions": record["correctness_exclusions"]
            })
    if rows:
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    primary = [row for row in rows if row["comparison"] == "independent_winners"]
    summary = {
        "requested_workloads": requested, "processed_workloads": len(records), "completed_comparisons": len(primary),
        "counts": dict(Counter(row["winner"] for row in primary)),
        "counts_by_comparison": {
            name: dict(Counter(row["winner"] for row in rows if row["comparison"] == name))
            for name in sorted({row["comparison"] for row in rows})
        },
        "statuses": dict(Counter(record["status"] for record in records)),
        "correctness_exclusions": sum(record["correctness_exclusions"] for record in records),
        "pairs_per_process": PAIRS, "tie_fraction": TIE_FRACTION,
        "scope": "Finite B580 CI-input grid; passing candidates only; each workload tuned independently"
    }
    write_json(output / "workloads.json", records)
    write_json(output / "summary.json", summary)
    return summary


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if args.replay:
        workloads = [json.loads(args.replay.read_text())["case"]]
    else:
        workloads = json.loads(args.manifest.read_text())
    records = []
    if not workloads:
        write_reports(args.output, records, 0)
        return
    runner = Runner(args.device)
    with runner.torch.inference_mode():
        if args.replay:
            frozen = json.loads(args.replay.read_text())
            runner.allocate(frozen["case"], args.seed)
            confirmations = {
                name: runner.compare(configs, args.output / name, reverse=True)
                for name, configs in frozen["comparisons"].items()
            }
            write_json(args.output / "confirmation.json", confirmations)
            return
        for index, case in enumerate(workloads):
            directory = args.output / case["id"]
            directory.mkdir()
            print(f"Workload {index + 1}/{len(workloads)}: {case['id']}", flush=True)
            options = []
            record = {"workload": case["id"], "status": "pending", "correctness_exclusions": 0}
            try:
                runner.allocate(case, args.seed + index)
                key = runner.tuner.runtime.make_attention_key(**runner.inputs)
                configs = mode_candidates(runner.tuner, key)
                random.Random(args.seed + index).shuffle(configs)
                for config in configs:
                    if not runner.tuner.runtime.validate_config(runner.tuner.runtime.AttentionConfig(**config), key,
                                                                "xpu"):
                        continue
                    option = {"id": runner.tuner.config_id(config), "config": config, "status": "pending",
                              "samples_ms": []}
                    options.append(option)
                    runner.prepare(option, runner.inputs, runner.device)
                    if len(options) % 32 == 0:
                        write_json(directory / "search.json", options)
                        print(f"Prepared {len(options)} candidates", flush=True)
                runner.search(options, directory, args.seed + index, args.rounds)
                available = {requested_mode(option["config"]) for option in options
                             if option["status"] == "ok" and option["samples_ms"]}
                if available != set(MODES):
                    record.update(status="unavailable", missing_modes=sorted(set(MODES) - available))
                    continue
                winners = select_winners(options)
                groups, missing = comparison_groups(winners, options)
                frozen = {"case": case, "key": asdict(key), "comparisons": groups,
                          "input_seed": 20, "search_seed": args.seed + index}
                write_json(directory / "frozen.json", frozen)
                results = {name: runner.compare(configs, directory / name) for name, configs in groups.items()}
                write_json(directory / "initial.json", results)
                # Free the parent buffers before the confirmation process allocates them.
                runner.release()
                subprocess.run([
                    sys.executable,
                    str(Path(__file__).resolve()), "--replay",
                    str(directory / "frozen.json"), "--device", args.device, "--seed",
                    str(args.seed + index), "--output",
                    str(directory / "fresh")
                ], check=True)
                confirmations = json.loads((directory / "fresh/confirmation.json").read_text())
                results = {name: confirm_result(result, confirmations[name]) for name, result in results.items()}
                record.update(status="completed", comparisons=results, matched_unavailable=missing)
            except Exception as error:  # noqa: BLE001 - Preserve the rest of the workload manifest.
                record.update(status="error", error=f"{type(error).__name__}: {error}")
                print(f"Workload failed: {record['error']}", flush=True)
            finally:
                record["correctness_exclusions"] = sum(option["status"] == "correctness_error" for option in options)
                record["candidate_statuses"] = {
                    mode: dict(Counter(option["status"] for option in options
                                       if requested_mode(option["config"]) == mode)) for mode in MODES
                }
                write_json(directory / "search.json", options)
                write_json(directory / "comparison.json", record)
                records.append(record)
                summary = write_reports(args.output, records, len(workloads))
                runner.release()
        print(f"GRF comparison: {summary}", flush=True)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(f"\nDefault vs 256 ({args.manifest.name}): `{json.dumps(summary)}`\n")
        if summary["correctness_exclusions"] or summary["statuses"].get("error"):
            raise RuntimeError("Comparison finished with errors; inspect summary.json and per-workload records")


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
