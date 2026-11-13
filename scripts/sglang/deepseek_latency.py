#!/usr/bin/env python3
"""DeepSeek latency on XPU, Triton vs SYCL (issue #8453).

Runs sglang.benchmark.one_batch --repeat times per model and backend, then once more
with the XPU profiler on, and prints a markdown table per model: prefill and decode
latency, Triton next to SYCL, and the top kernels of each with their time and number of
calls. Report only: a metric no run produced is shown as "failed", and a failed run
makes the exit code non-zero; a slow one does not.

  triton: --attention-backend triton --moe-runner-backend triton
  sycl:   --attention-backend intel_xpu (sgl-kernel-xpu attention and MoE)

usage: deepseek_latency.py --out DIR [--models v2-lite,ocr] [--backends triton,sycl]
"""

import argparse
import collections
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time

MODELS = {
    "v2-lite": "gaunernst/DeepSeek-V2-Lite-Chat-FP8",
    "ocr": "deepseek-ai/DeepSeek-OCR",
}
BACKENDS = {
    "triton": ["--attention-backend", "triton", "--moe-runner-backend", "triton"],
    "sycl": ["--attention-backend", "intel_xpu"],
}
BATCH_SIZES = [1, 16]
INPUT_LEN = 1024
OUTPUT_LEN = 32
COMMON = [
    # bs16 twice: the warmup runs only the first batch size, so the first bs16 run
    # includes the JIT compiles. The table takes the best result per batch size.
    "--batch-size", "1", "16", "16",
    "--input-len", str(INPUT_LEN),
    "--output-len", str(OUTPUT_LEN),
    "--mem-fraction-static", "0.7",
    "--trust-remote-code",
    "--device", "xpu",
]  # yapf: disable
PROFILE_STEPS = 8
# The library each backend brings; the other kernels get their library next to the name.
OWN_LIBRARY = {"triton": "Triton", "sycl": "sgl-kernel-xpu"}
METRICS = [("prefill", "prefill_latency"), ("decode step", "median_decode_latency")]
NAMES = {"triton": "Triton", "sycl": "SYCL"}


def run(model: str, backend: str, out: str, timeout: int, profile: bool = False) -> bool:
    name = f"{model}_{backend}"
    cmd = [
        sys.executable, "-m", "sglang.benchmark.one_batch", "--model-path", MODELS[model], *COMMON, *BACKENDS[backend],
        "--run-name", name, "--result-filename",
        os.path.join(out, "profile.jsonl" if profile else "results.jsonl")
    ]
    env = os.environ.copy()
    if profile:
        # A separate run, so the profiler does not slow down the one that is timed.
        cmd += [
            "--profile", "--profile-activities", "XPU", "--profile-steps",
            str(PROFILE_STEPS), "--profile-prefix", name
        ]
        env["SGLANG_TORCH_PROFILER_DIR"] = os.path.join(out, "traces")
    log = f"{name}_profile.log" if profile else f"{name}.log"
    print("+", " ".join(cmd), flush=True)
    rc: int | str
    # Appended, so the log keeps all repeats.
    with open(os.path.join(out, log), "a", encoding="utf-8") as f:
        try:
            # cwd is not the SGLang checkout: its ./sglang would shadow the package.
            rc = subprocess.run(cmd, cwd=out, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                                check=False).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    print(f"{log}: rc={rc}", flush=True)
    return rc == 0


def load_results(out: str) -> dict[tuple[str, int], dict[str, float]]:
    """The best (lowest) value of each metric per run name and batch size, over all repeats."""
    results: dict[tuple[str, int], dict[str, float]] = collections.defaultdict(dict)
    path = os.path.join(out, "results.jsonl")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # cut off by a killed run
                best = results[(r["run_name"], r["batch_size"])]
                for _, key in METRICS:
                    if key in r:
                        best[key] = min(best.get(key, r[key]), r[key])
    return results


def short(name: str) -> str:
    name = name.replace("void __sycl_kernel_", "").replace("(anonymous namespace)::", "")
    base, _, args = name.partition("<")
    if base == "KernelCur" and args:
        # SYCL-TLA wrapper, the kernel itself is the first template argument.
        base = args.split("<")[0]
    return base.split("::")[-1][:48]


def library(event: dict) -> str:
    name, src = event["name"], event.get("args", {}).get("source_file_name", "")
    if name.startswith("triton_"):
        return "Inductor"
    if src == "main.cpp":
        # The launcher Triton compiles for each kernel.
        return "Triton"
    if "sgl-kernel-xpu" in src:
        return "sgl-kernel-xpu"
    if name == "gemm_kernel":
        return "oneDNN"
    return "torch"


def kernels(out: str, model: str, backend: str, bs: int, key: str) -> list[tuple[str, float, float]] | None:
    """(kernel, ms, calls) per prefill or per decode step, most time first. The kernel name says its library
    when that is not the backend's own."""
    stage, steps = ("prefill", 1) if key == "prefill_latency" else ("decode", PROFILE_STEPS)
    path = os.path.join(out, "traces",
                        f"{model}_{backend}_batch{bs}_input{INPUT_LEN}_output{OUTPUT_LEN}_{stage}.trace.json.gz")
    try:
        with gzip.open(path, "rt") as f:
            events = json.load(f)["traceEvents"]
    except (OSError, EOFError, ValueError, KeyError):
        return None  # missing, or cut off by a killed run
    dur: dict[str, float] = collections.defaultdict(float)
    calls: dict[str, int] = collections.defaultdict(int)
    for e in events:
        if e.get("ph") == "X" and e.get("cat") == "kernel":
            lib = library(e)
            name = f"`{short(e['name'])}`" + ("" if lib == OWN_LIBRARY.get(backend) else f" ({lib})")
            dur[name] += e.get("dur", 0)
            calls[name] += 1
    return sorted(((n, t / 1e3 / steps, calls[n] / steps) for n, t in dur.items()), key=lambda k: -k[1])


def ms(v: float) -> str:
    return f"{v:.2f} ms" if v < 1 else f"{v:.1f} ms" if round(v, 1) < 100 else f"{v:.0f} ms"


def count(c: float) -> str:
    # Per decode step. Kernels of the neighbour steps can fall into the profiled window, so it is not exact.
    return f"{c:.1f}" if c < 1 else f"{c:.0f}"


def rows_needed(ks: list[tuple[str, float, float]] | None, others_pct: float) -> int:
    """Rows for the kernels until the smallest ones, together under `others_pct` of the kernel time, are left
    for one others row."""
    if ks is None:
        return 1
    shown, rest = len(ks), 0.0
    while shown > 0 and rest + ks[shown - 1][1] < others_pct / 100 * sum(t for _, t, _ in ks):
        shown -= 1
        rest += ks[shown][1]
    return shown + (shown < len(ks))


def kernel_rows(ks: list[tuple[str, float, float]] | None, n: int) -> list[list[str]]:
    """n [kernel, time, calls] rows, most time first; when the kernels do not fit, the last row sums up the
    rest as others."""
    if ks is None:
        return [["-", "-", "-"]] + [["", "", ""]] * (n - 1)
    shown = len(ks) if len(ks) <= n else n - 1
    rows = [[k, ms(t), count(c)] for k, t, c in ks[:shown]]
    if shown < len(ks):
        rest = ks[shown:]
        rows.append([f"{len(rest)} others", ms(sum(t for _, t, _ in rest)), count(sum(c for _, _, c in rest))])
    return rows + [["", "", ""]] * (n - len(rows))


def host_row(ks: list[tuple[str, float, float]] | None, latency: float | None) -> list[str]:
    """The part of the latency that no kernel covers, the time the GPU waits for the host."""
    if ks is None or latency is None:
        return ["host side (GPU idle)", "-", "-"]
    return ["host side (GPU idle)", ms(max(0.0, latency - sum(t for _, t, _ in ks))), "-"]


def table(out: str, models: list[str], backends: list[str], others_pct: float, repeat: int) -> str:
    """Per model and metric: the latency of each backend, then its kernels, most time first, one per row."""
    results = load_results(out)
    ratio = backends == ["triton", "sycl"]

    def block(model: str, bs: int, metric: str, key: str) -> list[str]:
        vals = [results.get((f"{model}_{b}", bs), {}).get(key) for b in backends]
        vals = [None if v is None else v * 1000 for v in vals]
        found = [kernels(out, model, b, bs, key) for b in backends]
        # Every column gets the rows the longest one needs, so a shorter one lists more kernels, and the host
        # side comes last in all of them.
        n = max(rows_needed(k, others_pct) for k in found)
        ks = [kernel_rows(f, n) + [host_row(f, v)] for f, v in zip(found, vals)]
        rows = []
        for i in range(n + 1):
            cells = [
                (f"prefill {bs}x{INPUT_LEN}" if key == "prefill_latency" else f"{metric}, bs{bs}") if i == 0 else ""
            ]
            cells += [(("failed" if v is None else ms(v)) if i == 0 else "") for v in vals]
            if ratio:
                t, s = vals
                cells.append(("-" if t is None or s is None else f"{t / s:.1f}x") if i == 0 else "")
            for k in ks:
                cells += k[i]
            rows.append("| " + " | ".join(cells) + " |")
        return rows

    lines = [f"- {NAMES[b]} = `{' '.join(BACKENDS[b])}`" for b in backends]
    lines.append(f"- latency: the best of {repeat} runs" if repeat > 1 else "- latency: a single run")
    lines.append(
        f"- kernels: XPU time and calls per prefill and per decode step, from a separate profiled run "
        f"({PROFILE_STEPS} decode steps); the library is given when it is not the backend's own; the smallest "
        f"kernels, together under {others_pct:g}% of the kernel time, are summed up as others; host side is the "
        f"latency minus the kernel time")
    head = ["metric"] + [NAMES[b] for b in backends] + (["Triton slower"] if ratio else [])
    for b in backends:
        head += [f"{NAMES[b]} kernel", "time", "calls"]
    for model in models:
        lines += [
            "", f"**{MODELS[model].split('/')[-1]}**", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)
        ]
        for bs in BATCH_SIZES:
            for metric, key in METRICS:
                lines += block(model, bs, metric, key)
    return "\n".join(lines) + "\n"


VERSIONS = """
import importlib.metadata as md
import torch
def version(package):
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return "not installed"
print(torch.xpu.get_device_name())
for package in ("torch", "triton", "sglang", "sglang-kernel-xpu"):
    print(version(package))
"""


def first_line(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    except OSError:
        return ""
    return out.splitlines()[0].strip() if out.strip() else ""


def pin(name: str) -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if not os.path.exists(path):
        return "?"
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def environment() -> tuple[str, str]:
    """The device name and a one-line description of the stack."""
    # In a child process, so this one holds no XPU context while the runs need the memory.
    out = subprocess.run([sys.executable, "-c", VERSIONS], capture_output=True, text=True, check=False).stdout
    device, torch_v, triton_v, sglang_v, sglk_v = [v or "?" for v in (out.splitlines() + [""] * 5)[:5]]
    stack = [
        f"torch {torch_v}", f"Triton {triton_v}", f"SGLang {sglang_v} at {pin('sglang-pin.txt')[:10]} + "
        "`sglang-test-fix.patch`", f"sgl-kernel-xpu {sglk_v} at {pin('sgl-kernel-xpu-pin.txt')[:7]}"
    ]
    icpx = re.search(r"Compiler (\S+)", first_line(["icpx", "--version"]))
    if icpx:
        stack.append(f"icpx {icpx.group(1)}")
    neo = first_line(["dpkg-query", "-W", "-f", "${Version}\n", "intel-opencl-icd"]).split("-")[0]
    igc = first_line(["dpkg-query", "-W", "-f", "${Version}\n", "libigc2"]).split("-")[0]
    if neo or igc:
        stack.append(f"NEO {neo or '?'} / IGC {igc or '?'}")
    return "XPU" if device == "?" else device, ", ".join(stack)


def report(out: str, models: list[str], backends: list[str], others_pct: float, repeat: int) -> str:
    """The whole markdown report: what was measured on which stack, the tables, and how it was run."""
    device, stack = environment()
    script = os.path.relpath(os.path.abspath(__file__))
    times = f"{repeat} times" if repeat > 1 else "once"
    return (f"### DeepSeek E2E on {device}: {' vs '.join(NAMES[b] for b in backends)}\n\n"
            f"- stack: {stack}\n" + table(out, models, backends, others_pct, repeat) + "\nHow it was run:\n\n"
            f"    python {script} {' '.join(sys.argv[1:])}\n\n"
            f"which runs, per model ({', '.join(f'`{MODELS[m]}`' for m in models)}) and backend, {times}:\n\n"
            f"    python -m sglang.benchmark.one_batch --model-path <model> {' '.join(COMMON)} <backend flags>\n\n"
            f"with the backend flags above, and then once more with `--profile --profile-activities XPU "
            f"--profile-steps {PROFILE_STEPS}` for the kernels.\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="directory for the results, logs, traces and summary.md")
    p.add_argument("--models", default=",".join(MODELS), help="comma list of: " + ", ".join(MODELS))
    p.add_argument("--backends", default=",".join(BACKENDS), help="comma list of: " + ", ".join(BACKENDS))
    p.add_argument("--others-pct", type=float, default=5.0,
                   help="sum up the smallest kernels as others while they stay under this share of the kernel time")
    p.add_argument("--repeat", type=int, default=3, help="timed runs per model and backend, the best one is reported")
    p.add_argument("--timeout", type=int, default=1800, help="seconds per run")
    p.add_argument("--budget", type=int, default=2700,
                   help="seconds for all runs; the runs left when it is spent are skipped and the report still written")
    a = p.parse_args()
    models, backends = a.models.split(","), a.backends.split(",")
    for name, known in ((models, MODELS), (backends, BACKENDS)):
        if not set(name) <= set(known):
            p.error(f"unknown {sorted(set(name) - set(known))}, choose from {', '.join(known)}")
    # Absolute, because the runs use it as their cwd.
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    # The results of an earlier run in the same directory would mix into the best of.
    for old in ("results.jsonl", "profile.jsonl"):
        if os.path.exists(os.path.join(out, old)):
            os.remove(os.path.join(out, old))
    shutil.rmtree(os.path.join(out, "traces"), ignore_errors=True)

    deadline = time.monotonic() + a.budget

    def timed(m: str, b: str, profile: bool = False) -> bool:
        left = int(deadline - time.monotonic())
        if left <= 0:
            print(f"{m}_{b}{'_profile' if profile else ''}: skipped, --budget spent", flush=True)
            return False
        return run(m, b, out, min(a.timeout, left), profile)

    # Interleaved, so a slow phase of the machine does not hit only one backend.
    ok = [timed(m, b) for _ in range(a.repeat) for m in models for b in backends]
    ok += [timed(m, b, profile=True) for m in models for b in backends]
    summary = report(out, models, backends, a.others_pct, a.repeat)
    # Marked off in the log, so it can be copied from there as it is.
    print("==== DeepSeek report (markdown, also in the job summary and in summary.md) ====")
    print(summary)
    print("==== end of the DeepSeek report ====")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(summary)
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
