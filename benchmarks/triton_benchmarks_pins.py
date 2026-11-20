"""Pin autotuning decisions, so that benchmark runs can skip autotuning.

A full run (``AUTOTUNE_PINS_MODE=record``) autotunes as usual and records every decision together
with the timings of all candidates. ``triton-benchmarks-pins update`` turns the records of one or
more full runs into pin files, one per GPU and benchmark. A pinned run (``AUTOTUNE_PINS_MODE=pinned``)
gives every autotuner only its pinned config, so each autotune key compiles one config and benchmarks
none. A key without a pin, or a pin that is no longer among the candidates, is autotuned so that the
run still measures everything, and then fails the run. ``triton-benchmarks-pins show`` prints the pin
file behind a ``pinned@<version>`` report value.

Covered: ``triton.autotune`` kernels and Inductor's autotuning (``AlgorithmSelectorCache``, e.g. flex
attention and grouped GEMM templates). Not covered: Inductor's runtime autotuning of its generated
pointwise and reduction kernels.

Environment:
  AUTOTUNE_PINS_MODE     ``record``, ``pinned``, or ``auto`` (pinned if the benchmark has a pin file,
                         otherwise record). Unset: inactive.
  AUTOTUNE_PINS_DIR      pin files root: ``<dir>/<device id>/<benchmark>.json``.
  AUTOTUNE_PINS_RECORDS  where full runs append their decisions: ``<dir>/<benchmark>.jsonl``.
  AUTOTUNE_PINS_DEVICE   device id override; default derived from the XPU name, e.g. ``b580``.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.abc
import json
import math
import os
import re
import statistics
import subprocess
import sys
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

FORMAT_VERSION = 1
MODES = ("record", "pinned", "auto")


class AutotunePinsError(RuntimeError):
    pass


def device_id(device_name: str) -> str:
    """Short file-name-safe GPU id: 'Intel(R) Arc(TM) B580 Graphics' -> 'b580'."""
    name = re.sub(r"\((R|TM|C)\)", " ", device_name, flags=re.IGNORECASE)
    name = re.sub(r"\b(intel|graphics|data center gpu|arc)\b", " ", name, flags=re.IGNORECASE)
    return re.sub(r"[^0-9a-zA-Z]+", "-", name).strip("-").lower() or "unknown"


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _finite_or_none(value: Any) -> Optional[float]:
    if isinstance(value, (list, tuple)):  # triton do_bench with quantiles: median first
        value = value[0] if value else None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def triton_config_dict(config) -> Dict[str, Any]:
    return _canonical(config.all_kwargs())


def inductor_choice_id(choice) -> str:
    # The description lists the template parameters. The generated code is deliberately not part of
    # the id: it changes with Inductor's codegen, and masks or score mods are told apart by the
    # benchmark point instead.
    description = getattr(choice, "description", "")
    return f"{type(choice).__name__}: {description}"


def _triton_key(autotuner, kwargs) -> str:
    # Same key as triton.runtime.autotuner.Autotuner.run.
    all_args = {**autotuner.nargs, **kwargs}
    args = {name: value for name, value in all_args.items() if name in autotuner.arg_names}
    key = [args[name] for name in autotuner.keys if name in args]
    key += [str(arg.dtype) for arg in args.values() if hasattr(arg, "dtype")]
    return str(tuple(key))


def _function_name(fn) -> str:
    return f"{fn.__module__}.{fn.__qualname__}"


def _inputs_key(module, input_nodes) -> str:
    try:
        return module.create_inputs_key(input_nodes)
    except Exception:  # pylint: disable=broad-except
        return ""


@dataclass
class PinFile:
    benchmark: str
    device: str
    triton: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict)
    inductor: Dict[Tuple[str, str, str], str] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        triton = [{"kernel": k, "key": key, "config": cfg} for (k, key), cfg in sorted(self.triton.items())]
        inductor = [{"name": n, "point": p, "inputs": i, "choice": c} for (n, p, i), c in sorted(self.inductor.items())]
        return {
            "format": FORMAT_VERSION,
            "benchmark": self.benchmark,
            "device": self.device,
            "triton": triton,
            "inductor": inductor,
        }

    @property
    def version(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def load(cls, path: Path) -> PinFile:
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")), str(path))

    @classmethod
    def from_json(cls, data: Dict[str, Any], where: str) -> PinFile:
        version = data.get("format")
        if version != FORMAT_VERSION:
            raise AutotunePinsError(f"{where}: unsupported pin file format {version!r}")
        triton = {(e["kernel"], e["key"]): e["config"] for e in data["triton"]}
        inductor = {(e["name"], e["point"], e["inputs"]): e["choice"] for e in data["inductor"]}
        return cls(benchmark=data["benchmark"], device=data["device"], triton=triton, inductor=inductor)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1, sort_keys=True) + "\n", encoding="utf-8")


def pin_path(pins_dir: Path, device: str, bench: str) -> Path:
    return pins_dir / device / f"{bench}.json"


class _Session:  # pylint: disable=too-many-instance-attributes
    """Process-wide state: the benchmark being run, its pins or record file, and the outcome."""

    def __init__(self) -> None:
        self.mode = os.getenv("AUTOTUNE_PINS_MODE", "")
        if self.mode and self.mode not in MODES:
            raise AutotunePinsError(f"AUTOTUNE_PINS_MODE must be one of {MODES}, got {self.mode!r}")
        self.pins_dir = Path(os.getenv("AUTOTUNE_PINS_DIR", "autotune_pins"))
        records = os.getenv("AUTOTUNE_PINS_RECORDS", "")
        self.records_dir = Path(records) if records else None
        self._device: Optional[str] = os.getenv("AUTOTUNE_PINS_DEVICE") or None
        self.benchmark: Optional[str] = None
        self.pins: Optional[PinFile] = None  # set while a pinned benchmark runs
        self.record_path: Optional[Path] = None  # set while a recorded benchmark runs
        self.point = ""
        self.hits = 0
        self.recorded = 0
        self.misses: List[str] = []

    @property
    def device(self) -> str:
        if self._device is None:
            import torch  # pylint: disable=import-outside-toplevel
            self._device = device_id(torch.xpu.get_device_name())
        return self._device

    def write(self, entry: Dict[str, Any]) -> None:
        entry = {"benchmark": self.benchmark, "device": self.device, **entry}
        with open(self.record_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        self.recorded += 1

    def miss(self, what: str) -> None:
        self.misses.append(f"{self.benchmark}: {what}")
        print(f"[autotune-pins] {self.benchmark}: {what}; autotuning it, the run will fail", file=sys.stderr,
              flush=True)


_SESSION: Optional[_Session] = None
_INSTALLED = False
_AUTOTUNERS: weakref.WeakSet[Any] = weakref.WeakSet()  # Triton autotuners seen while a benchmark ran


def _session() -> Optional[_Session]:
    """The active session, created from the environment on first use; None when inactive."""
    global _SESSION  # pylint: disable=global-statement
    if _SESSION is None:
        _SESSION = _Session()
    return _SESSION if _SESSION.mode else None


def _running() -> Optional[_Session]:
    """The active session while a benchmark runs, else None; the hooks pass through then."""
    return _SESSION if _SESSION is not None and _SESSION.mode and _SESSION.benchmark else None


def _install_triton() -> None:
    from triton import knobs  # pylint: disable=import-outside-toplevel
    from triton.runtime import autotuner  # pylint: disable=import-outside-toplevel

    previous_listener = knobs.autotuning.listener

    def listener(fn, key, best_config, configs_timings, duration, cache_hit):
        if previous_listener is not None:
            previous_listener(fn=fn, key=key, best_config=best_config, configs_timings=configs_timings,
                              duration=duration, cache_hit=cache_hit)
        session = _running()
        if session is None or session.record_path is None:
            return
        timings = [[triton_config_dict(c), _finite_or_none(t)] for c, t in (configs_timings or {}).items()]
        session.write({
            "kind": "triton",
            "kernel": _function_name(fn.fn),
            "key": str(key),
            "best": triton_config_dict(best_config),
            "timings": timings,
            "seconds": duration,
        })

    knobs.autotuning.listener = listener
    original_prune = autotuner.Autotuner.prune_configs

    def prune_configs(self, kwargs):
        pruned = original_prune(self, kwargs)
        session = _running()
        if session is None:
            return pruned
        _AUTOTUNERS.add(self)
        if session.pins is None:
            return pruned
        kernel, key = _function_name(self.base_fn), _triton_key(self, kwargs)
        pinned = session.pins.triton.get((kernel, key))
        match = next((c for c in pruned if triton_config_dict(c) == pinned), None)
        if match is not None:
            session.hits += 1
            return [match]
        if len(pruned) > 1:  # a single config left after pruning is not benchmarked, so it needs no pin
            problem = "no pin" if pinned is None else "pinned config is not among the configs"
            session.miss(f"{problem} for {kernel} {key}")
        return pruned

    autotuner.Autotuner.prune_configs = prune_configs


def _install_inductor(module) -> None:
    original_call = module.AlgorithmSelectorCache.__call__

    def saver(timings, name, input_nodes, *_args, **_kwargs):
        session = _running()
        if session is None or session.record_path is None or not timings:
            return
        session.write({
            "kind": "inductor",
            "name": name,
            "point": session.point,
            "inputs": _inputs_key(module, input_nodes),
            "best": inductor_choice_id(min(timings, key=timings.get)),
            "timings": [[inductor_choice_id(c), _finite_or_none(t)] for c, t in timings.items()],
        })

    def call(self, name, choices, input_nodes, layout, *args, **kwargs):
        session = _running()
        pins = session.pins if session is not None else None
        if pins is not None and len(choices) > 1:
            pin_key = (name, session.point, _inputs_key(module, input_nodes))
            pinned = pins.inductor.get(pin_key)
            picked = [c for c in choices if inductor_choice_id(c) == pinned] if pinned is not None else []
            if picked:
                session.hits += 1
                choices = picked[:1]
            elif pinned is None:
                session.miss(f"no pin for Inductor {name} at {session.point}")
            else:
                session.miss(f"pinned choice of Inductor {name} at {session.point} is not among its choices")
        return original_call(self, name, choices, input_nodes, layout, *args, **kwargs)

    module.add_feedback_saver(saver)
    module.AlgorithmSelectorCache.__call__ = call


class _PostImportHook(importlib.abc.MetaPathFinder):
    """Runs a callback right after a module is imported, without importing it early."""

    def __init__(self, name, callback):
        self.name, self.callback = name, callback

    def find_spec(self, fullname, _path, _target=None):
        if fullname != self.name:
            return None
        sys.meta_path.remove(self)
        from importlib.util import find_spec  # pylint: disable=import-outside-toplevel
        spec = find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_hook(module):
            exec_module(module)
            self.callback(module)

        spec.loader.exec_module = exec_and_hook
        return spec


def _install() -> None:
    global _INSTALLED  # pylint: disable=global-statement
    if _INSTALLED:
        return
    _INSTALLED = True
    _install_triton()
    name = "torch._inductor.select_algorithm"
    if name in sys.modules:
        _install_inductor(sys.modules[name])
    else:
        sys.meta_path.insert(0, _PostImportHook(name, _install_inductor))


def _forget_cached_decisions() -> None:
    """Cached autotuning decisions skip the hooks, so every benchmark starts without them."""
    for tuner in _AUTOTUNERS:
        tuner.cache.clear()
    try:
        from torch._inductor import config  # pylint: disable=import-outside-toplevel
    except ImportError:
        return
    config.force_disable_caches = True  # a cached Inductor graph skips its template autotuning


@contextlib.contextmanager
def benchmark(key: str) -> Iterator[None]:
    """Applies the pins of benchmark ``key``, or records its autotuning, while the block runs."""
    session = _session()
    if session is None:
        yield
        return
    _install()
    _forget_cached_decisions()
    path = pin_path(session.pins_dir, session.device, key)
    if session.mode == "pinned" or (session.mode == "auto" and path.exists()):
        # Benchmarks that never autotune have no pin file; any key they do autotune is reported missing.
        session.pins = PinFile.load(path) if path.exists() else PinFile(key, session.device)
    else:
        if session.records_dir is None:
            raise AutotunePinsError("AUTOTUNE_PINS_RECORDS must be set to record autotuning")
        session.records_dir.mkdir(parents=True, exist_ok=True)
        session.record_path = session.records_dir / f"{key}.jsonl"
    session.benchmark = key
    hits, misses, recorded = session.hits, len(session.misses), session.recorded
    try:
        yield
    finally:
        if session.pins is not None:
            source = f"{path}@{session.pins.version}" if path.exists() else "no pin file"
            print(
                f"[autotune-pins] {key}: {session.hits - hits} pinned, {len(session.misses) - misses} missing ({source})",
                file=sys.stderr, flush=True)
        else:
            print(f"[autotune-pins] {key}: {session.recorded - recorded} recorded to {session.record_path}",
                  file=sys.stderr, flush=True)
        session.benchmark, session.pins, session.record_path = None, None, None


@contextlib.contextmanager
def benchmark_point(x_args: Dict[str, Any], provider: Any) -> Iterator[None]:
    """Names the shape and provider being measured; Inductor pins are keyed by it."""
    session = _session()
    if session is None:
        yield
        return
    previous = session.point
    session.point = json.dumps({**x_args, "provider": provider}, sort_keys=True, default=str)
    try:
        yield
    finally:
        session.point = previous


def report_value() -> Optional[str]:
    """Value of the ``autotune_pins`` report column for the running benchmark, None when inactive."""
    session = _session()
    if session is None or session.benchmark is None:
        return None
    if session.pins is None:
        return "full"
    return f"pinned@{session.pins.version}" if session.pins.triton or session.pins.inductor else "pinned"


def raise_for_misses() -> None:
    session = _session()
    if session is not None and session.misses:
        raise AutotunePinsError(f"{len(session.misses)} autotune pins missing or stale, refresh the pins:\n  " +
                                "\n  ".join(session.misses))


# Pin updates from records.


def _candidate(kind: str, value: Any) -> str:
    return json.dumps(value, sort_keys=True) if kind == "triton" else value


def _read_records(directory: Path) -> Dict[Tuple[str, str], Dict[Tuple[str, ...], Dict[str, Any]]]:
    """(device, benchmark) -> pin key -> first record of that key in one full run."""
    runs: Dict[Tuple[str, str], Dict[Tuple[str, ...], Dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry["kind"] == "triton":
                pin_key = ("triton", entry["kernel"], entry["key"])
            else:
                pin_key = ("inductor", entry["name"], entry["point"], entry["inputs"])
            runs.setdefault((entry["device"], entry["benchmark"]), {}).setdefault(pin_key, entry)
    return runs


@dataclass
class UpdateSummary:
    benchmark: str
    device: str
    new: int = 0  # key had no pin
    kept: int = 0
    changed: List[str] = field(default_factory=list)  # pin moved to a faster config
    replaced: int = 0  # pinned config is no longer a candidate
    dropped: int = 0  # key no longer autotuned


def _choose(kind: str, pin_key: Tuple[str, ...], runs: List[Dict[str, Any]], pinned: Optional[str], min_gain: float,
            summary: UpdateSummary) -> Any:
    """Picks the pin of one key from its records in several runs, the newest last."""
    values: Dict[str, Any] = {}
    times: Dict[str, List[float]] = {}
    for entry in runs:
        for value, time in entry["timings"]:
            cid = _candidate(kind, value)
            values[cid] = value
            if time is not None:
                times.setdefault(cid, []).append(time)
    # A config that fails in the newest run is out, whatever its older timings.
    candidates = {_candidate(kind, value) for value, time in runs[-1]["timings"] if time is not None}
    median = {cid: statistics.median(t) for cid, t in times.items() if cid in candidates}
    if not median:
        return None
    # On a tie, keep what the autotuner itself picked in the newest run.
    newest_best = _candidate(kind, runs[-1]["best"])
    best = min(median, key=lambda cid: (median[cid], cid != newest_best))
    if pinned is None:
        summary.new += 1
        return values[best]
    if pinned not in median:
        summary.replaced += 1
        return values[best]
    spread = (max(times[pinned]) - min(times[pinned])) / median[pinned]
    gain = 1.0 - median[best] / median[pinned]
    if gain > max(min_gain, spread):
        # Triton: kernel and key; Inductor: op and benchmark point (the input layouts are long).
        name = pin_key[1].rsplit(".", 1)[-1]
        summary.changed.append(f"{name} {pin_key[2]}: {gain:.1%} faster")
        return values[best]
    summary.kept += 1
    return values[pinned]


def update_pins(record_dirs: List[Path], pins_dir: Path, output_dir: Path, min_gain: float) -> List[UpdateSummary]:
    """Writes new pin files for every benchmark in the newest record dir; older dirs add history.

    Each candidate is rated by its median time over the runs. A pin moves to the best candidate only
    when that is faster by more than ``min_gain`` and more than the pin's own spread across the runs.
    """
    history = [_read_records(d) for d in record_dirs]
    summaries = []
    for (device, bench), newest in sorted(history[-1].items()):
        old_path = pin_path(pins_dir, device, bench)
        old = PinFile.load(old_path) if old_path.exists() else PinFile(bench, device)
        new = PinFile(bench, device)
        summary = UpdateSummary(bench, device)
        for pin_key in sorted(newest):
            kind, key = pin_key[0], pin_key[1:]
            runs = [run[(device, bench)][pin_key] for run in history if pin_key in run.get((device, bench), {})]
            old_value = old.triton.get(key) if kind == "triton" else old.inductor.get(key)
            pinned = _candidate(kind, old_value) if old_value is not None else None
            value = _choose(kind, pin_key, runs, pinned, min_gain, summary)
            if value is None:
                continue
            if kind == "triton":
                new.triton[key] = value
            else:
                new.inductor[key] = value
        summary.dropped = len(set(old.triton) - set(new.triton)) + len(set(old.inductor) - set(new.inductor))
        new.save(pin_path(output_dir, device, bench))
        summaries.append(summary)
    return summaries


def _format_summary(summaries: List[UpdateSummary]) -> str:
    lines = ["| device | benchmark | new | kept | changed | replaced | dropped |", "|---|---|---:|---:|---:|---:|---:|"]
    lines += [
        f"| {s.device} | {s.benchmark} | {s.new} | {s.kept} | {len(s.changed)} | {s.replaced} | {s.dropped} |"
        for s in summaries
    ]
    changes = [f"- {s.device}/{s.benchmark} {c}" for s in summaries for c in s.changed]
    return "\n".join(lines + ([""] + changes if changes else [])) + "\n"


# Looking up the pin file behind a report value.


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def _parse(text: str, where: str) -> Optional[PinFile]:
    try:
        return PinFile.from_json(json.loads(text), where)
    except (ValueError, KeyError, TypeError, AttributeError, AutotunePinsError):
        return None


def find_pin_files(pins_dir: Path, version: str) -> List[Tuple[str, PinFile]]:
    """(location, pin file) of every pin file with this version in ``pins_dir`` and its git history."""
    found = []
    for path in sorted(pins_dir.glob("*/*.json")):
        pin_file = _parse(path.read_text(encoding="utf-8"), str(path))
        if pin_file is not None and pin_file.version == version:
            found.append((f"{path} (working tree)", pin_file))
    try:
        log = _git(pins_dir, "log", "--format=commit %H %as %s", "--raw", "--no-abbrev", "--no-renames", "--", ".")
    except (OSError, subprocess.CalledProcessError):  # not in a git checkout
        return found
    commit, seen = "", set()
    for line in log.splitlines():
        if line.startswith("commit "):
            sha, subject = line[len("commit "):].split(" ", 1)
            commit = f"{sha[:12]} {subject}"
        elif line.startswith(":") and line.endswith(".json"):
            meta, path = line.split("\t", 1)
            blob = meta.split()[3]
            if blob in seen or not blob.strip("0"):  # all zeros: the file was deleted
                continue
            seen.add(blob)
            pin_file = _parse(_git(pins_dir, "cat-file", "blob", blob), path)
            if pin_file is not None and pin_file.version == version:
                found.append((f"{path}, added in {commit}", pin_file))
    return found


def _natural(text: str) -> List[Any]:
    return [int(part) if part.isdecimal() else part for part in re.split(r"(\d+)", text)]


def _config_text(config: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(config.items()))


def format_pin_file(pin_file: PinFile) -> str:
    """One line per pin, grouped by kernel or Inductor template."""
    entries = [(kernel, key, _config_text(config)) for (kernel, key), config in pin_file.triton.items()]
    entries += [(name, point, choice) for (name, point, _inputs), choice in pin_file.inductor.items()]
    lines: List[str] = []
    previous = None
    for name, key, value in sorted(entries, key=lambda e: (e[0], _natural(e[1]))):
        if name != previous:
            lines.append(name)
            previous = name
        lines.append(f"  {key}: {value}")
    return "".join(f"{line}\n" for line in lines)


def _show(pins_dir: Path, version: str) -> None:
    found = find_pin_files(pins_dir, version)
    if not found:
        raise SystemExit(f"no pin file with version {version} in {pins_dir} or its git history")
    print(f"pinned@{version}")
    for location, _ in found:
        print(f"  {location}")
    print(format_pin_file(found[0][1]), end="")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="triton-benchmarks-pins", description=__doc__.split("\n\n", maxsplit=1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    update = commands.add_parser("update", help="write pin files from the records of full runs")
    update.add_argument("--records", type=Path, nargs="+", required=True,
                        help="record dirs of full runs, oldest first; pins cover the benchmarks of the last one")
    update.add_argument("--pins", type=Path, required=True, help="current pin files root")
    update.add_argument("--output", type=Path, help="where to write the new pin files, default: --pins")
    update.add_argument("--min-gain", type=float, default=0.03,
                        help="minimum relative gain to move a pin to another config (default: 0.03)")
    update.add_argument("--summary", type=Path, help="also write the summary as markdown to this file")
    show = commands.add_parser("show", help="print the pin file behind a pinned@<version> report value")
    show.add_argument("version", help="pin file version, with or without the pinned@ prefix")
    show.add_argument("--pins", type=Path, required=True, help="pin files root, searched with its git history")
    args = parser.parse_args(argv)
    if args.command == "show":
        _show(args.pins, args.version.removeprefix("pinned@"))
        return
    summaries = update_pins(args.records, args.pins, args.output or args.pins, args.min_gain)
    text = _format_summary(summaries)
    print(text, end="")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
