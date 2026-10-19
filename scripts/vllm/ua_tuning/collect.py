# SPDX-License-Identifier: Apache-2.0
"""Collect tuning winners while preserving configs for workloads that were not retuned."""

import argparse
import json
import os
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def entries_by_key(profile):
    entries = {canonical(entry["key"]): entry for entry in profile["entries"]}
    if len(entries) != len(profile["entries"]):
        raise ValueError("Duplicate workload keys in tuning profile")
    return entries


def same_config(first, second):
    return {"heads_per_program": "MAX", **first} == {"heads_per_program": "MAX", **second}


def merge_profile(previous, incoming):
    """Update tuned workload configs while retaining results for workloads that were not retuned."""
    if any(previous[name] != incoming[name] for name in ("device_name", "static_key")):
        raise ValueError("Profile device or static key changed")
    merged = entries_by_key(previous)
    changed = 0
    for key, entry in entries_by_key(incoming).items():
        if key not in merged or not same_config(merged[key]["config"], entry["config"]):
            merged[key] = entry
            changed += 1
    return {**previous, **incoming, "entries": [merged[key] for key in sorted(merged)]}, changed


def collect(artifacts, destination):
    """Merge all four successful parts and return changed files and their entry counts."""
    for part in range(1, 5):
        if not (artifacts / f"ua-cold-graph-part-{part}-profiles").is_dir():
            raise ValueError(f"Missing final artifact for part {part}")
    updates, seen, identities = {}, set(), set()
    for path in sorted(artifacts.glob("ua-cold-graph-part-*-profiles/configs/*.json")):
        if path.name in seen:
            raise ValueError(f"Multiple parts produced {path.name}")
        seen.add(path.name)
        incoming = json.loads(path.read_text(encoding="utf-8"))
        identities.add((incoming["device_name"], incoming["triton_version"]))
        target = destination / path.name
        previous = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {**incoming, "entries": []}
        profile, changed = merge_profile(previous, incoming)
        if changed:
            updates[path.name] = (profile, changed)
    if len(identities) > 1:
        raise ValueError("Tuning parts used different GPUs or Triton versions")
    for name, (profile, _) in updates.items():
        destination.mkdir(parents=True, exist_ok=True)
        (destination / name).write_text(
            json.dumps(profile, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {name: changed for name, (_, changed) in updates.items()}, identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    updates, identities = collect(args.artifacts, args.configs)
    lines = [
        f"Tuning winners from [the source run]({args.run_url}) at `{args.source_sha}`.",
        "",
        "These changes use the tuning search results, "
        "existing configs for workloads that were not retuned are preserved.",
        "",
    ]
    if identities:
        device, version = next(iter(identities))
        lines.extend([f"GPU: {device}. Triton: {version}.", ""])
    if updates:
        lines.extend(["| Config file | Updated workload entries |", "| --- | ---: |"])
        lines.extend(f"| `{name}` | {count} |" for name, count in updates.items())
    else:
        lines.append("No tuning winners changed; no config PR is needed.")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.report.read_text(encoding="utf-8"))
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"changed={str(bool(updates)).lower()}\n")


if __name__ == "__main__":
    main()
