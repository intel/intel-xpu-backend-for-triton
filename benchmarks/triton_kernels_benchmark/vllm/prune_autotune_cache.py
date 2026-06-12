#!/usr/bin/env python3
# Post-run cleanup for a Triton cache directory: keep only the autotune winner
# (and a small number of runners-up) per autotuned kernel, delete the rest.
# Non-autotuned kernels and helper-extension dirs (`*.so`) are left untouched.
#
# Usage: python3 prune_autotune_cache.py CACHE_DIR [--keep-top N] [--dry-run]
import argparse
import json
import shutil
import sys
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def winner_configs(autotune_json, keep_top):
    # configs_timings is [(config_dict, [t0, t1, ...]), ...]; autotune ranks by min(t).
    # Returns a list of (num_warps, num_stages) tuples — the top-N configs by min time.
    ranked = sorted(autotune_json["configs_timings"], key=lambda e: min(e[1]))
    out = []
    for cfg, _ in ranked[:keep_top]:
        out.append((cfg.get("num_warps"), cfg.get("num_stages")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--keep-top", type=int, default=1,
                    help="Per autotuned kernel, keep this many configs (winner first). Default 1.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.cache_dir.is_dir():
        print(f"prune_autotune_cache: not a directory: {args.cache_dir}", file=sys.stderr)
        return 1

    # 1) Find all autotune.json files. Each is "<kernel_name>.autotune.json".
    autotune_files = list(args.cache_dir.glob("*/*.autotune.json"))
    if not autotune_files:
        print("prune_autotune_cache: no *.autotune.json found, nothing to do.")
        return 0

    # 2) For each autotuned kernel name, build the keeper set of (num_warps, num_stages).
    keepers = {}  # kernel_name -> set of (num_warps, num_stages) to keep
    for af in autotune_files:
        kernel_name = af.name[:-len(".autotune.json")]
        data = load_json(af)
        winners = winner_configs(data, args.keep_top)
        keepers.setdefault(kernel_name, set()).update(winners)
        print(f"prune_autotune_cache: kernel={kernel_name} winners={winners}")

    # 3) Walk every cache subdir; if it has <kernel_name>.json for an autotuned
    #    kernel, keep iff its (num_warps, num_stages) is in the keeper set.
    deleted = 0
    kept = 0
    for sub in args.cache_dir.iterdir():
        if not sub.is_dir():
            continue
        for kernel_name, keep_set in keepers.items():
            meta = sub / f"{kernel_name}.json"
            if not meta.is_file():
                continue
            md = load_json(meta)
            cfg = (md.get("num_warps"), md.get("num_stages"))
            if cfg in keep_set:
                kept += 1
                print(f"  keep   {sub.name}  {kernel_name}  {cfg}")
            else:
                action = "WOULD-DELETE" if args.dry_run else "delete"
                print(f"  {action} {sub.name}  {kernel_name}  {cfg}")
                if not args.dry_run:
                    shutil.rmtree(sub)
                deleted += 1
            break  # this dir matched one autotuned kernel; move on
    print(f"prune_autotune_cache: kept={kept} deleted={deleted} (dry_run={args.dry_run})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
