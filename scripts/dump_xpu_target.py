#!/usr/bin/env python3
"""Dump the current XPU target to a JSON file for `TRITON_INTEL_TARGET_JSON`.

Run it on a machine with the GPU. A runner without a GPU can then compile kernels
for exactly this device, e.g. to pre-warm the Triton compile cache.

    dump_xpu_target.py OUTPUT.json   write the target and the `ocloc` version
    dump_xpu_target.py --check FILE  fail unless `ocloc` here is the version FILE was dumped with

The binaries are built by `ocloc`, so a runner with a different driver stack cannot warm the cache
(an older one rejects SPIR-V extensions that the newer one emits).
"""
import json
import subprocess
import sys


def ocloc_version():
    return subprocess.run(["ocloc", "--version"], check=True, capture_output=True, text=True).stdout.strip()


def dump(path):
    from triton.runtime import driver  # pylint: disable=import-outside-toplevel
    target = driver.active.get_current_target()
    props = driver.active.utils.get_device_properties(driver.active.get_current_device())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "warp_size": target.warp_size, "arch": target.arch, "device_properties": props, "ocloc_version":
                ocloc_version()
            }, f, indent=2, sort_keys=True)
    arch, name, driver_version = (target.arch[key] for key in ("arch", "name", "driver_version"))
    print(f"{arch}: {name}, driver {driver_version}, ocloc {ocloc_version()}")


def check(path):
    with open(path, encoding="utf-8") as f:
        expected = json.load(f)["ocloc_version"]
    actual = ocloc_version()
    if actual != expected:
        sys.exit(f"ocloc {actual} on this runner differs from {expected} on the GPU runner the target was dumped on")
    print(f"ocloc {actual} matches the GPU runner")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        check(sys.argv[2])
    elif len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        dump(sys.argv[1])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
