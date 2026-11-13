#!/usr/bin/env python3
"""Dump the current XPU target to a JSON file for `TRITON_INTEL_TARGET_JSON`.

Run it on a machine with the GPU. A runner without a GPU can then compile kernels
for exactly this device, e.g. to pre-warm the Triton compile cache.
"""
import json
import sys

from triton.runtime import driver


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} OUTPUT.json")
    target = driver.active.get_current_target()
    props = driver.active.utils.get_device_properties(driver.active.get_current_device())
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump({"warp_size": target.warp_size, "arch": target.arch, "device_properties": props}, f, indent=2,
                  sort_keys=True)
    arch, name, driver_version = (target.arch[key] for key in ("arch", "name", "driver_version"))
    print(f"{arch}: {name}, driver {driver_version}")


if __name__ == "__main__":
    main()
