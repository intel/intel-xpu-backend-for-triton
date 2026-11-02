"""
Dedicated helper subprocess for `spirv_probes.has_spirv_capability`. Invoked
as `python -m triton.backends.intel._spirv_probe_worker <device_id>` with
the probe SPIR-V on stdin; prints `True`/`False`/`None` and exits 0.
"""

import os
import sys
from pathlib import Path

from triton.backends.intel.driver import compile_module_from_src


def _get_probe_checker():
    dirname = os.path.dirname(os.path.realpath(__file__))
    src = Path(os.path.join(dirname, "spirv_probes.c")).read_text()
    return compile_module_from_src(src=src, name="spirv_probes_impl")


def main() -> None:
    device_id = int(sys.argv[1])
    probe_spirv = sys.stdin.buffer.read()
    try:
        result = _get_probe_checker().probe_spirv_module(device_id, probe_spirv)
    except Exception:
        result = None
    print(result)


if __name__ == "__main__":
    main()
