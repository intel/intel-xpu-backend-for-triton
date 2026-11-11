"""
Dedicated helper subprocess for `spirv_probes.has_spirv_capability`. Invoked
as `python -I _spirv_probe_worker.py <probe library> <device_id>` with the
probe SPIR-V on stdin; prints a `SPIRV_PROBE_RESULT:`-prefixed
`True`/`False`/`None` line and exits 0.

Deliberately stdlib-only and run as a plain script rather than `-m
triton.backends.intel...`: importing the `triton` package costs ~100 ms, which
dominated the probe. The parent has already built the probe library
(`spirv_probes.c`) and passes its path, so all that is needed here is `ctypes`.
"""

import ctypes
import sys

# Keep in sync with `spirv_probes.PROBE_RESULT_MARKER` (this script can't import it).
PROBE_RESULT_MARKER = 'SPIRV_PROBE_RESULT:'


def main() -> None:
    library_path, device_id = sys.argv[1], int(sys.argv[2])
    probe_spirv = sys.stdin.buffer.read()
    try:
        lib = ctypes.PyDLL(library_path)
        lib.probe_spirv_module.restype = ctypes.py_object
        lib.probe_spirv_module.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t)
        result = lib.probe_spirv_module(device_id, probe_spirv, len(probe_spirv))
    except Exception:
        result = None
    print(f'{PROBE_RESULT_MARKER}{result}')


if __name__ == "__main__":
    main()
