"""
Ground-truth probes for whether a SPIR-V extension is safe to emit, asking
whichever side actually consumes the compiled module: the installed driver's
own JIT (via Level Zero) for a regular compile, or `ocloc` for an AOT
(`generate_native_code`) one. `accepts_long_vector`/`accepts_maximum_registers`
below are the only entry points callers need; everything else is how each
probe is actually run.
"""

import functools
import os
import struct
import subprocess
import tempfile

from triton.backends.intel.extension_utils import has_spirv_capability


def _spirv_words(*words):
    return b''.join(struct.pack('<I', w) for w in words)


def _spirv_op(opcode, *operands):
    return _spirv_words(((1 + len(operands)) << 16) | opcode, *operands)


# Minimal, valid SPIR-V kernel declaring `SPV_EXT_long_vector`.
_LONG_VECTOR_PROBE_SPV = (
    _spirv_words(0x07230203, 0x00010400, 0, 5, 0)  # magic, v1.4, generator, bound, schema
    + _spirv_op(17, 6)  # OpCapability Kernel
    + _spirv_op(17, 4)  # OpCapability Addresses
    + _spirv_op(10, *struct.unpack('<5I', b'SPV_EXT_long_vector\0'.ljust(20, b'\0')))  # OpExtension
    + _spirv_op(14, 2, 2)  # OpMemoryModel Physical64 OpenCL
    + _spirv_op(15, 6, 3, *struct.unpack('<2I', b'main\0\0\0\0'))  # OpEntryPoint Kernel %3 "main"
    + _spirv_op(19, 1)  # %1 = OpTypeVoid
    + _spirv_op(33, 2, 1)  # %2 = OpTypeFunction %1
    + _spirv_op(54, 1, 3, 0, 2)  # %3 = OpFunction %1 None %2
    + _spirv_op(248, 4)  # %4 = OpLabel
    + _spirv_op(253)  # OpReturn
    + _spirv_op(56)  # OpFunctionEnd
)

# Same shape, requesting a `MaximumRegistersINTEL` execution mode (register
# count 128, picked arbitrarily) instead. Opcode/enum values cross-checked
# against this project's vendored `SPIRVWriter.cpp` and `spirv.h`, not guessed.
_MAX_REGISTERS_PROBE_SPV = (
    _spirv_words(0x07230203, 0x00010400, 0, 5, 0)  # magic, v1.4, generator, bound, schema
    + _spirv_op(17, 6)  # OpCapability Kernel
    + _spirv_op(17, 4)  # OpCapability Addresses
    + _spirv_op(10, *struct.unpack('<7I', b'SPV_INTEL_maximum_registers\0'))  # OpExtension
    + _spirv_op(14, 2, 2)  # OpMemoryModel Physical64 OpenCL
    + _spirv_op(15, 6, 3, *struct.unpack('<2I', b'main\0\0\0\0'))  # OpEntryPoint Kernel %3 "main"
    + _spirv_op(16, 3, 6461, 128)  # OpExecutionMode %3 MaximumRegistersINTEL 128
    + _spirv_op(19, 1)  # %1 = OpTypeVoid
    + _spirv_op(33, 2, 1)  # %2 = OpTypeFunction %1
    + _spirv_op(54, 1, 3, 0, 2)  # %3 = OpFunction %1 None %2
    + _spirv_op(248, 4)  # %4 = OpLabel
    + _spirv_op(253)  # OpReturn
    + _spirv_op(56)  # OpFunctionEnd
)


def _ocloc_accepts_extension(device_arch: str, extension: str, probe_spv: bytes) -> bool:
    """Whether the `ocloc` on PATH accepts SPIR-V that declares `extension`.

    `ocloc`'s IGC is independent of the installed driver: it may predate
    `extension` entirely ("unknown extension"), or know it but have it off by
    default ("which were disabled by --spirv-ext option", seen for
    `SPV_INTEL_maximum_registers` on IGC 2026.21.6.0.17_160000). Either way
    the rejection names the extension while reading the module, so a module
    that only declares it is a sufficient probe; every other failure is
    ignored.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        spv = os.path.join(temp_dir, 'probe.spv')
        with open(spv, 'wb') as f:
            f.write(probe_spv)
        try:
            proc = subprocess.run([
                'ocloc', 'compile', '-file', spv, '-o',
                os.path.join(temp_dir, 'probe.o'), '-spirv_input', '-device', device_arch
            ], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return True  # No usable `ocloc`: `make_zebin` reports it.
    output = proc.stdout + proc.stderr
    rejected = extension in output and ('unknown extension' in output or 'disabled by --spirv-ext option' in output)
    return not rejected


@functools.lru_cache(maxsize=None)
def ocloc_accepts_long_vector(device_arch: str) -> bool:
    """Whether the `ocloc` on PATH accepts SPIR-V that uses `SPV_EXT_long_vector`."""
    return _ocloc_accepts_extension(device_arch, 'SPV_EXT_long_vector', _LONG_VECTOR_PROBE_SPV)


@functools.lru_cache(maxsize=None)
def ocloc_accepts_maximum_registers(device_arch: str) -> bool:
    """Whether the `ocloc` on PATH accepts SPIR-V that uses `SPV_INTEL_maximum_registers`.

    See https://github.com/intel/intel-xpu-backend-for-triton/pull/8322: off LTS, an
    explicit `grf_mode` is requested through this extension instead of an IGC
    build flag. This is the AOT half of that probe; `has_spirv_capability` is
    the driver-JIT half.
    """
    return _ocloc_accepts_extension(device_arch, 'SPV_INTEL_maximum_registers', _MAX_REGISTERS_PROBE_SPV)


def accepts_long_vector(native: bool, device_arch: str, device_id: int | None) -> bool:
    """Whether `SPV_EXT_long_vector` is safe to emit for this compile.

    `native` picks the actual consumer of the SPIR-V: `ocloc` for an AOT
    (`generate_native_code`) compile, otherwise the installed driver's own
    JIT. Defaults to False (don't use it) when the JIT probe can't reach a
    Level Zero device (e.g. `device_id` unset, driver absent) -- this only
    ever picks a fallback encoding, never affects correctness, so there's no
    need to guess.
    """
    if native:
        return ocloc_accepts_long_vector(device_arch)
    return bool(device_id is not None and has_spirv_capability(device_id, _LONG_VECTOR_PROBE_SPV))


def accepts_maximum_registers(native: bool, device_arch: str, device_id: int | None) -> bool:
    """Whether `SPV_INTEL_maximum_registers` is safe to emit for this compile.

    Same dispatch as `accepts_long_vector`; see
    https://github.com/intel/intel-xpu-backend-for-triton/pull/8322 for why
    this one is probed at all.
    """
    if native:
        return ocloc_accepts_maximum_registers(device_arch)
    return bool(device_id is not None and has_spirv_capability(device_id, _MAX_REGISTERS_PROBE_SPV))
