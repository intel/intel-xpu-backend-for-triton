"""
Ground-truth probes for whether a SPIR-V extension is safe to emit, asking
whichever side actually consumes the compiled module: the installed driver's
own JIT (via Level Zero) for a regular compile, or `ocloc` for an AOT
(`generate_native_code`) one. `accepts_*`
below are the only entry points callers need; everything else is how each
probe is actually run.
"""

import functools
import hashlib
import os
import signal
import struct
import subprocess
import sys
import tempfile
import threading

# Generous bound on how long a probe child may run before it's assumed hung.
_PROBE_TIMEOUT_SECONDS = 60

# Prefix the worker puts on its result line, so the parent can pick the
# answer out of stdout even if some unrelated library the worker imports
# also writes to stdout.
PROBE_RESULT_MARKER = 'SPIRV_PROBE_RESULT:'

# Standalone script (not `-m`): the worker must not import the `triton` package, see its docstring.
_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_spirv_probe_worker.py')

_probe_library_lock = threading.Lock()


@functools.lru_cache(maxsize=None)
def _build_probe_library() -> str:
    from triton.backends.intel.driver import compile_module_from_src  # deferred: heavy, and not needed to import this
    with open(os.path.join(os.path.dirname(_WORKER_SCRIPT), 'spirv_probes.c')) as f:
        return compile_module_from_src(src=f.read(), name='spirv_probes_impl').cache_path


def _probe_library_path() -> str:
    """Path of the built `spirv_probes.c` library, which the worker loads directly.

    `lru_cache` doesn't lock the call it wraps, and probes run concurrently, so serialize the (first) build to
    avoid several threads compiling the same library at once.
    """
    with _probe_library_lock:
        return _build_probe_library()


@functools.lru_cache(maxsize=None)
def _run_probe_worker(device_id: int, probe_spirv: bytes) -> tuple[bool | None, bool]:
    """Runs one JIT probe in the helper subprocess.

    Returns `(result, definitive)`. `definitive` means the answer is a property of the driver build, device and
    module, so callers may persist it: a clean True/False report, or the worker exiting on its own with a non-zero
    status / aborting (how the driver typically rejects an unsupported extension). A timeout, being killed by another
    signal, or a missing report is `False`/`None` for this process's safety but may be transient (busy GPU, load,
    OOM kill), so callers must not persist it.
    """
    # An unsupported extension can abort the whole process instead of
    # returning a clean error, so the probe runs in a dedicated helper
    # subprocess (`_spirv_probe_worker`, not `multiprocessing`, which would
    # re-execute the caller's own `__main__` to bootstrap) -- an abort then
    # only costs that child.
    try:
        proc = subprocess.run(
            [sys.executable, '-I', _WORKER_SCRIPT,
             _probe_library_path(), str(device_id)], input=probe_spirv, capture_output=True,
            timeout=_PROBE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return False, False  # Hung: treat like a crash.
    if proc.returncode != 0:
        # Exited with an error or aborted: the capability is unsafe to use. Other signals (e.g. an OOM kill) say
        # nothing about the module.
        return False, proc.returncode > 0 or proc.returncode == -signal.SIGABRT
    lines = proc.stdout.decode(errors='replace').splitlines()
    marker_line = next((line for line in lines if line.startswith(PROBE_RESULT_MARKER)), None)
    if marker_line is None:
        return None, False  # Worker didn't get far enough to report.
    result = {'True': True, 'False': False}.get(marker_line.removeprefix(PROBE_RESULT_MARKER))
    return result, result is not None


def has_spirv_capability(device_id: int, probe_spirv: bytes) -> bool | None:
    """
    Ground-truth probe for a SPIR-V capability/extension, via the driver's own JIT.

    Unlike `has_device_extension`, which only checks what the driver
    *advertises*, this feeds `probe_spirv` -- a minimal SPIR-V module
    declaring only the capability/extension under test -- to `zeModuleCreate`,
    the same call `load_binary` makes for a real kernel, so a rolling driver
    that hasn't advertised a newly-landed capability yet is still measured
    correctly.

    Args:
        device_id: Device ID to query (from sycl::ext::intel::info::device::device_id)
        probe_spirv: a minimal SPIR-V module declaring only the capability/extension to test.

    Returns:
        True/False if the driver accepted/rejected the probe module, or None
        if the device isn't found.
    """
    return _run_probe_worker(device_id, probe_spirv)[0]


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
    + _spirv_op(17, 6460)  # OpCapability RegisterLimitsINTEL (required by MaximumRegistersINTEL)
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


def _capability_probe(capability: int, extension: str) -> bytes:
    """Minimal, valid SPIR-V kernel declaring one `capability` and the `extension` providing it."""
    name = extension.encode() + b'\0'
    name = name.ljust((len(name) + 3) // 4 * 4, b'\0')
    return (_spirv_words(0x07230203, 0x00010400, 0, 5, 0)  # magic, v1.4, generator, bound, schema
            + _spirv_op(17, 6)  # OpCapability Kernel
            + _spirv_op(17, 4)  # OpCapability Addresses
            + _spirv_op(17, capability)  # OpCapability <capability>
            + _spirv_op(10, *struct.unpack(f'<{len(name) // 4}I', name))  # OpExtension
            + _spirv_op(14, 2, 2)  # OpMemoryModel Physical64 OpenCL
            + _spirv_op(15, 6, 3, *struct.unpack('<2I', b'main\0\0\0\0'))  # OpEntryPoint Kernel %3 "main"
            + _spirv_op(19, 1)  # %1 = OpTypeVoid
            + _spirv_op(33, 2, 1)  # %2 = OpTypeFunction %1
            + _spirv_op(54, 1, 3, 0, 2)  # %3 = OpFunction %1 None %2
            + _spirv_op(248, 4)  # %4 = OpLabel
            + _spirv_op(253)  # OpReturn
            + _spirv_op(56)  # OpFunctionEnd
            )


# Capability ids: this project's vendored `spirv_internal.hpp` / `spirv.hpp`.
_16BIT_ATOMICS_PROBE_SPV = _capability_probe(6260, 'SPV_INTEL_16bit_atomics')  # AtomicInt16CompareExchangeINTEL
_SIGMOID_PROBE_SPV = _capability_probe(6167, 'SPV_INTEL_sigmoid')  # SigmoidINTEL
_SCALED_DPAS_PROBE_SPV = _capability_probe(
    6263, 'SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate')  # SubgroupScaledMatrixMultiplyAccumulateINTEL

# Instruction-level probes. Some capabilities have no SPIR-V capability of their own (BF8 DPAS is an operand mask on
# an existing instruction, 256B prefetch is just a wider tile), so these probe modules contain the real call, as
# emitted by the TritonGEN lowering, and are accepted only if the compiler (IGC) builds them.
_LLVM_HEADER = ('target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-'
                'v1024:1024-n8:16:32:64"\ntarget triple = "spir64-unknown-unknown"\n')


def _dpas_probe_ir(k_dim: int, operands: int) -> str:
    fn = '_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi'
    return _LLVM_HEADER + f"""
declare spir_func <8 x float> @{fn}(i32, <8 x i16>, <8 x i32>, <8 x float>, i32) convergent nounwind willreturn memory(none)
define spir_kernel void @probe(ptr addrspace(1) %o, <8 x i16> %a, <8 x i32> %b, <8 x float> %c) {{
  %r = call spir_func <8 x float> @{fn}(i32 {k_dim}, <8 x i16> %a, <8 x i32> %b, <8 x float> %c, i32 {operands})
  store <8 x float> %r, ptr addrspace(1) %o
  ret void
}}
"""


def _prefetch_probe_ir(elem_bytes: int, width: int, height: int, v_blocks: int) -> str:
    fn = '_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i'
    return _LLVM_HEADER + f"""
declare spir_func void @{fn}(i32, i32, i32, i32, ptr addrspace(1), i32, i32, i32, <2 x i32>) nounwind
define spir_kernel void @probe(ptr addrspace(1) %p, i32 %bw, i32 %bh, i32 %pitch, <2 x i32> %xy) {{
  call spir_func void @{fn}(i32 {elem_bytes}, i32 {width}, i32 {height}, i32 {v_blocks}, ptr addrspace(1) %p, i32 %bw, i32 %bh, i32 %pitch, <2 x i32> %xy)
  ret void
}}
"""


_translate_lock = threading.Lock()


@functools.lru_cache(maxsize=None)
def _llvm_ir_to_spirv(llvm_ir: str) -> bytes:
    from triton._C.libtriton import intel  # deferred: only needed by instruction-level probes
    # Probes run on concurrent threads and the translator keeps global (LLVM `cl::opt`/writer) state, so serialize it.
    # It runs in-process, but only on these fixed, known-good IR strings, and only on a disk-cache miss.
    with _translate_lock:
        spirv, _ = intel.translate_to_spirv(llvm_ir, False, {})
    return bytes(spirv)


# BF8 DPAS: k=32, E5M2 A/B operands (0x10000 | 0x20000).
_BF8_DPAS_IR = _dpas_probe_ir(32, 0x10000 | 0x20000)
# 256B prefetch: 4 blocks x 16 x 4B per row.
_PREFETCH_256B_IR = _prefetch_probe_ir(4, 16, 1, 4)


def _ocloc_builds(device_arch: str, spirv: bytes) -> bool:
    """Whether the `ocloc` on PATH builds `spirv` for `device_arch`."""
    with tempfile.TemporaryDirectory() as temp_dir:
        spv = os.path.join(temp_dir, 'probe.spv')
        with open(spv, 'wb') as f:
            f.write(spirv)
        try:
            proc = subprocess.run([
                'ocloc', 'compile', '-file', spv, '-o',
                os.path.join(temp_dir, 'probe.o'), '-spirv_input', '-device', device_arch
            ], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return True  # No usable `ocloc`: `make_zebin` reports it.
    return proc.returncode == 0


_ocloc_builds_cached = functools.lru_cache(maxsize=None)(_ocloc_builds)


def _disk_cached(driver_version: str | None, device_id: int | None, probe: bytes | str, compute) -> bool:
    """Memoize a JIT probe result on disk, so a warm process doesn't re-spawn the probe subprocesses.

    The result depends only on the driver build, the device and the probe module, which form the key. Without a
    `driver_version` the result can't be keyed safely, so nothing is cached. `compute` returns
    `(result, definitive)`; only definitive results are persisted, so a transient failure (timeout, crash, device
    not found) can't permanently disable a feature.
    """
    if not driver_version or device_id is None:
        return compute()[0]
    from triton import knobs  # deferred: keeps this module importable standalone
    data = probe.encode() if isinstance(probe, str) else probe
    key = hashlib.sha256(b'\0'.join([str(driver_version).encode(), str(device_id).encode(), data])).hexdigest()
    path = os.path.join(knobs.cache.dir, 'spirv_probes', key)
    try:
        with open(path) as f:
            value = f.read()
        if value in ('True', 'False'):
            return value == 'True'
    except OSError:
        pass
    result, definitive = compute()
    if not definitive:
        return result
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.{os.getpid()}.tmp'
        with open(tmp, 'w') as f:
            f.write(str(result))
        os.replace(tmp, path)  # atomic: concurrent processes never see a partial file
    except OSError:
        pass
    return result


def _jit_probe(device_id: int | None, probe_spv: bytes) -> tuple[bool, bool]:
    if device_id is None:
        return False, False
    result, definitive = _run_probe_worker(device_id, probe_spv)
    return bool(result), definitive


@functools.lru_cache(maxsize=None)  # also in-process: `parse_options` consults these on every kernel compile
def _jit_accepts(device_id: int | None, driver_version: str | None, probe_spv: bytes) -> bool:
    return _disk_cached(driver_version, device_id, probe_spv, lambda: _jit_probe(device_id, probe_spv))


def _builds(native: bool, device_arch: str, device_id: int | None, probe_ir: str,
            driver_version: str | None = None) -> bool:
    """Whether the compiler (`ocloc` if `native`, else the driver's JIT) builds `probe_ir`."""
    if native:
        return _ocloc_builds_cached(device_arch, _llvm_ir_to_spirv(probe_ir))
    return _disk_cached(driver_version, device_id, probe_ir, lambda: _jit_probe(device_id, _llvm_ir_to_spirv(probe_ir)))


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
    """Whether the `ocloc` on PATH accepts SPIR-V that uses `SPV_INTEL_maximum_registers`."""
    return _ocloc_accepts_extension(device_arch, 'SPV_INTEL_maximum_registers', _MAX_REGISTERS_PROBE_SPV)


def accepts_long_vector(native: bool, device_arch: str, device_id: int | None,
                        driver_version: str | None = None) -> bool:
    """Whether `SPV_EXT_long_vector` is safe to emit for this compile."""
    if native:
        return ocloc_accepts_long_vector(device_arch)
    return _jit_accepts(device_id, driver_version, _LONG_VECTOR_PROBE_SPV)


def accepts_maximum_registers(native: bool, device_arch: str, device_id: int | None,
                              driver_version: str | None = None) -> bool:
    """Whether `SPV_INTEL_maximum_registers` is safe to emit for this compile."""
    if native:
        return ocloc_accepts_maximum_registers(device_arch)
    return _jit_accepts(device_id, driver_version, _MAX_REGISTERS_PROBE_SPV)


def _accepts(native: bool, device_arch: str, device_id: int | None, extension: str, probe_spv: bytes,
             driver_version: str | None = None) -> bool:
    if native:
        return _ocloc_accepts_extension_cached(device_arch, extension, probe_spv)
    return _jit_accepts(device_id, driver_version, probe_spv)


_ocloc_accepts_extension_cached = functools.lru_cache(maxsize=None)(_ocloc_accepts_extension)


def accepts_16bit_atomics(native: bool, device_arch: str, device_id: int | None,
                          driver_version: str | None = None) -> bool:
    """Whether `SPV_INTEL_16bit_atomics` is safe to emit for this compile."""
    return _accepts(native, device_arch, device_id, 'SPV_INTEL_16bit_atomics', _16BIT_ATOMICS_PROBE_SPV, driver_version)


def accepts_sigmoid(native: bool, device_arch: str, device_id: int | None, driver_version: str | None = None) -> bool:
    """Whether `SPV_INTEL_sigmoid` is safe to emit for this compile."""
    return _accepts(native, device_arch, device_id, 'SPV_INTEL_sigmoid', _SIGMOID_PROBE_SPV, driver_version)


def accepts_scaled_dpas(native: bool, device_arch: str, device_id: int | None,
                        driver_version: str | None = None) -> bool:
    """Whether `SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate` is safe to emit for this compile."""
    return _accepts(native, device_arch, device_id, 'SPV_INTEL_subgroup_scaled_matrix_multiply_accumulate',
                    _SCALED_DPAS_PROBE_SPV, driver_version)


def accepts_bf8_dpas(native: bool, device_arch: str, device_id: int | None, driver_version: str | None = None) -> bool:
    """Whether DPAS with BF8 (E5M2) operands builds for this compile."""
    return _builds(native, device_arch, device_id, _BF8_DPAS_IR, driver_version)


def accepts_256b_prefetch(native: bool, device_arch: str, device_id: int | None,
                          driver_version: str | None = None) -> bool:
    """Whether a 2D block prefetch of 256 bytes per row builds for this compile."""
    return _builds(native, device_arch, device_id, _PREFETCH_256B_IR, driver_version)
