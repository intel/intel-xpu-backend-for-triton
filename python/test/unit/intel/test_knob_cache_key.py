"""Every `knobs.intel.*` knob must be accounted for in the compile cache key.

https://github.com/intel/intel-xpu-backend-for-triton/issues/6790

A knob that changes the compiled kernel but is absent from the cache key makes
the env var look inert: compile once, set the var, re-run, and the previously
compiled binary is served from the on-disk cache. That is worst for the
`disable_*` debugging switches, which exist precisely to A/B a suspected bug.

There are three ways a knob can legitimately reach the key, and
`test_every_intel_knob_is_cache_classified` requires every knob to use one:

1. Its env var is in `CACHE_INVALIDATING_ENV_VARS` (`Tools/Sys/GetEnv.h`).
   `get_cache_invalidating_env_vars()` reads that set out of the *process
   environment*, so this works for a knob read only from Python just as well as
   for one read from C++ -- and `knobs.base_knobs.__set__` mirrors programmatic
   assignment into `os.environ`, so it covers that too. This is the only option
   for a knob consumed somewhere other than `XPUOptions` (a build flag, an
   `ocloc` argument, a raw `os.environ` read in the pass pipeline).
2. It is reflected into an `XPUOptions` field, which `XPUOptions.hash()`
   digests along with the rest of `__dict__`.
3. It is reflected into `XPUBackend.properties`, which `XPUBackend.hash()`
   digests.

A knob that does not change the compiled kernel at all (a disassembler path, a
runtime-only dump switch) belongs in `CACHE_NEUTRAL_KNOBS` with a reason, the
Python-side counterpart of `CACHE_NEUTRAL_ENV_VARS`.
"""

import pytest

import triton
import triton.language as tl
from triton import knobs
from triton._C.libtriton import get_cache_invalidating_env_vars
from triton._internal_testing import is_xpu
from triton.backends.intel.compiler import XPUOptions
from triton.compiler.compiler import ASTSource, make_backend
from triton.runtime import driver
from triton.runtime.cache import get_cache_key

# Knobs hashed by being present in the environment. Each changes the compiled
# kernel from somewhere `XPUOptions` does not reach: a pass that runs or does
# not run, an IGC build flag, the `ocloc -device` target.
INVALIDATING_KNOBS = {
    "descriptor_block_io_relayout",
    "device_arch",
    "disable_annotate_cache_control",
    "disable_canonicalize_pointers",
    "disable_igc_opt",
    "dump_shader_info",
    "fast_math",
    "opt_reduction_locality",
}

# Same, for cache-affecting env vars the backend reads directly rather than
# through a `knobs.intel` descriptor.
INVALIDATING_RAW_ENV_VARS = (
    # Gates `add_assign_latencies` / `add_schedule_loops` in `make_ttgir`, and
    # `annotationsAreUsable` in MatmulLoopPipeline.cpp.
    "TRITON_INTEL_ANNOTATE_LATENCIES",
    # Picks `runSpirvBackend` over `writeSpirv` in SPIRVTranslation.cpp.
    "TRITON_USE_SPIRV_BACKEND",
)

# Knobs reflected into the `XPUOptions` field named here.
KNOBS_REFLECTED_IN_OPTIONS = {
    "disable_optimize_load_masks": "optimize_load_masks",
    "dynamic_shared_memory": "dynamic_shared_memory",
    "enable_code_sinking": "code_sinking",
    "enable_loop_distribution": "loop_distribute",
    "enable_sub_32_dpas": "sub_32_dpas",
    "gen_native_code": "generate_native_code",
    "in_loop_sink": "in_loop_sink",
    "libdevice_path": "extern_libs",
}

# Knobs reflected into `XPUBackend.properties` (`parse_target` folds the
# queried -- or env-supplied -- device extensions into it).
KNOBS_REFLECTED_IN_TARGET = {"device_extensions"}

# Knobs that cannot change the compiled kernel, and why.
CACHE_NEUTRAL_KNOBS = {
    "spirv_dis": "disassembler used by triton.tools.disasm, never during compilation",
    "enable_dump_kernel_args": "XPULauncher runtime behaviour only",
    "dump_kernel_args_dir": "XPULauncher runtime behaviour only",
    "print_dump_kernel_args_info": "XPULauncher runtime behaviour only",
    "sycl_compiler": "builds the host launcher .so, which has its own cache key",
}


# A `tl.dot` feeding a row-wise `tl.sum`, so that
# `-tritonintelgpu-optimize-reduction-locality` -- which only matches a
# reduction whose operand carries a DPAS encoding -- really does rewrite it.
@triton.jit
def _dot_reduce_kernel(a_ptr, b_ptr, y_ptr):
    offs = tl.arange(0, 64)
    a = tl.load(a_ptr + offs[:, None] * 64 + offs[None, :])
    b = tl.load(b_ptr + offs[:, None] * 64 + offs[None, :])
    tl.store(y_ptr + offs, tl.sum(tl.dot(a, b), axis=1))


def _cache_key():
    backend = make_backend(driver.active.get_current_target())
    src = ASTSource(fn=_dot_reduce_kernel, signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "y_ptr": "*fp32"})
    return get_cache_key(src, backend, backend.parse_options({}), get_cache_invalidating_env_vars())


def test_every_intel_knob_is_cache_classified():
    """A knob added without a cache classification fails here.

    The Python-side analogue of `tools::assertIsRecognized`, which only guards
    env vars read from C++.
    """
    classified = (INVALIDATING_KNOBS | set(KNOBS_REFLECTED_IN_OPTIONS) | KNOBS_REFLECTED_IN_TARGET
                  | set(CACHE_NEUTRAL_KNOBS))
    assert set(knobs.intel.knob_descriptors) == classified


@pytest.mark.parametrize("knob", sorted(INVALIDATING_KNOBS))
def test_knob_env_var_is_cache_invalidating(knob, monkeypatch):
    env_var = knobs.intel.knob_descriptors[knob].key
    monkeypatch.setenv(env_var, "1")
    assert env_var in get_cache_invalidating_env_vars()


@pytest.mark.parametrize("env_var", INVALIDATING_RAW_ENV_VARS)
def test_raw_env_var_is_cache_invalidating(env_var, monkeypatch):
    monkeypatch.setenv(env_var, "1")
    assert env_var in get_cache_invalidating_env_vars()


@pytest.mark.parametrize("knob, field", sorted(KNOBS_REFLECTED_IN_OPTIONS.items()))
def test_reflected_knob_has_a_hashed_options_field(knob, field):
    # `XPUOptions.hash()` digests all of `__dict__`, so being a field is enough.
    assert field in XPUOptions.__dataclass_fields__


@pytest.mark.skipif(not is_xpu(), reason="needs an XPU target to build a cache key")
def test_target_properties_are_hashed():
    """Backs `KNOBS_REFLECTED_IN_TARGET`: anything in `properties` is hashed."""
    backend = make_backend(driver.active.get_current_target())
    assert str(backend.properties) in backend.hash()


@pytest.mark.skipif(not is_xpu(), reason="needs an XPU target to build a cache key")
def test_knob_changes_the_compile_cache_key(monkeypatch):
    """End-to-end check that a knob in `INVALIDATING_KNOBS` moves the key.

    `TRITON_INTEL_OPTIMIZE_REDUCTION_LOCALITY` stands in for the group: it adds
    a pass that measurably changes `_dot_reduce_kernel`'s TTGIR and SPIR-V, yet
    before #6790 the key was byte-identical with the knob on and off, so the
    second run was served the first run's binary.
    """
    before = _cache_key()
    monkeypatch.setenv("TRITON_INTEL_OPTIMIZE_REDUCTION_LOCALITY", "1")
    assert _cache_key() != before
