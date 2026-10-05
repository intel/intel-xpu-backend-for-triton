import os
import re
import shutil
import struct
import sys

import pytest
import torch
import triton
import triton.language as tl

import pathlib

from triton.runtime import build
from triton.runtime.driver import driver
from triton._internal_testing import is_xpu_cri
from triton.backends.intel import driver as intel_driver
from triton.backends.intel import extension_utils
from triton.backends.intel.compiler import REBUILD_SPILL_BYTES_PER_THREAD
from triton.backends.intel.driver import CompilationHelper, find_sycl_icpx
from triton.runtime.errors import IntelGPUError, OutOfResources


def test_auto_grf(device, monkeypatch, capfd):
    monkeypatch.setenv("TRITON_DEBUG", "1")
    # CRI's larger (512-GRF) register file needs a bigger tile to spill; other
    # targets already spill at 8K.
    BLOCK = 1024 * 32 if is_xpu_cri() else 1024 * 8
    z_tri = torch.empty(BLOCK, dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(z, BLOCK: tl.constexpr):
        # make it hard to re-schedule.
        off = tl.arange(0, BLOCK)
        a = tl.load(z + off)
        result = tl.sum(a, axis=0, keep_dims=True)
        tl.store(z + off, a + result)

    _kernel[(1, )](z_tri, BLOCK=BLOCK, num_warps=2)
    _ = torch.arange(0, BLOCK, dtype=torch.int32, device=device)

    outs = [line for line in capfd.readouterr().out.splitlines() if line]

    # The output should contain the recompiling information for large GRF mode.
    assert "retrying with large GRF mode" in outs[0]
    # The spill size of returned kernel should be same kernel as the one compiled with large GRF mode.
    # Compare the byte counts because they identify the *binary*: both lines
    # describe the retried build, so equality pins that it is the one returned. The
    # gate's own comparison (bytes vs `kRebuildSpillBytesPerThread`) is covered by
    # test_n_spills_reported_per_lane.
    retried = re.search(r"kernel has (\d+) spill bytes per hardware thread", outs[1])
    selected = re.search(r"Detected (\d+) spill bytes per hardware thread", outs[2])
    assert retried is not None, f"unexpected retry log line: {outs[1]!r}"
    assert selected is not None, f"unexpected selection log line: {outs[2]!r}"
    assert retried.group(1) == selected.group(1)


@pytest.mark.parametrize("warp_size", [16, 32])
def test_n_spills_reported_per_lane(device, monkeypatch, capfd, warp_size):
    """`n_spills` is dword-equivalents per lane, as on CUDA/HIP (issue #7896).

    Level Zero reports `spillMemSize` in bytes per hardware thread, so the value
    handed to Python is `bytes // (4 * SIMD)`. Both operands come from the log
    line for the *selected* pass rather than from the request, because the compiled
    width can differ from the requested `warp_size` under the auto-GRF retry; that
    agreement is asserted separately so a divergence fails loudly instead of
    being absorbed into the arithmetic.

    Also checks the rebuild gate, which needs both logs: the selected-pass line
    reports the post-retry spill once a rebuild succeeds, so only the retry
    announcement testifies about the decision. The threshold equality holds on every
    driver line -- it is a constant either way -- but it licenses no claim that the
    two *gates* agree, since driver.c has no `is_lts` input and compiler.py rebuilds
    on any spill for LTS.
    """
    monkeypatch.setenv("TRITON_DEBUG", "1")
    # Same fixture size as test_auto_grf: CRI's larger (512-GRF) register file
    # needs a bigger tile to spill; other targets already spill at 8K.
    BLOCK = 1024 * 32 if is_xpu_cri() else 1024 * 8
    z_tri = torch.empty(BLOCK, dtype=torch.int32, device=device)

    # Known-spilling fixture, shared with test_auto_grf.
    @triton.jit
    def _kernel(z, BLOCK: tl.constexpr):
        # make it hard to re-schedule.
        off = tl.arange(0, BLOCK)
        a = tl.load(z + off)
        result = tl.sum(a, axis=0, keep_dims=True)
        tl.store(z + off, a + result)

    kernel = _kernel[(1, )](z_tri, BLOCK=BLOCK, num_warps=2, warp_size=warp_size)

    out = capfd.readouterr().out
    selected = re.compile(r"Detected (\d+) spill bytes per hardware thread; "
                          r"n_spills (\d+) dword-equivalents/lane \(SIMD(\d+)\), "
                          r"rebuild at (\d+) B/hardware-thread")
    # Only the spill path logs numbers: the build-failure path enters the same branch
    # with an unknown `Spills` and reaches neither the threshold nor this format.
    retried = re.compile(r"Detected spills for \"[^\"]*\", retrying with large GRF mode "
                         r"\(\w+, spill (\d+) B/hardware-thread = (\d+) dword-equivalents/lane "
                         r"at SIMD(\d+), rebuild at (\d+) B/hardware-thread\)")
    # Keep the last match: it describes the finally selected binary.
    matches = selected.findall(out)
    if not matches:
        pytest.skip(f"fixture no longer spills on this IGC version; log was:\n{out}")
    spill_bytes, logged_slots, logged_simd, logged_threshold = (int(group) for group in matches[-1])

    # Pin the divisor against the request, so a compiled-width divergence is a
    # failure rather than something the arithmetic below hides.
    assert logged_simd == warp_size, f"compiled SIMD {logged_simd} != requested warp_size {warp_size}"
    assert kernel.metadata.threads_per_warp == warp_size

    assert spill_bytes > 0
    assert logged_slots == spill_bytes // (4 * warp_size)
    assert kernel.n_spills == logged_slots
    # The unit really changed: raw bytes must not reach Python any more.
    assert kernel.n_spills < spill_bytes

    # driver.c is compiled at runtime and cannot import the Python constant, so the
    # threshold is duplicated; this is what catches the two copies drifting apart.
    assert logged_threshold == REBUILD_SPILL_BYTES_PER_THREAD

    # Which log testifies about the decision depends on whether a rebuild happened, so
    # the oracle has two sides and each is valid only where the other is not.
    rebuilds = retried.findall(out)
    if rebuilds:
        pre_bytes, pre_threshold = (int(rebuilds[-1][i]) for i in (0, 3))
        assert pre_bytes >= pre_threshold, f"rebuilt below the threshold: {rebuilds[-1]}"
        assert pre_threshold == REBUILD_SPILL_BYTES_PER_THREAD
    else:
        assert spill_bytes < logged_threshold, (f"accepted {spill_bytes} B/hardware-thread at or above the "
                                                f"{logged_threshold} B rebuild threshold")


def test_n_spills_zero_without_spills(device):
    """A kernel that allocates no scratch reports 0, not the -1 error sentinel."""

    @triton.jit
    def _tiny(x_ptr, y_ptr):
        tl.store(y_ptr, tl.load(x_ptr))

    x = torch.ones(1, dtype=torch.float32, device=device)
    y = torch.empty(1, dtype=torch.float32, device=device)
    kernel = _tiny[(1, )](x, y)
    assert kernel.n_spills == 0


@pytest.fixture
def no_icpx(monkeypatch):
    """Hide `icpx`, which `find_sycl_icpx` checks first and would return early on."""
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda cmd, *args, **kwargs: None
                        if cmd == "icpx" else real_which(cmd, *args, **kwargs))


@pytest.mark.parametrize("layout, warns", [("no_compiler", False), ("header_only", True), ("lib_only", True)])
def test_find_sycl_skips_oneapi_root(monkeypatch, no_icpx, recwarn, tmp_path: pathlib.Path, layout, warns):
    """`ONEAPI_ROOT` is used only when a SYCL install is really under it.

    Every oneAPI component's `setvars.sh` sets `ONEAPI_ROOT`, so it does not mean a compiler is
    installed. Using it anyway hid a working `intel-sycl-rt` and broke the next build with
    `fatal error: sycl/sycl.hpp: No such file or directory`.
    See https://github.com/intel/intel-xpu-backend-for-triton/issues/7977.

    A root with no `compiler` directory is just a component install, so it is skipped quietly. Half
    an install is worth a warning: the header and the library directory are always used together,
    and with only the header the build reaches the link step and fails with `cannot find -lsycl`.
    """
    compiler_root = tmp_path / "compiler" / "latest"
    if layout == "no_compiler":
        (tmp_path / "dummy_component").mkdir()  # some other component, but no compiler
    elif layout == "header_only":
        (compiler_root / "include" / "sycl").mkdir(parents=True)
        (compiler_root / "include" / "sycl" / "sycl.hpp").touch()
    else:
        (compiler_root / "lib").mkdir(parents=True)
    monkeypatch.setenv("ONEAPI_ROOT", str(tmp_path))

    include_dir, sycl_dirs = find_sycl_icpx([])

    # Check the leak first: it is the real bug, and it gives the clearer failure message.
    assert not any(str(tmp_path) in d for d in include_dir + sycl_dirs), \
        f"rejected ONEAPI_ROOT leaked into the compiler flags: {include_dir + sycl_dirs}"
    warned = [str(w.message) for w in recwarn]
    if warns:
        assert any("does not provide SYCL" in m for m in warned), f"half an install was skipped silently: {warned}"
    else:
        assert not warned, f"unexpected warnings: {warned}"


def test_find_sycl_uses_oneapi_root(monkeypatch, no_icpx, recwarn, tmp_path: pathlib.Path):
    """A `ONEAPI_ROOT` with a full SYCL install is still used, ahead of the wheel."""
    compiler_root = tmp_path / "compiler" / "latest"
    (compiler_root / "include" / "sycl").mkdir(parents=True)
    (compiler_root / "include" / "sycl" / "sycl.hpp").touch()
    (compiler_root / "lib").mkdir()
    monkeypatch.setenv("ONEAPI_ROOT", str(tmp_path))

    include_dir, sycl_dirs = find_sycl_icpx([])

    assert sycl_dirs == [str(compiler_root / "lib")]
    assert str(compiler_root / "include") in include_dir
    assert str(compiler_root / "include" / "sycl") in include_dir
    assert not [str(w.message) for w in recwarn], f"unexpected warnings: {[str(w.message) for w in recwarn]}"


def _write_shared_library(path: pathlib.Path, soname: str):
    """Writes the smallest 64-bit ELF shared library with a soname: no code, just the dynamic section."""
    base, dynamic_offset = 0x1000, 64 + 2 * 56  # the ELF header, then two program headers
    strtab_offset = dynamic_offset + 3 * 16
    dynamic = struct.pack("<qQqQqQ", 5, base + strtab_offset, 14, 1, 0, 0)  # DT_STRTAB, DT_SONAME, DT_NULL
    strtab = b"\0" + soname.encode() + b"\0"
    size = strtab_offset + len(strtab)
    header = b"\x7fELF\x02\x01\x01" + bytes(9) + struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 64, 0, 0, 64, 56, 2, 64, 0,
                                                             0)
    load = struct.pack("<IIQQQQQQ", 1, 4, 0, base, base, size, size, 0x1000)  # PT_LOAD of the whole file
    dynamic_header = struct.pack("<IIQQQQQQ", 2, 4, dynamic_offset, base + dynamic_offset, base + dynamic_offset,
                                 len(dynamic), len(dynamic), 8)  # PT_DYNAMIC
    path.write_bytes(header + load + dynamic_header + dynamic + strtab)


def _make_sycl_install(root: pathlib.Path, soname: str, headers: bool = True, wheel: bool = False) -> pathlib.Path:
    """Lays out a SYCL runtime as oneAPI or, with `wheel`, the `intel-sycl-rt` wheel does, and returns its library."""
    library = root / "lib" / soname
    library.parent.mkdir(parents=True)
    _write_shared_library(library, soname)
    if wheel:
        # A wheel cannot hold symlinks, so `libsycl.so` is a copy.
        shutil.copyfile(library, library.with_name("libsycl.so"))
    else:
        library.with_name("libsycl.so").symlink_to(soname)
    if headers:
        (root / "include" / "sycl").mkdir(parents=True)
        (root / "include" / "sycl" / "sycl.hpp").touch()
    return library


def _fake_sycl_setup(monkeypatch, tmp_path: pathlib.Path, loaded: str, torch_headers: bool = True):
    """An `icpx` from oneAPI 2025.3 on `PATH` next to the newer SYCL runtime of PyTorch's wheels.

    Makes the runtimes named by `loaded` ("oneapi", "torch", "both" or "nothing") look loaded into
    the process, or makes the process look unable to tell ("unreadable"). "torch_two" maps two
    runtimes from PyTorch's directory; "torch_deleted" and "oneapi_deleted" map a runtime after it
    was removed from disk. Returns oneAPI's compiler root as `icpx` reports it, PyTorch's root, and
    the file the fake `icpx` creates when it runs.
    """
    # As in oneAPI, `latest` is a symlink, and the process maps the library by its real path.
    oneapi = tmp_path / "oneapi" / "compiler" / "latest"
    oneapi_lib = _make_sycl_install(oneapi.with_name("2025.3"), "libsycl.so.8")
    oneapi.symlink_to("2025.3")
    icpx_ran = tmp_path / "icpx_ran"
    icpx = oneapi / "bin" / "icpx"
    icpx.parent.mkdir()
    icpx.write_text(f"#!/bin/sh\ntouch '{icpx_ran}'\nexit 1\n")
    icpx.chmod(0o755)
    monkeypatch.setenv("PATH", f"{icpx.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("TRITON_INTEL_SYCL_COMPILER", raising=False)

    torch_root = tmp_path / "venv"
    torch_lib = _make_sycl_install(torch_root, "libsycl.so.9", headers=torch_headers, wheel=True)

    maps = tmp_path / "maps"
    if loaded != "unreadable":
        mapped = {
            "oneapi": [oneapi_lib], "torch": [torch_lib], "both": [torch_lib, oneapi_lib], "nothing": [], "torch_two":
            [torch_lib,
             torch_lib.with_name("libsycl.so.8")], "torch_deleted": [torch_lib], "oneapi_deleted": [oneapi_lib]
        }[loaded]
        # The kernel marks a mapped file that was unlinked since. oneAPI's `libsycl.so` symlink then dangles.
        suffix = ""
        if loaded.endswith("_deleted"):
            mapped[0].unlink()
            suffix = " (deleted)"
        lines = [
            "01f17000-0a5d7000 rw-p 00000000 00:00 0                                  [heap]",
            "71d16c800000-71d16e600000 rw-p 00000000 00:00 0 ",
        ]
        for inode, library in enumerate(mapped, start=3184608):
            lines += [
                f"71d182000000-71d1820fd000 r--p 00000000 fc:01 {inode}                    {library}{suffix}",
                f"71d1820fd000-71d1823d1000 r-xp 000fc000 fc:01 {inode}                    {library}{suffix}",
            ]
        maps.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(intel_driver, "_PROC_SELF_MAPS", str(maps))
    return oneapi, torch_root, icpx_ran


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the loaded SYCL runtime is found through /proc")
@pytest.mark.parametrize("loaded, torch_headers, builds_against", [
    pytest.param("torch", True, "torch", id="issue_8200"),
    pytest.param("oneapi", True, "oneapi", id="icpx_runtime_loaded"),
    pytest.param("nothing", True, "oneapi", id="no_runtime_loaded"),
    pytest.param("both", True, "oneapi", id="unknown_which_runtime_pytorch_uses"),
    pytest.param("torch_two", True, "oneapi", id="unknown_which_runtime_in_one_directory_pytorch_uses"),
    pytest.param("unreadable", True, "oneapi", id="not_linux"),
    pytest.param("torch", False, "oneapi", id="no_headers_to_build_against"),
    pytest.param("torch_deleted", True, "oneapi", id="loaded_runtime_removed_from_disk"),
    pytest.param("oneapi_deleted", True, "oneapi", id="loaded_runtime_removed_from_disk_leaving_dangling_symlink"),
])
def test_find_sycl_prefers_loaded_runtime(monkeypatch, recwarn, tmp_path: pathlib.Path, loaded, torch_headers,
                                          builds_against):
    """Triton's helpers are built against the SYCL runtime PyTorch has already loaded.

    PyTorch hands them its `sycl::queue`, so building them against another runtime puts two SYCL
    runtimes with different ABIs into one process. With PyTorch 2.13 wheels (SYCL 2026.0) and an
    `icpx` from oneAPI 2025.3 on `PATH`, the first call on the queue segfaulted in
    `sycl::context::get_devices()`.
    See https://github.com/intel/intel-xpu-backend-for-triton/issues/8200.
    """
    oneapi, torch_root, _ = _fake_sycl_setup(monkeypatch, tmp_path, loaded, torch_headers)
    expected, other = (torch_root, tmp_path / "oneapi") if builds_against == "torch" else (oneapi, torch_root)

    helper = CompilationHelper()

    assert helper.libsycl_dir == [str(expected / "lib")]
    assert str(expected / "include" / "sycl") in helper.include_dir
    assert not any(str(other) in d for d in helper.include_dir + helper.library_dir), \
        f"the other SYCL runtime leaked into the compiler flags: {helper.include_dir + helper.library_dir}"
    # `icpx` adds its own SYCL to the build, so it may build only against that one.
    assert helper.use_sycl_compiler == (builds_against == "oneapi")
    warned = [str(w.message) for w in recwarn]
    # One runtime is known to be the one loaded, yet cannot be built against.
    unusable = {
        "torch": torch_root / "lib", "torch_deleted": torch_root / "lib", "oneapi_deleted":
        oneapi.with_name("2025.3") / "lib"
    }.get(loaded)
    if unusable and builds_against == "oneapi":
        assert any(str(unusable) in m for m in warned), f"a possible crash was not reported: {warned}"
    else:
        assert not warned, f"unexpected warnings: {warned}"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the loaded SYCL runtime is found through /proc")
def test_find_sycl_checks_which_runtime_links(monkeypatch, recwarn, tmp_path: pathlib.Path):
    """Holding the loaded runtime does not make a directory safe to build against.

    Here oneAPI's directory also holds a newer runtime, which `libsycl.so` links, so a helper built
    against it would load that one next to the one already loaded.
    """
    oneapi, _, _ = _fake_sycl_setup(monkeypatch, tmp_path, "oneapi")
    lib = oneapi.with_name("2025.3") / "lib"
    _write_shared_library(lib / "libsycl.so.9", "libsycl.so.9")
    (lib / "libsycl.so").unlink()
    (lib / "libsycl.so").symlink_to("libsycl.so.9")

    helper = CompilationHelper()

    assert helper.libsycl_dir == [str(oneapi / "lib")]
    warned = [str(w.message) for w in recwarn]
    assert any(str(lib / "libsycl.so.8") in m for m in warned), f"a possible crash was not reported: {warned}"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the loaded SYCL runtime is found through /proc")
def test_helper_cache_key_follows_runtime_upgrade_in_place(monkeypatch, request, tmp_path: pathlib.Path):
    """A cached helper is not reused after the SYCL runtime in the same directory changes.

    `pip install -U` replaces the runtime without moving it, so the directory in the cache key stays
    the same while the helper built before still needs the old soname. The cache is consulted before
    any helper is loaded, so no runtime is mapped yet.
    """
    oneapi, _, _ = _fake_sycl_setup(monkeypatch, tmp_path, "nothing")
    request.addfinalizer(intel_driver.get_hasher_common.cache_clear)

    def cache_key() -> tuple[list[str], str]:
        monkeypatch.setattr(intel_driver, "COMPILATION_HELPER", CompilationHelper())
        intel_driver.get_hasher_common.cache_clear()
        return intel_driver.COMPILATION_HELPER.libsycl_dir, intel_driver.get_hasher_common().hexdigest()

    dirs_before, key_before = cache_key()
    lib = oneapi.with_name("2025.3") / "lib"
    _write_shared_library(lib / "libsycl.so.9", "libsycl.so.9")
    (lib / "libsycl.so").unlink()
    (lib / "libsycl.so").symlink_to("libsycl.so.9")
    dirs_after, key_after = cache_key()

    assert dirs_before == dirs_after == [str(oneapi / "lib")]
    assert key_before != key_after


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads the libraries mapped into this process")
def test_soname_of_a_real_library():
    """`_soname` reads real shared libraries, not only the fakes above: libc's soname is the same everywhere."""
    with open("/proc/self/maps") as maps:
        mapped = {
            fields[5]
            for fields in (line.split(maxsplit=5) for line in maps.read().splitlines())
            if len(fields) == 6
        }
    libc = next((path for path in mapped if re.fullmatch(r"libc(\.so\.6|-[\d.]+\.so)", os.path.basename(path))), None)
    if libc is None:
        pytest.skip(f"no glibc mapped into this process: {sorted(mapped)}")

    assert intel_driver._soname(libc) == "libc.so.6"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the loaded SYCL runtime is found through /proc")
@pytest.mark.parametrize("loaded, runs_icpx", [("torch", False), ("oneapi", True)])
def test_helpers_built_without_icpx_for_loaded_runtime(monkeypatch, request, tmp_path: pathlib.Path, loaded, runs_icpx):
    """Helpers built against PyTorch's SYCL runtime are compiled by the host compiler, not `icpx`.

    `icpx` adds its own SYCL headers and runtime to the build, which need not be PyTorch's.
    """
    _, _, icpx_ran = _fake_sycl_setup(monkeypatch, tmp_path, loaded)
    monkeypatch.setattr(intel_driver, "COMPILATION_HELPER", CompilationHelper())
    # `get_hasher_common` caches a hash of the fake helper above.
    request.addfinalizer(intel_driver.get_hasher_common.cache_clear)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("CXX", raising=False)
    # The SYCL compiler is chosen only on XPU; do not depend on a device being present.
    monkeypatch.setattr(build, "is_xpu", lambda: True)

    # The build fails with either compiler, so no fake runtime is ever loaded; what is checked is
    # which compiler ran.
    with pytest.raises(RuntimeError):
        intel_driver.compile_module_from_src("#error stop after choosing the compiler\n", "sycl_compiler_choice")

    assert icpx_ran.exists() == runs_icpx


def test_get_properties_error(device):
    device_count, = driver.active.utils.device_count

    with pytest.raises(RuntimeError, match="Device is not found"):
        # Expected an exception when querying an invalid device index
        driver.active.utils.get_device_properties(device_count)


def test_load_binary_error_device_error(device, tmp_path: pathlib.Path):
    ir = """
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64"} {
      tt.func public @empty_func() {
        tt.return
      }
    }
    """

    temp_file = tmp_path / "test_regression_load_binary_error.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    device_count, = driver.active.utils.device_count

    with pytest.raises(RuntimeError, match="Device is not found"):
        # Expected an exception when loading binary on an invalid device index
        _ = driver.active.utils.load_binary(kernel.name, kernel.kernel, kernel.metadata.shared,
                                            kernel.metadata.build_flags, not kernel.metadata.generate_native_code,
                                            device_count)


def test_load_binary_error_kernel_error(device, tmp_path: pathlib.Path):
    ir = """
    module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64"} {
      tt.func public @empty_func() {
        tt.return
      }
    }
    """

    temp_file = tmp_path / "test_regression_load_binary_error.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    device = driver.active.get_current_device()

    with pytest.raises(IntelGPUError, match=r".*ZE_RESULT_ERROR_INVALID_KERNEL_NAME.*"):
        _ = driver.active.utils.load_binary("invalid name", kernel.kernel, kernel.metadata.shared,
                                            kernel.metadata.build_flags, not kernel.metadata.generate_native_code,
                                            device)


def test_wait_on_sycl_queue_error(device):
    # Pass an invalid (non-pointer) value to trigger conversion error
    with pytest.raises(RuntimeError, match=r"Failed to convert PyObject to void\* for queue.*"):
        driver.active.utils.wait_on_sycl_queue("invalid_queue_pointer")


def test_has_opencl_extension_error(device):
    device_idx = torch.xpu.current_device()
    device_id = extension_utils.get_device_id(device_idx)

    # Test that we can query extensions using the new API
    extensions = extension_utils.query_device_extensions(device_id=device_id)

    # Verify we got a dictionary with expected extension keys
    assert isinstance(extensions, dict)
    assert "has_subgroup_matrix_multiply_accumulate" in extensions
    assert "has_subgroup_matrix_multiply_accumulate_tensor_float32" in extensions
    assert "has_2d_block_io" in extensions
    assert "has_bfloat16_conversion" in extensions
    if device_id == 3034:
        # PVC 1100
        assert extensions["has_subgroup_matrix_multiply_accumulate"] is True
        assert extensions["has_subgroup_matrix_multiply_accumulate_tensor_float32"] is False
        assert extensions["has_2d_block_io"] is True
        assert extensions["has_bfloat16_conversion"] is True

    # Test individual extension checking
    result = extension_utils.has_device_extension(device_id, "cl_intel_subgroup_2d_block_io")
    assert isinstance(result, bool)
    if device_id == 3034:
        # PVC 1100
        assert result is True  # This extension should be supported

    # Test checking for a non-existent/wrong extension name
    result_wrong = extension_utils.has_device_extension(device_id, "cl_intel_nonexistent_extension")
    assert isinstance(result_wrong, bool)
    assert result_wrong is False  # This extension should not be supported

    assert extension_utils.has_device_extension(9999, "cl_intel_subgroup_2d_block_io") is None


@pytest.mark.parametrize("grf_mode, expect_retry", [("default", True),  # Should auto-retry with large GRF and succeed
                                                    ("256", False),  # Explicit large GRF — compiles on first attempt
                                                    ("128", False),  # Explicit small GRF — should fail, no retry
                                                    ])
@pytest.mark.parametrize("generate_native_code", [False, True], ids=["load_binary", "make_zebin"])
def test_auto_grf_on_build_failure(device, monkeypatch, capfd, grf_mode, expect_retry, generate_native_code):
    """Test GRF mode behavior for register-heavy kernels on both compilation paths:
    - load_binary (generate_native_code=False): L0 runtime compilation via zeModuleCreate
    - make_zebin (generate_native_code=True): offline compilation via ocloc
    """
    monkeypatch.setenv("TRITON_DEBUG", "1")

    @triton.jit
    def _register_heavy_kernel(
        output_ptr,
        input_ptr,
        q_ptr,
        size,
        BLOCK: tl.constexpr,
    ):
        off = tl.arange(0, BLOCK)
        mask = off < size
        x = tl.load(input_ptr + off, mask=mask, other=0.0)
        q = tl.load(q_ptr + off, mask=mask, other=float("-inf"))
        result = tl.argmax(x / q, axis=-1)
        tl.store(output_ptr, result)

    BLOCK = 131072  # Large enough to exceed PTSS with default/small GRF
    size = 128000

    x = torch.randn(size, dtype=torch.float32, device=device)
    q = torch.rand(size, dtype=torch.float32, device=device)
    out = torch.empty(1, dtype=torch.int32, device=device)

    try:
        _register_heavy_kernel[(1, )](out, x, q, size, BLOCK=BLOCK, grf_mode=grf_mode,
                                      generate_native_code=generate_native_code)
    except (IntelGPUError, OutOfResources):
        # OutOfResources is the new spill-related error class introduced by
        # the PTSS-overflow handling in this PR; both error types are
        # acceptable here since this test exercises a kernel intentionally
        # too large for the chosen GRF mode.
        pass

    outs = capfd.readouterr().out
    if expect_retry and not generate_native_code:
        # load_binary path prints a retry message to stdout.
        assert "retrying with large GRF mode" in outs
    elif expect_retry and generate_native_code:
        # make_zebin path retries silently via ocloc — no stdout message.
        # Success without exception is sufficient verification.
        pass
    else:
        assert "retrying with large GRF mode" not in outs
        assert "Build failed" not in outs


def test_sycl_global_range_overflow(device):
    # for details: https://github.com/intel/intel-xpu-backend-for-triton/issues/7201

    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0).to(tl.int64)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    n = 1379584
    x = torch.randint(0, 100, (n, 2048), dtype=torch.int8, device=device)
    output = torch.empty_like(x)
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    add_kernel[grid](x, x, output, n_elements, 16)

    torch.testing.assert_close(output.cpu(), (x + x).cpu(), rtol=0, atol=0)
