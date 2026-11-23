import importlib.metadata
import importlib.util
import mmap
import os
import json
import re
import struct
import sys
import hashlib
import shutil
import ctypes
import subprocess
import sysconfig
import tempfile
import warnings
from pathlib import Path
from functools import cached_property, lru_cache

from triton import knobs
import triton
from triton.runtime import _allocation
from triton.runtime.build import _build, platform_key, _load_module_from_path
from triton.runtime.cache import get_cache_manager
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase, TensorDescABI, decompose_descriptor
from triton.backends.driver import expand_signature, get_kernel_argument_layout, wrap_handle_tensordesc_impl
from . import native

# A hard-coded cache version that can be updated when we know that the cached file is invalid and
# there are no other ways to detect that the runtime environment has changed. For example, a shared
# library has been updated as a result of updated dependencies.
# See https://github.com/intel/intel-xpu-backend-for-triton/issues/3095.
__CACHE_VERSION = "1"

PyKernelArg = None
ARG_CONSTEXPR = None
ARG_KERNEL = None
ARG_TUPLE = None


def _find_sycl_from_icpx(include_dir: list[str]) -> tuple[list[str], list[str]]:
    """The SYCL of the `icpx` on `PATH`: only the compiler knows where its runtime and headers are."""
    include_dir = include_dir.copy()
    icpx_path = shutil.which("icpx")
    if not icpx_path:
        return include_dir, []
    compiler_root = os.path.abspath(f"{icpx_path}/../..")
    include_dir += [os.path.join(compiler_root, "include"), os.path.join(compiler_root, "include", "sycl")]
    return include_dir, [os.path.join(compiler_root, "lib")]


def _find_sycl_from_oneapi_root(include_dir: list[str]) -> tuple[list[str], list[str]]:
    """The SYCL of the compiler under `ONEAPI_ROOT`, if the variable names an install that has one."""
    include_dir = include_dir.copy()
    oneapi_root = os.getenv("ONEAPI_ROOT")
    if not oneapi_root:
        return include_dir, []
    # Any oneAPI component's `setvars.sh` sets `ONEAPI_ROOT`, so it does not mean a compiler is
    # installed here (a VTune-only install has no `compiler` directory at all). Check that both
    # paths exist before returning them.
    # See https://github.com/intel/intel-xpu-backend-for-triton/issues/7977.
    compiler_root = os.path.join(oneapi_root, "compiler", "latest")
    sycl_dir = os.path.join(compiler_root, "lib")
    if os.path.isfile(os.path.join(compiler_root, "include", "sycl", "sycl.hpp")) and os.path.isdir(sycl_dir):
        include_dir += [os.path.join(compiler_root, "include"), os.path.join(compiler_root, "include", "sycl")]
        return include_dir, [sycl_dir]
    if os.path.isdir(compiler_root):
        # A compiler directory with only half of SYCL in it is not something a normal install
        # leaves behind, and the SYCL found below may be a different version than the one
        # meant to be used, so do not skip it silently.
        warnings.warn(
            f"{compiler_root} does not provide SYCL (need include/sycl/sycl.hpp and lib); "
            "ignoring ONEAPI_ROOT and looking for SYCL elsewhere.",
            stacklevel=3,
        )
    return include_dir, []


def _find_sycl_from_wheel(include_dir: list[str]) -> tuple[list[str], list[str]]:
    """The SYCL of the `intel-sycl-rt` wheel, which PyTorch's XPU wheels depend on."""
    include_dir = include_dir.copy()
    try:
        sycl_rt = importlib.metadata.metadata("intel-sycl-rt")
    except importlib.metadata.PackageNotFoundError:
        return include_dir, []

    if sycl_rt.get("version", "0.0.0").startswith("2024"):
        return include_dir, []

    sycl_dirs = []
    for f in importlib.metadata.files("intel-sycl-rt"):
        # sycl/sycl.hpp and sycl/CL/sycl.hpp results in both folders
        # being add: include and include/sycl.
        if "sycl.hpp" in f.name:
            include_dir += [str(f.locate().parent.parent.resolve())]
        # Only the library itself: the wheel also ships `libsycl.so.N-gdb.py` and its `__pycache__`.
        if re.fullmatch(r"libsycl\.so(\.\d+)*|sycl\.lib", f.name):
            sycl_dir = f.locate().parent.resolve()
            if os.name == "nt":
                # for sycl8.dll loading on Windows
                dll_path = sycl_dir.parent.joinpath("bin")
                if str(dll_path) not in sycl_dirs:
                    sycl_dirs.append(str(dll_path))
                    _ = os.add_dll_directory(str(dll_path))
            if str(sycl_dir) not in sycl_dirs:
                sycl_dirs.append(str(sycl_dir))

    return include_dir, sycl_dirs


def find_sycl_icpx(include_dir: list[str]) -> tuple[list[str], list[str]]:
    """The first SYCL found among `icpx` on `PATH`, `ONEAPI_ROOT` and the `intel-sycl-rt` wheel."""
    for finder in (_find_sycl_from_icpx, _find_sycl_from_oneapi_root, _find_sycl_from_wheel):
        found_include_dir, sycl_dirs = finder(include_dir)
        if sycl_dirs:
            return found_include_dir, sycl_dirs
    return include_dir.copy(), []


def find_sycl_dpclang(include_dir: list[str]) -> tuple[list[str], list[str]]:
    include_dir = include_dir.copy()

    if not shutil.which("pkg-config"):
        return include_dir, []

    dpclang = shutil.which(knobs.intel.sycl_compiler if knobs.intel.sycl_compiler else "dpclang++")
    cmd = [dpclang, "-E", "-dM", "-"]

    major = None
    try:
        result = subprocess.run(cmd, input="", capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if "__dpcpp_major__" in line:
                # The line looks like: "#define __dpcpp_major__ 7"
                parts = line.split()
                if len(parts) >= 3:
                    major = int(parts[-1])
                    break
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return include_dir, []

    if major is None:
        return include_dir, []

    package_name = f"sycl-dpcpp-{major}"
    sycl_dirs = []
    try:
        cflags_I_res = subprocess.run(
            ["pkg-config", "--cflags-only-I", package_name],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip().split()
        include_dir += [flag[2:] for flag in cflags_I_res if flag.startswith("-I") and len(flag) > 2]

        libs_L_res = subprocess.run(
            ["pkg-config", "--libs-only-L", package_name],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip().split()
        sycl_dirs = [flag[2:] for flag in libs_L_res if flag.startswith("-L") and len(flag) > 2]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return include_dir, sycl_dirs


def _elf_dynamic(path: str) -> tuple[str | None, list[str]] | None:
    """Returns the soname and the needed libraries of the 64-bit little-endian ELF file at `path`.

    None if it is not such a file. Read from the dynamic section, so no toolchain is needed.
    """
    try:
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as elf:
            if elf[:6] != b"\x7fELF\x02\x01":
                return None
            phoff, = struct.unpack_from("<Q", elf, 0x20)
            phentsize, phnum = struct.unpack_from("<HH", elf, 0x36)
            loads, dynamic = [], range(0)
            for i in range(phnum):
                p_type, _, p_offset, p_vaddr, _, p_filesz = struct.unpack_from("<IIQQQQ", elf, phoff + i * phentsize)
                if p_type == 1:  # PT_LOAD
                    loads.append((p_vaddr, p_offset, p_filesz))
                elif p_type == 2:  # PT_DYNAMIC
                    dynamic = range(p_offset, p_offset + p_filesz, 16)
            strtab, soname, needed = None, None, []
            for entry in dynamic:
                tag, value = struct.unpack_from("<qQ", elf, entry)
                if tag == 0:  # DT_NULL
                    break
                if tag == 1:  # DT_NEEDED
                    needed.append(value)
                elif tag == 5 and strtab is None:  # DT_STRTAB
                    strtab = value
                elif tag == 14 and soname is None:  # DT_SONAME
                    soname = value
            if strtab is None:
                return None, []
            # DT_STRTAB is the address the string table is loaded at, so find it in the file.
            for p_vaddr, p_offset, p_filesz in loads:
                if p_vaddr <= strtab < p_vaddr + p_filesz:
                    file_strtab = p_offset + strtab - p_vaddr

                    def string(offset: int) -> str | None:
                        start = file_strtab + offset
                        end = elf.find(b"\0", start)
                        return elf[start:end].decode() if end >= 0 else None

                    return (None if soname is None else string(soname),
                            [name for name in map(string, needed) if name is not None])
            return None, []
    except (OSError, ValueError, struct.error):  # unreadable, empty or truncated
        return None


def _soname(path: str) -> str | None:
    """Returns the soname of the ELF shared library at `path`, if it has one."""
    dynamic = _elf_dynamic(path)
    return dynamic[0] if dynamic else None


def _pe_imports(path: str) -> list[str] | None:
    """Returns the names of the DLLs the PE image at `path` imports, or None if it is not a PE image."""
    try:
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as pe:
            if pe[:2] != b"MZ":
                return None
            pe_offset, = struct.unpack_from("<I", pe, 0x3C)
            if pe[pe_offset:pe_offset + 4] != b"PE\0\0":
                return None
            coff = pe_offset + 4
            num_sections, = struct.unpack_from("<H", pe, coff + 2)
            optional_size, = struct.unpack_from("<H", pe, coff + 16)
            optional = coff + 20
            magic, = struct.unpack_from("<H", pe, optional)
            # The data directories follow the optional header's fixed part: 96 bytes in PE32, 112 in PE32+.
            directories = optional + (112 if magic == 0x20B else 96)
            import_rva, = struct.unpack_from("<I", pe, directories + 8)  # the second directory
            if import_rva == 0:
                return []
            sections = []
            for i in range(num_sections):
                virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
                    "<IIII", pe, optional + optional_size + i * 40 + 8)
                sections.append((virtual_address, max(virtual_size, raw_size), raw_offset))

            def file_offset(rva: int) -> int:
                for virtual_address, size, raw_offset in sections:
                    if virtual_address <= rva < virtual_address + size:
                        return raw_offset + rva - virtual_address
                raise ValueError(f"RVA {rva:#x} is in no section")

            names = []
            entry = file_offset(import_rva)
            while True:
                lookup, _, _, name_rva, _ = struct.unpack_from("<IIIII", pe, entry)
                if lookup == 0 and name_rva == 0:
                    break
                start = file_offset(name_rva)
                end = pe.find(b"\0", start)
                if end < 0:
                    raise ValueError("unterminated DLL name")
                names.append(pe[start:end].decode())
                entry += 20
            return names
    except (OSError, ValueError, struct.error):  # unreadable, empty or truncated
        return None


def _linked_libraries(path: str) -> list[str] | None:
    """Returns the libraries the shared library at `path` links, as the dynamic loader names them."""
    if os.name == "nt":
        return _pe_imports(path)
    dynamic = _elf_dynamic(path)
    return dynamic[1] if dynamic else None


# The file name the dynamic loader resolves a SYCL runtime by: `libsycl.so.N` (its soname) on Linux,
# `syclN.dll` on Windows.
_WINDOWS_SYCL_RUNTIME = re.compile(r"sycl\d*\.dll", re.IGNORECASE)
_SYCL_RUNTIME = _WINDOWS_SYCL_RUNTIME if os.name == "nt" else re.compile(r"libsycl\.so(\.\d+)*")

# The PyTorch libraries that link the SYCL runtime; both carry the dependency.
_TORCH_XPU_LIBRARIES = ("torch_xpu.dll", "c10_xpu.dll") if os.name == "nt" else ("libtorch_xpu.so", "libc10_xpu.so")


def _torch_xpu_libraries() -> list[str]:
    """Paths of the installed PyTorch's XPU libraries, found without importing PyTorch."""
    try:
        files = importlib.metadata.files("torch")
    except importlib.metadata.PackageNotFoundError:
        files = None
    libraries = [str(f.locate()) for f in files or [] if f.name in _TORCH_XPU_LIBRARIES]
    if libraries:
        return libraries
    # The metadata of an editable install (`pip install -e`, `setup.py develop`) lists only the files
    # that redirect to the source tree, and a tree on `PYTHONPATH` has no metadata at all. Look in the
    # package itself then, where `torch/lib` holds the libraries; finding a top-level package does
    # not import it.
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        spec = None
    for location in (spec.submodule_search_locations or []) if spec else []:
        for name in _TORCH_XPU_LIBRARIES:
            library = os.path.join(location, "lib", name)
            if os.path.isfile(library):
                libraries.append(library)
    return libraries


def _torch_sycl_runtime() -> str | None:
    """Returns the SYCL runtime PyTorch's XPU libraries link, as the loader names it (e.g. `libsycl.so.9`).

    PyTorch does not look SYCL up at run time: `libtorch_xpu.so` and `libc10_xpu.so` record the
    runtime's soname at build time (`libsycl.so.8` for 2025.x, `libsycl.so.9` for 2026.x) and the
    dynamic loader resolves it. That soname is read from the files here, so it needs neither PyTorch
    imported nor its runtime loaded yet. None if PyTorch is not installed or has no XPU support,
    or if its libraries do not tell or disagree; the last two warn, as the search then ignores PyTorch.
    """
    runtimes = {}
    libraries = _torch_xpu_libraries()
    for library in libraries:
        linked = _linked_libraries(library)
        if linked is None:
            continue
        found = {name for name in linked if _SYCL_RUNTIME.fullmatch(name)}
        if len(found) == 1:
            runtimes[library] = found.pop()
    if not runtimes:
        if libraries:
            warnings.warn(
                "Cannot tell which SYCL runtime PyTorch's XPU libraries link (" +
                ", ".join(os.path.basename(library) for library in libraries) +
                " could not be read, or link none or several). Triton builds its helpers as if PyTorch were not "
                "installed, which crashes if the SYCL runtime it picks has another ABI than PyTorch's.",
                stacklevel=3,
            )
        return None
    if len(set(runtimes.values())) > 1:
        warnings.warn(
            "PyTorch's XPU libraries link different SYCL runtimes: " +
            ", ".join(f"{os.path.basename(library)} needs {runtime}" for library, runtime in runtimes.items()) +
            ". Triton cannot tell which one its helpers must use and builds them as if PyTorch were not installed.",
            stacklevel=3,
        )
        return None
    return next(iter(runtimes.values()))


def _windows_runtime_in(sycl_dir: str) -> str | None:
    """Returns the `syclN.dll` of the Windows SYCL installation whose library directory is `sycl_dir`.

    The DLL is beside the import library, or in the sibling `bin`: that is where oneAPI and the
    `intel-sycl-rt` wheel keep it, with only `sycl.lib` in `lib`. The wheel's finder returns both
    directories, `icpx`'s and `ONEAPI_ROOT`'s `lib` alone. None if there is no DLL, or several:
    the names do not tell which one `sycl.lib` links.
    """
    sycl_dir = os.path.normpath(sycl_dir)
    for directory in (sycl_dir, os.path.join(os.path.dirname(sycl_dir), "bin")):
        try:
            dlls = [name for name in os.listdir(directory) if _WINDOWS_SYCL_RUNTIME.fullmatch(name)]
        except OSError:
            continue
        if dlls:
            return dlls[0] if len(dlls) == 1 else None
    return None


def _runtime_in(sycl_dirs: list[str]) -> str | None:
    """Returns the SYCL runtime `-lsycl` in `sycl_dirs` links, as the loader names it.

    On Linux that is the soname of `libsycl.so`: a helper is linked against that soname and the
    loader reuses a loaded library with the same one, whatever directory either comes from. The
    directory may hold other runtimes too, and a wheel ships `libsycl.so` as a copy, so neither the
    directory nor the file name says which. None if `libsycl.so` is missing or unreadable, e.g. a
    symlink left dangling by an uninstall. On Windows it is the `syclN.dll` of the installation.
    """
    for sycl_dir in sycl_dirs:
        if os.name == "nt":
            dll = _windows_runtime_in(sycl_dir)
            if dll is not None:
                return dll
        else:
            soname = _soname(os.path.join(sycl_dir, "libsycl.so"))
            if soname is not None:
                return soname
    return None


def _has_sycl_headers(include_dir: list[str]) -> bool:
    """Whether the helpers can be compiled against the SYCL headers in `include_dir`."""
    return any(os.path.isfile(os.path.join(directory, "sycl", "sycl.hpp")) for directory in include_dir)


def _links_runtime(sycl_dirs: list[str], runtime: str) -> bool:
    """Whether `-lsycl` in `sycl_dirs` links the SYCL runtime named `runtime`, which helpers then use."""
    found = _runtime_in(sycl_dirs)
    return found is not None and (found.lower() == runtime.lower() if os.name == "nt" else found == runtime)


def _sycl_header_version(include_dir: list[str]) -> int | None:
    """Returns `__SYCL_COMPILER_VERSION` of the SYCL headers in `include_dir`: the runtime's build date."""
    for directory in include_dir:
        try:
            with open(os.path.join(directory, "sycl", "version.hpp")) as header:
                for line in header:
                    if match := re.match(r"\s*#\s*define\s+__SYCL_COMPILER_VERSION\s+(\d+)", line):
                        return int(match.group(1))
        except OSError:
            continue
    return None


def find_sycl(include_dir: list[str]) -> tuple[list[str], list[str], bool]:
    """
    Looks for the sycl library in known places.

    Arguments:
      include_dir: list of include directories to pass to compiler.

    Returns:
      enriched include_dir, libsycl.so location, and whether a SYCL compiler may build against it.

    Raises:
      AssertionError: if library was not found.
    """

    # Every SYCL installation found, in the order they were preferred in before PyTorch's runtime
    # was taken into account: what it is, its include and library directories, and whether it is a
    # SYCL compiler's own, which that compiler may then build against.
    candidates: list[tuple[str, list[str], list[str], bool]] = []
    csycl = knobs.intel.sycl_compiler
    if not csycl or csycl == "icpx":
        finders = [
            ("icpx", _find_sycl_from_icpx, True),
            ("ONEAPI_ROOT", _find_sycl_from_oneapi_root, False),
            ("the intel-sycl-rt wheel", _find_sycl_from_wheel, False),
        ]
        for what, finder, is_compiler in finders:
            found_include_dir, sycl_dirs = finder(include_dir)
            if sycl_dirs:
                candidates.append((what, found_include_dir, sycl_dirs, is_compiler))
    if not candidates and (not csycl or csycl.startswith("dpclang")):
        found_include_dir, sycl_dirs = find_sycl_dpclang(include_dir)
        if sycl_dirs:
            candidates.append((csycl or "dpclang++", found_include_dir, sycl_dirs, True))
    if not candidates:
        raise AssertionError("sycl headers not found, please install `icpx` compiler, "
                             "or provide `ONEAPI_ROOT` environment "
                             "or install `intel-sycl-rt>=2025.0.0` wheel"
                             "or instal `dpclang` compiler (experimental)")

    # The helpers are handed PyTorch's `sycl::queue`, so they must use the SYCL runtime PyTorch
    # links: another one, with another ABI, crashes on the first call on the queue. Build against the
    # installation whose `libsycl.so` has that soname. Of several, take the one with the oldest
    # headers: a runtime runs code built against older headers of the same soname, never newer ones.
    # A SYCL compiler adds its own SYCL to the build, so it may only build against its own.
    # See https://github.com/intel/intel-xpu-backend-for-triton/issues/8200.
    chosen = candidates[0]
    runtime = _torch_sycl_runtime()
    if runtime is not None:
        matching = [
            candidate for candidate in candidates
            if _links_runtime(candidate[2], runtime) and _has_sycl_headers(candidate[1])
        ]
        if matching:
            chosen = min(
                matching, key=lambda candidate:
                (_sycl_header_version(candidate[1]) is None, _sycl_header_version(candidate[1]) or 0))
        else:
            found = ", ".join(f"{what}: {sycl_dirs[0]} links {_runtime_in(sycl_dirs) or 'no libsycl.so'}" +
                              ("" if _has_sycl_headers(found_include_dir) else " and has no headers")
                              for what, found_include_dir, sycl_dirs, _ in candidates)
            warnings.warn(
                f"PyTorch links the SYCL runtime {runtime}, but no SYCL installation found provides it ({found}). "
                f"Triton builds its helpers against {chosen[0]}, which crashes if the two runtimes' ABIs differ.",
                stacklevel=2,
            )
    _, include_dir, sycl_dirs, is_compiler = chosen
    return include_dir, sycl_dirs, is_compiler


class CompilationHelper:
    _library_dir: list[str]
    _include_dir: list[str]
    libraries: list[str]

    def __init__(self):
        self._library_dir = None
        self._include_dir = None
        self._libsycl_dir = None
        self._use_sycl_compiler = None
        self.libraries = ['sycl', 'ze_loader']

    @property
    def inject_pytorch_dep(self):
        return os.environ.get("INJECT_PYTORCH", "False") == "True"

    @cached_property
    def _compute_compilation_options_lazy(self):
        ze_root = os.getenv("LEVEL_ZERO_V1_SDK_PATH")
        if ze_root is None:
            ze_root = os.getenv("ZE_PATH", default="/usr/local")
        include_dir = [os.path.join(ze_root, "include")]

        library_dir = []
        include_dir, self._libsycl_dir, self._use_sycl_compiler = find_sycl(include_dir)
        if self._libsycl_dir:
            library_dir += self._libsycl_dir
        if os.name == "nt":
            library_dir += [os.path.join(ze_root, "lib")]

        dirname = os.path.dirname(os.path.realpath(__file__))
        include_dir += [os.path.join(dirname, "include")]
        library_dir += [os.path.join(dirname, "lib")]

        if self.inject_pytorch_dep:
            import torch

            torch_path = torch.utils.cmake_prefix_path
            include_dir += [
                os.path.join(torch_path, "../../include"),
                os.path.join(torch_path, "../../include/torch/csrc/api/include"),
            ]
            library_dir += [os.path.join(torch_path, "../../lib")]
            self.libraries += ['torch']

        self._library_dir = library_dir
        self._include_dir = include_dir

    @cached_property
    def library_dir(self) -> list[str]:
        self._compute_compilation_options_lazy
        return self._library_dir

    @cached_property
    def include_dir(self) -> list[str]:
        self._compute_compilation_options_lazy
        return self._include_dir

    @cached_property
    def libsycl_dir(self) -> list[str]:
        self._compute_compilation_options_lazy
        return self._libsycl_dir

    @cached_property
    def use_sycl_compiler(self) -> bool:
        self._compute_compilation_options_lazy
        return self._use_sycl_compiler


COMPILATION_HELPER = CompilationHelper()


class ArchParser:

    def __init__(self, cache_path: str):
        self.shared_library = ctypes.CDLL(cache_path)
        self.shared_library.parse_device_arch.restype = ctypes.c_char_p
        self.shared_library.parse_device_arch.argtypes = (ctypes.c_uint64, )

    def __getattribute__(self, name):
        if name == "parse_device_arch":
            shared_library = super().__getattribute__("shared_library")
            attr = getattr(shared_library, name)

            def wrapper(*args, **kwargs):
                return attr(*args, **kwargs).decode("utf-8")

            return wrapper

        return super().__getattribute__(name)

    if os.name != 'nt':

        def __del__(self):
            if hasattr(self, "shared_library"):
                handle = self.shared_library._handle
                self.shared_library.dlclose.argtypes = (ctypes.c_void_p, )
                self.shared_library.dlclose(handle)
    else:

        def __del__(self):
            if hasattr(self, "shared_library"):
                handle = self.shared_library._handle
                ctypes.windll.kernel32.FreeLibrary.argtypes = (ctypes.c_uint64, )
                ctypes.windll.kernel32.FreeLibrary(handle)


class SpirvUtils:

    def __init__(self, cache_path: str):
        self.shared_library = ctypes.PyDLL(cache_path)
        methods = ("init_devices", "load_binary", "wait_on_sycl_queue", "sycl_queue_memset", "launch",
                   "build_signature_metadata")
        for method in methods:
            getattr(self.shared_library, method).restype = ctypes.py_object
            getattr(self.shared_library, method).argtypes = (ctypes.py_object, )
        self.shared_library.get_device_properties.restype = ctypes.py_object
        self.shared_library.get_device_properties.argtypes = (ctypes.c_int, )
        self.shared_library.get_last_selected_build_flags.restype = ctypes.py_object

        self.shared_library.build_signature_metadata.restype = ctypes.py_object
        self.shared_library.build_signature_metadata.argtypes = (ctypes.py_object, )

        self.shared_library.init_PyKernelArgType.restype = ctypes.py_object
        self.shared_library.init_PyKernelArgType.argtypes = tuple()

    def __getattribute__(self, name):
        if name in ("get_device_properties", "init_devices", "wait_on_sycl_queue", "get_last_selected_build_flags",
                    "sycl_queue_memset", "build_signature_metadata", "init_PyKernelArgType"):
            shared_library = super().__getattribute__("shared_library")
            return getattr(shared_library, name)

        return super().__getattribute__(name)

    def launch(self, *args):
        # the same reason as for `load_binary`
        return self.shared_library.launch(args)

    def load_binary(self, *args):
        # if we don't use parameter passing in this way,
        # we will need to rewrite the line in the general part of the code:
        # driver.active.utils.load_binary(
        #     self.name, self.kernel, self.metadata.shared,
        #     self.metadata.build_flags, is_spv, device, max_grf_mode) ->
        # driver.active.utils.load_binary(
        #     (self.name, self.kernel, self.metadata.shared,
        #      self.metadata.build_flags, is_spv, device, max_grf_mode))
        # PTSS-overflow detection happens at the C level (driver.c
        # tryRaisePTSSOutOfResources): when zeModuleCreate fails and the
        # IGC build log carries a PTSS marker, OutOfResources is raised
        # directly so triton.runtime.autotuner can skip the offending tile.
        # No per-launch overhead on the success path.
        try:
            return self.shared_library.load_binary(args)
        except Exception as e:
            from .errors import IntelGPUError
            from triton.runtime.errors import OutOfResources
            if isinstance(e, OutOfResources):
                raise
            if str(e).startswith("ZE_"):
                raise IntelGPUError("Error during Intel load_binary: " + str(e)) from e
            else:
                raise e

    # Deliberately no `__del__`: `spirv_utils` must never be unloaded.
    #
    # `driver.c` defines the statically allocated `PyKernelArgType`, and instances of it
    # are cached for the lifetime of every compiled kernel (`XPULauncher.arg_annotations`).
    # Unloading the module frees the storage of that type object while those instances are
    # still reachable, so the next cyclic-GC pass dereferences a dangling `ob_type` and the
    # interpreter dies with `Fatal Python error: Segmentation fault`
    # (`tupletraverse` -> `visit_decref` -> `_PyObject_IS_GC` -> `Py_TYPE`).
    # See https://github.com/intel/intel-xpu-backend-for-triton/issues/7682 and
    # https://github.com/python/cpython/issues/114538.
    #
    # The type is not the only thing here that outlives an unload, so do not "fix" this by
    # moving `PyKernelArgType` into Python and restoring the unload: `load_binary` returns
    # capsules whose destructors (`freeKernel`, `freeKernelBundle`) and whose name strings
    # live in this module, and every `XPULauncher` holds callables from it. Unloading is safe
    # only once nothing reachable from Python can reference this module's code or storage.
    #
    # Windows is why this unload existed (#3090: a mapped `.pyd` cannot be deleted), so do not
    # restore it for that reason either. Nothing here needs it any more: `runtime/cache.py`
    # tolerates the `PermissionError` from `os.replace` on `nt`, and the `fresh_triton_cache`
    # fixtures defer cache deletion past process exit (#7312).
    #
    # `ArchParser` and `ExtensionUtils` keep their unload because they are out of scope here,
    # not because they are proven safe. `ArchParser.__getattribute__` returns a closure over
    # the raw ctypes function that does not keep its owner alive, so a caller outliving the
    # parser would call into an unloaded library.


class ExtensionUtils:
    """Lightweight utility for checking device extensions without full driver initialization."""

    def __init__(self, cache_path: str):
        self.shared_library = ctypes.PyDLL(cache_path)
        self.shared_library.check_extension.restype = ctypes.py_object
        self.shared_library.check_extension.argtypes = (ctypes.c_int, ctypes.c_char_p)
        self.shared_library.get_device_id.restype = ctypes.py_object
        self.shared_library.get_device_id.argtypes = (ctypes.c_int, )

    def check_extension(self, device_id: int, extension: bytes) -> bool:
        return self.shared_library.check_extension(device_id, extension)

    def get_device_id(self, device_idx: int) -> int:
        return self.shared_library.get_device_id(device_idx)

    if os.name != 'nt':

        def __del__(self):
            if hasattr(self, "shared_library"):
                handle = self.shared_library._handle
                self.shared_library.dlclose.argtypes = (ctypes.c_void_p, )
                self.shared_library.dlclose(handle)
    else:

        def __del__(self):
            if hasattr(self, "shared_library"):
                handle = self.shared_library._handle
                ctypes.windll.kernel32.FreeLibrary.argtypes = (ctypes.c_uint64, )
                ctypes.windll.kernel32.FreeLibrary(handle)


class SpirvProbeUtils:
    """Ground-truth probe for whether a SPIR-V capability is accepted by the driver's own JIT."""

    def __init__(self, cache_path: str):
        self.shared_library = ctypes.PyDLL(cache_path)
        self.shared_library.probe_spirv_module.restype = ctypes.py_object
        self.shared_library.probe_spirv_module.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t)

    def probe_spirv_module(self, device_id: int, spirv: bytes) -> bool | None:
        return self.shared_library.probe_spirv_module(device_id, spirv, len(spirv))

    if os.name != 'nt':

        def __del__(self):
            if hasattr(self, "shared_library"):
                handle = self.shared_library._handle
                self.shared_library.dlclose.argtypes = (ctypes.c_void_p, )
                self.shared_library.dlclose(handle)
    else:

        def __del__(self):
            if hasattr(self, "shared_library"):
                handle = self.shared_library._handle
                ctypes.windll.kernel32.FreeLibrary.argtypes = (ctypes.c_uint64, )
                ctypes.windll.kernel32.FreeLibrary(handle)


_VERSION_PATTERN = re.compile(r'(\d+)\.(\d+)\.(\d+)(?:\+(\d+))?')


def is_lts(ver) -> bool:
    if not ver:
        return True
    m = _VERSION_PATTERN.match(ver)
    if not m:
        return True
    return tuple(int(x) if x is not None else 0 for x in m.groups()) < (1, 6, 35096, 9)


def has_spirv_2d_block_io(ver: str | None) -> bool:
    """
    Whether the SPIR-V 2D block I/O builtins can be used with this driver (#8858).

    IGC before 2.38 (compute-runtime 26.27.39122) only has a fixed table of `intel_sub_group_2d_block_*` builtins, so
    other shapes fail to build; 2.38 added a generic resolver. `ver` is the driver version
    (`1.<minor>.<build>[+<patch>]`) or the `ocloc` version (`26.18.38308.1`); either way the compute-runtime build
    number is the third field. A missing or unparsable version keeps the SPIR-V default.
    """
    if not ver:
        return True
    m = _VERSION_PATTERN.match(ver)
    if not m:
        return True
    return int(m.group(3)) >= 39122


@lru_cache
def get_ocloc_version() -> str | None:
    """The version of the `ocloc` on PATH, e.g. `26.18.38308.1`, or None if it can't be determined."""
    try:
        proc = subprocess.run(['ocloc', '--version'], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r'\d+\.\d+\.\d+(?:\.\d+)?', proc.stdout)
    return m.group(0) if m else None


@lru_cache
def get_hasher_common(eventless_submit: bool = True):
    hasher = hashlib.sha256((__CACHE_VERSION + platform_key()).encode("utf-8"))
    # Include libsycl_dir in the hash to prevent cache collisions across
    # environments with different oneAPI versions (e.g. 2025.3 vs 2026.0).
    # The compiled .so has libsycl_dir baked in as RPATH; without this,
    # two envs with identical extension_utils.c but different oneAPI stacks
    # share the same cache entry and load an incompatible .so.
    if COMPILATION_HELPER.libsycl_dir:
        hasher.update(str(COMPILATION_HELPER.libsycl_dir).encode("utf-8"))
        # The directory stays when the runtime in it is upgraded in place (`pip install -U`), but the
        # helper is linked against one soname, so it must be rebuilt then.
        for sycl_dir in COMPILATION_HELPER.libsycl_dir:
            hasher.update(str(_soname(os.path.join(sycl_dir, "libsycl.so"))).encode("utf-8"))
        # Releases of one SYCL major share a soname, and a runtime only runs code built against its
        # own headers or older ones. So a helper must also be rebuilt when the headers change under
        # the same directory and soname, e.g. when `intel-sycl-rt` is replaced by an older release.
        hasher.update(str(_sycl_header_version(COMPILATION_HELPER.include_dir)).encode("utf-8"))
    if eventless_submit:
        hasher.update("eventless_submit=True".encode("utf-8"))
    return hasher


def compile_module_from_src(src: str, name: str, eventless_submit: bool = True):
    hasher = get_hasher_common(eventless_submit).copy()
    hasher.update(src.encode("utf-8"))
    # `spirv_utils` is compiled with `-DTRITON_INTEL_INJECT_PYTORCH=1` when
    # `INJECT_PYTORCH=True` (see below), which wraps kernel launches in a
    # `RECORD_FUNCTION` scope. That flag is not part of `src`, so without it in the key
    # an uninstrumented .so cached by an earlier run (e.g. the unit tests, which never
    # set `INJECT_PYTORCH`) is reused and the profiler scope silently disappears.
    if name == "spirv_utils" and COMPILATION_HELPER.inject_pytorch_dep:
        hasher.update("inject_pytorch=True".encode("utf-8"))
    key = hasher.hexdigest()
    cache = get_cache_manager(key)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    cache_path = cache.get_file(f"{name}{suffix}")
    if cache_path is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.cpp")
            with open(src_path, "w") as f:
                f.write(src)
            extra_compiler_args = []
            if COMPILATION_HELPER.libsycl_dir:
                if os.name == "nt":
                    extra_compiler_args += ["/LIBPATH:" + dir for dir in COMPILATION_HELPER.libsycl_dir]
                else:
                    extra_compiler_args += ["-Wl,-rpath," + dir for dir in COMPILATION_HELPER.libsycl_dir]

            if COMPILATION_HELPER.inject_pytorch_dep and name == "spirv_utils":
                if os.name == "nt":
                    extra_compiler_args += ["/DTRITON_INTEL_INJECT_PYTORCH=1", "/std:c++20"]
                else:
                    extra_compiler_args += ["-DTRITON_INTEL_INJECT_PYTORCH=1", "-std=c++20"]

            if name == "spirv_utils" and eventless_submit:
                if os.name == "nt":
                    extra_compiler_args += ["/DENABLE_EXPERIMENTAL_EVENTLESS_SUBMIT"]
                else:
                    extra_compiler_args += ["-DENABLE_EXPERIMENTAL_EVENTLESS_SUBMIT"]

            so = _build(name, src_path, tmpdir, COMPILATION_HELPER.library_dir, COMPILATION_HELPER.include_dir,
                        COMPILATION_HELPER.libraries, ccflags=extra_compiler_args,
                        use_sycl_compiler=COMPILATION_HELPER.use_sycl_compiler)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}{suffix}", binary=True)

    if name == 'arch_utils':
        return ArchParser(cache_path)
    if name == 'spirv_utils':
        return SpirvUtils(cache_path)
    if name == 'extension_utils_impl':
        return ExtensionUtils(cache_path)
    if name == 'spirv_probes_impl':
        return SpirvProbeUtils(cache_path)
    if name == 'proton_utils':
        return cache_path

    return _load_module_from_path(name, cache_path)


# ------------------------
# Utils
# ------------------------

PEAK_DRAM_GBPS = {
    "Intel(R) Data Center GPU Max 1100": 1228.8,
    "Intel(R) Data Center GPU Max 1550": 3276.8,
    "Intel(R) Arc(TM) B580 Graphics": 456.0,
    "Intel(R) Arc(TM) B570 Graphics": 380.0,
    "Intel(R) Arc(TM) B560 Graphics": 456.0,
}


class XPUUtils(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(XPUUtils, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        dirname = os.path.dirname(os.path.realpath(__file__))
        # we save `spirv_utils` module so that the destructor is not called prematurely, which will unload the dll
        # and can cause `Fatal Python error: Segmentation fault`
        mod = compile_module_from_src(src=Path(os.path.join(dirname, "driver.c")).read_text(), name="spirv_utils",
                                      eventless_submit=self._use_eventless_submit())
        global PyKernelArg
        global ARG_CONSTEXPR
        global ARG_KERNEL
        global ARG_TUPLE
        PyKernelArg = mod.init_PyKernelArgType()  # mod.PyKernelArg
        ARG_CONSTEXPR = 0
        ARG_KERNEL = 1
        ARG_TUPLE = 2
        # ARG_CONSTEXPR = mod.ARG_CONSTEXPR
        # ARG_KERNEL = mod.ARG_KERNEL
        # ARG_TUPLE = mod.ARG_TUPLE
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.device_count = mod.init_devices(self.get_sycl_queue())
        self.wait_on_sycl_queue = mod.wait_on_sycl_queue
        self.get_last_selected_build_flags = mod.get_last_selected_build_flags
        self.sycl_queue_memset = mod.sycl_queue_memset
        self.unload_module = lambda module: None
        self.launch = mod.launch
        self.build_signature_metadata = mod.build_signature_metadata
        self._initialized = True

    def get_sycl_queue(self):
        import torch
        return torch.xpu.current_stream().sycl_queue

    def _use_eventless_submit(self):
        import torch
        properties = torch.xpu.get_device_capability(torch.xpu.current_device())
        # Event-less submission only saves time on the Level Zero v2 adapter, whose platform name ends in "V2". On v1
        # (PVC's default) it does not, and PTI's default collection mode then reports zero kernel times or hangs.
        return not is_lts(properties.get('driver_version')) and properties.get('platform_name', '').endswith(' V2')

    def wait(self):
        self.wait_on_sycl_queue(self.get_sycl_queue())

    def memset(self, ptr, value, count):
        """Wrapper for SYCL queue memset"""
        return self.sycl_queue_memset((self.get_sycl_queue(), ptr, value, count))

    def get_bandwidth(self, device):
        import torch
        return PEAK_DRAM_GBPS.get(torch.xpu.get_device_name(device))


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int8_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint8_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        # Pass 16-bit floating-point scalars as raw bit patterns.
        "fp16": "uint16_t",
        "bf16": "uint16_t",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def annotate_arguments(signature):
    """
    This recreates the signature with annotations as C objects which can then
    be used to efficiently flatten tuples, and remove constexpr in the launcher.
    """
    annotated_arguments = []
    for sig in signature:
        if isinstance(sig, tuple):
            annotated_arguments.append((PyKernelArg(nested_tuple=annotate_arguments(sig), type=ARG_TUPLE)))
        elif sig != "constexpr":
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_KERNEL))
        else:
            annotated_arguments.append(PyKernelArg(nested_tuple=None, type=ARG_CONSTEXPR))
    return annotated_arguments


def _make_intel_tensordesc_arg(arg, _meta, _base_args):
    # Intel does not use TMA descriptors, so _meta (tensordesc metadata) and
    # _base_args (launcher base arguments) are unused.  We simply decompose the
    # TensorDescriptor into base pointer, shape, strides, and flags.
    return decompose_descriptor(arg)


def wrap_handle_tensordesc(launcher, signature, tensordesc_meta):
    return wrap_handle_tensordesc_impl(launcher, signature, tensordesc_meta, _make_intel_tensordesc_arg)


def serialize_args(args, constants, signature, dir_path):
    import torch
    import numbers
    os.makedirs(dir_path, exist_ok=True)

    def serialize_kernel_metadata(arg, args_dict):
        args_dict['num_warps'] = arg.num_warps
        args_dict['threads_per_warp'] = arg.threads_per_warp
        args_dict['shared_memory'] = arg.shared
        args_dict['kernel_name'] = arg.name
        is_native_binary = bool(getattr(arg, "generate_native_code", False))
        binary_ext = "zebin" if is_native_binary else "spv"
        args_dict['binary_name'] = f"{arg.name}.{binary_ext}"
        args_dict['is_native_binary'] = is_native_binary
        args_dict['build_flags'] = arg.build_flags

    cnt = 0
    args_dict = {"gridX": int(args[cnt]), "gridY": int(args[cnt + 1]), "gridZ": int(args[cnt + 2])}
    # 3: stream
    # 4: function
    # 5: packed kernel metadata
    assert type(args[cnt + 5]).__name__ == "KernelMetadata"
    serialize_kernel_metadata(args[cnt + 5], args_dict)
    # 6: launch_metadata
    # 7: launch_enter_hook
    # 8: launch_exit_hook
    args_dict['argument_list'] = []
    counts = {"tensors": 0, "scalars": 0, "karg_cnt": 0}
    cnt += 9
    for arg in args[cnt:]:
        sig_name = list(signature.keys())[counts['karg_cnt']]
        if isinstance(arg, torch.Tensor):
            cpu_tensor = arg.cpu()
            tensor_path = os.path.join(dir_path, f"tensor_{counts['tensors']}.pt")
            with open(tensor_path, 'wb') as f:
                torch.save(cpu_tensor, f)
            new_arg = {
                "name": f"tensor_{counts['tensors']}", "type": "tensor", "dtype": str(arg.dtype), "ctype":
                signature[sig_name]
            }
            args_dict['argument_list'].append(new_arg)
            counts['tensors'] += 1
        if isinstance(arg, numbers.Number):
            if (counts['karg_cnt'], ) not in constants.keys():
                new_arg = {
                    "name": f"scalarArg_{counts['scalars']}", "type": "scalar", "value": arg, "ctype":
                    signature[sig_name]
                }
                args_dict['argument_list'].append(new_arg)
            counts['scalars'] += 1
        counts['karg_cnt'] += 1

    # Dump argument info as a JSON file
    json_path = os.path.join(dir_path, 'args_data.json')
    with open(json_path, 'w') as json_file:
        import json
        json.dump(args_dict, json_file, indent=4)


class XPULauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        utils = triton.runtime.driver.active.utils
        expanded_signature = expand_signature(signature.values(), TensorDescABI.DECOMPOSED)
        self.arg_annotations = annotate_arguments(expanded_signature)
        self.kernel_signature = utils.build_signature_metadata(
            ([ty for _, ty in get_kernel_argument_layout(signature.values(), TensorDescABI.DECOMPOSED)], ))
        self.launch = wrap_handle_tensordesc(utils.launch, signature, tensordesc_meta=[])

        # Serialize KernelArguments for TritonKernelRunner
        self.serialize_kernel_args = knobs.intel.enable_dump_kernel_args
        self.cache_dir = metadata.cache_dir
        self.dump_dir = self._resolve_dump_dir(metadata.cache_dir)
        self.print_dump_kernel_args_info = knobs.intel.print_dump_kernel_args_info
        self.constants = constants
        self.signature = signature
        self.dump_launch_params = os.environ.get("TRITON_DUMP_LAUNCH_PARAMS") == "1"
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align

    def _resolve_dump_dir(self, cache_dir):
        dump_dir_root = knobs.intel.dump_kernel_args_dir
        if not dump_dir_root:
            return cache_dir
        cache_dir_name = os.path.basename(os.path.normpath(cache_dir))
        return os.path.join(dump_dir_root, cache_dir_name)

    def _dump_launch_params(self, args, constants, signature):
        # inspired by `def _dump_launch_params(args, kwargs, launcher, kernel_name, grid):` from
        # torch/_inductor/runtime/triton_heuristics.py
        grid = args[:3]
        new_args = args[9:]
        call_args = []
        call_kwargs = {}
        for arg in new_args:
            if hasattr(arg, "shape"):
                call_args.append(f"T{list(arg.shape)}")
            else:
                call_args.append(str(arg))

        # handle kwargs
        signature = list(signature.keys())
        for idx, value in constants.items():
            # In general this is not the case, but it is sufficient for llama 3.1 kernels
            assert len(idx) == 1
            call_kwargs[signature[idx[0]]] = value
        call_kwargs["num_warps"] = args[5].num_warps
        call_kwargs["num_stages"] = args[5].num_stages

        # adjust args
        constants = [(idx[0], value) for idx, value in constants.items()]
        constants = sorted(constants, reverse=True)
        for idx, _ in constants:
            # it have been added as kwargs
            call_args.pop(idx)

        args_str = [*call_args]
        args_str.extend(f"{k}={v}" for k, v in call_kwargs.items())
        args_str = ", ".join(args_str)
        abs_path = os.path.abspath(sys.argv[0])
        file_path = f"{abs_path}.launch_params_triton_{os.getpid()}"
        print(f"file with launch params: {file_path}")
        with open(file_path, "a") as f:
            entry = {args[5].name: {"launch_args": args_str, "grid": list(grid)}}
            f.write(json.dumps(entry) + "\n")

    def __call__(self, gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                 launch_exit_hook, *args):
        active_driver = triton.runtime.driver.active

        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_fn = allocator.get()
                return alloc_fn(grid_size * size, align, stream)
            return None

        def allocate_default_profile_scratch(size, align):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                return active_driver.allocate_default_profile_scratch(grid_size * size, align, stream)
            return None

        global_scratch = allocate_scratch(self.global_scratch_size, self.global_scratch_align, _allocation._allocator)
        if _allocation.has_profile_allocator():
            profile_scratch = allocate_scratch(self.profile_scratch_size, self.profile_scratch_align,
                                               _allocation._profile_allocator)
        else:
            profile_scratch = allocate_default_profile_scratch(self.profile_scratch_size, self.profile_scratch_align)

        if self.serialize_kernel_args:
            if self.print_dump_kernel_args_info:
                print(
                    f"Triton kernel dump info: kernel_name={kernel_metadata.name}, cache_dir={self.cache_dir}, dump_dir={self.dump_dir}"
                )
            serialize_args((gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                            launch_exit_hook, *args), self.constants, self.signature, self.dump_dir)

        if self.dump_launch_params:
            # This function does not cover all cases, for example when the arguments are tuple,
            # but it is sufficient for llama 3.1 kernels
            self._dump_launch_params((gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata,
                                      launch_enter_hook, launch_exit_hook, *args), self.constants, self.signature)

        self.launch(gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                    launch_exit_hook, global_scratch, profile_scratch, self.arg_annotations, self.kernel_signature,
                    args)


class XPUDriver(DriverBase):

    def __init__(self):
        self.launcher_cls = XPULauncher
        try:
            from torch._C import _xpu_getCurrentRawStream, _xpu_getDevice
            self.get_current_device = _xpu_getDevice
            self.get_current_stream = _xpu_getCurrentRawStream
        except ImportError:
            import torch
            self.get_current_device = torch.xpu.current_device
            self.get_current_stream = lambda idx: torch.xpu.current_stream(idx).sycl_queue
        super().__init__()

    def __getattr__(self, name):
        # Lazily initialize utils to avoid unnecessary XPU runtime invocations.
        # See https://github.com/intel/intel-xpu-backend-for-triton/issues/624
        if name == "utils":
            self.utils = XPUUtils()
            return self.utils
        else:
            raise AttributeError

    @lru_cache
    def _construct_target(self, device):
        import torch
        from .extension_utils import query_device_extensions

        dev_property = torch.xpu.get_device_capability(device)

        def update_device_arch(dev_property):
            if not (arch := knobs.intel.device_arch):
                dirname = os.path.dirname(os.path.realpath(__file__))
                parser = compile_module_from_src(src=Path(os.path.join(dirname, "arch_parser.c")).read_text(),
                                                 name="arch_utils")
                arch = parser.parse_device_arch(dev_property["architecture"])
            dev_property["arch"] = arch

        # All GPUs with the same device_id have the same extensions, so we just
        # need to query any GPU device
        device_id = dev_property.get("device_id")
        extensions = query_device_extensions(device_id)
        dev_property.update(extensions)
        dev_property["__intel_already_queried_extensions__"] = True
        dev_property["core_clock_rate"] = self.utils.get_device_properties(device).get("sm_clock_rate", 0)
        update_device_arch(dev_property)

        return GPUTarget("xpu", dev_property, warp_size=32)

    def get_current_target(self):
        device = self.get_current_device()
        return self._construct_target(device)

    def build_proton_help_lib(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        return compile_module_from_src(src=Path(dirname).joinpath("proton_utils.cpp").read_text(), name="proton_utils")

    def get_active_torch_device(self):
        import torch
        return torch.device("xpu", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.xpu

    def allocate_default_profile_scratch(self, size: int, alignment: int, stream):
        import torch
        device = self.get_active_torch_device()
        if not stream:
            with torch.xpu.stream(torch.xpu.current_stream(device)):
                return torch.zeros(size, dtype=torch.int8, device=device)
        launch_stream = torch.xpu.get_stream_from_external(stream, device=device)
        with torch.xpu.stream(launch_stream):
            scratch = torch.zeros(size, dtype=torch.int8, device=device)
        scratch.record_stream(launch_stream)
        return scratch

    @staticmethod
    def is_active():
        if native.error:  # the backend's C++ module could not be loaded (native.py)
            return False
        try:
            import torch
            return torch.xpu.is_available()
        except ImportError:
            return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='xpu')

    def clear_cache(self, cache):
        cache.zero_()
