"""
Which SYCL runtime a library links, read from the library file itself: the ELF and PE readers, and
the lookup of the runtime PyTorch's XPU libraries link. `driver.find_sycl` uses them to build the
helpers against PyTorch's runtime. See https://github.com/intel/intel-xpu-backend-for-triton/issues/8200.
"""

import importlib.metadata
import importlib.util
import mmap
import os
import re
import struct
import warnings


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
    # They are in the package's `torch/lib`, for a wheel and for an editable install or a source tree
    # on `PYTHONPATH` alike. Finding a top-level package does not import it, and it finds the one
    # `import torch` would load.
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        spec = None
    libraries = []
    for location in (spec.submodule_search_locations or []) if spec else []:
        for name in _TORCH_XPU_LIBRARIES:
            library = os.path.join(location, "lib", name)
            if os.path.isfile(library):
                libraries.append(library)
    if libraries:
        return libraries
    # Otherwise ask the package metadata where they are. That parses every file the wheel installed,
    # about 15,000 for PyTorch, so it is only the fallback.
    try:
        files = importlib.metadata.files("torch")
    except importlib.metadata.PackageNotFoundError:
        files = None
    return [str(f.locate()) for f in files or [] if f.name in _TORCH_XPU_LIBRARIES]


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
