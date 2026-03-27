from __future__ import annotations

import atexit
import functools
import hashlib
import importlib.util
import locale
import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

from types import ModuleType

from .cache import get_cache_manager
from .. import knobs


class _PerfLogger:
    """Centralized performance logger gated by TRITON_XPU_PERF_LOG env var.

    Levels:
      0 / unset: disabled
      1: only events slower than threshold (100ms)
      2: all events
      summary: print per-category summary at process exit
    """

    def __init__(self):
        val = os.environ.get("TRITON_XPU_PERF_LOG", "0").strip().lower()
        self.level = 0  # 0=off, 1=slow-only, 2=all, 3=summary
        self.summary_mode = False
        if val in ("1",):
            self.level = 1
        elif val in ("2",):
            self.level = 2
        elif val == "summary":
            self.level = 2  # also log all events for accumulation
            self.summary_mode = True
        # running totals for summary mode
        self._totals: dict[str, list[float]] = {}
        # wall-clock tracking
        self._create_time = time.perf_counter()
        # Get actual process start time from /proc for Linux
        self._process_start: float | None = None
        try:
            import struct
            clock_ticks = os.sysconf("SC_CLK_TCK")
            with open("/proc/self/stat", "r") as f:
                fields = f.read().split(")")[-1].split()
                # field index 19 (0-based from after comm) = starttime in clock ticks
                starttime_ticks = int(fields[19])
            with open("/proc/stat", "r") as f:
                for line in f:
                    if line.startswith("btime "):
                        btime = int(line.split()[1])
                        break
            process_start_epoch = btime + starttime_ticks / clock_ticks
            # Convert to perf_counter scale: current epoch - current perf_counter = epoch_offset
            import time as _time_mod
            epoch_offset = _time_mod.time() - _time_mod.perf_counter()
            self._process_start = process_start_epoch - epoch_offset
        except Exception:
            self._process_start = None
        self._first_log_time: float | None = None
        self._last_log_time: float = 0.0
        # import-time milestones (filled in by other modules via record_milestone)
        self._milestones: list[tuple[str, float]] = []
        self._milestones.append(("triton.runtime.build imported", self._create_time))
        if self.summary_mode:
            atexit.register(self.print_summary)

    @property
    def enabled(self):
        return self.level > 0

    def record_milestone(self, name: str) -> None:
        """Record a wall-clock milestone for the summary timeline."""
        if self.summary_mode:
            self._milestones.append((name, time.perf_counter()))

    def log(self, category: str, msg: str, elapsed_s: float, threshold_s: float = 0.1):
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._first_log_time is None:
            self._first_log_time = now - elapsed_s  # approximate start of first timed call
        self._last_log_time = now
        if self.summary_mode:
            self._totals.setdefault(category, []).append(elapsed_s)
        if self.level >= 2 or (self.level == 1 and elapsed_s >= threshold_s):
            import sys as _sys
            print(f"[TRITON_PERF] {category}: {msg}  ({elapsed_s*1000:.1f}ms)", file=_sys.stderr, flush=True)

    def print_summary(self):
        import sys as _sys

        # Tree structure: each node has (name, [children_names])
        # Children times should add up to parent. Any gap is shown as "other".
        _TREE = {
            "JITFunction.run": ["create_binder", "compile", "_init_handles", "kernel_launch"],
            "create_binder": ["binder.import", "binder.make_backend", "binder.create_fn_sig"],
            "compile": [
                "triton_key", "compile.make_ir",
                "compile.ttir", "compile.ttgir", "compile.llir",
                "compile.ptx", "compile.cubin",
                "compile.spv", "compile.zebin",
            ],
            "compile.llir": ["stage.llir_mlir_passes", "stage.llir_to_llvm", "stage.llir_llvm_opt"],
            "compile.cubin": ["stage.cubin_ptxas"],
            "_init_handles": ["launcher_cls", "load_binary"],
            "launcher_cls": ["launcher.generic_setup", "launcher.codegen", "launcher.c_compile"],
        }
        # Display order (flat list for iteration)
        _DISPLAY_ORDER = [
            "JITFunction.run",
            "create_binder",
            "binder.import", "binder.make_backend", "binder.create_fn_sig",
            "compile",
            "triton_key", "compile.make_ir",
            "compile.ttir", "compile.ttgir",
            "compile.llir",
            "stage.llir_mlir_passes", "stage.llir_to_llvm", "stage.llir_llvm_opt",
            "compile.ptx",
            "compile.cubin",
            "stage.cubin_ptxas",
            "compile.spv", "compile.zebin",
            "_init_handles",
            "launcher_cls",
            "launcher.generic_setup", "launcher.codegen", "launcher.c_compile",
            "load_binary",
            "kernel_launch",
        ]

        def _get_total(name):
            times = self._totals.get(name)
            if times is None:
                return 0.0
            return sum(times)

        def _get_count(name):
            times = self._totals.get(name)
            if times is None:
                return 0
            return len(times)

        # Compute indent from tree structure
        def _get_indent(name, tree, depth=0):
            if depth == 0 and name in ("JITFunction.run",):
                return 0
            for parent, children in tree.items():
                if name in children:
                    return _get_indent(parent, tree, depth + 1) + 1
            return 0

        # Find parent of a node
        def _get_parent(name):
            for parent, children in _TREE.items():
                if name in children:
                    return parent
            return None

        print("\n[TRITON_PERF] === Process Summary ===", file=_sys.stderr)
        printed = set()
        for cat in _DISPLAY_ORDER:
            total_s = _get_total(cat)
            n = _get_count(cat)
            if n == 0:
                continue
            printed.add(cat)
            avg = total_s / n if n else 0
            indent = _get_indent(cat, _TREE)
            prefix = "  " * indent
            parent = _get_parent(cat)
            parent_total = _get_total(parent) if parent else 0

            # Format: name : total, calls, avg, %parent
            label = f"{prefix}{cat}"
            pct_str = ""
            if parent_total > 0:
                pct = (total_s / parent_total) * 100
                pct_str = f"  ({pct:5.1f}%)"
            print(f"[TRITON_PERF] {label:42s}: {total_s*1000:10.1f}ms  {n:5d} calls  avg {avg*1000:8.1f}ms{pct_str}",
                  file=_sys.stderr)

            # After printing all children of a node, print "other" if there's a gap
            children = _TREE.get(cat)
            if children:
                children_total = sum(_get_total(c) for c in children)
                gap = total_s - children_total
                if abs(gap) > 0.001:  # > 1ms
                    child_indent = "  " * (indent + 1)
                    gap_label = f"{child_indent}(other/overhead)"
                    pct = (gap / total_s) * 100 if total_s > 0 else 0
                    print(f"[TRITON_PERF] {gap_label:42s}: {gap*1000:10.1f}ms                          ({pct:5.1f}%)",
                          file=_sys.stderr)

        # Print any categories not in the display order (future-proofing)
        known = set(_DISPLAY_ORDER)
        for cat in sorted(self._totals.keys()):
            if cat in known:
                continue
            times = self._totals[cat]
            total_s = sum(times)
            n = len(times)
            avg = total_s / n if n else 0
            print(f"[TRITON_PERF] {'??? ' + cat:42s}: {total_s*1000:10.1f}ms  {n:5d} calls  avg {avg*1000:8.1f}ms",
                  file=_sys.stderr)

        # Wall-clock accounting
        summary_time = time.perf_counter()
        wall_since_import = summary_time - self._create_time
        instrumented = sum(self._totals.get("JITFunction.run", []))
        before_triton = (self._first_log_time - self._create_time) if self._first_log_time is not None else 0.0
        after_triton = summary_time - self._last_log_time if self._last_log_time else 0.0
        between_calls = wall_since_import - before_triton - instrumented - after_triton
        print(f"[TRITON_PERF] ---", file=_sys.stderr)
        if self._process_start is not None:
            proc_to_import = self._create_time - self._process_start
            total_wall = summary_time - self._process_start
            print(f"[TRITON_PERF] Process total wall time                  : {total_wall*1000:10.1f}ms", file=_sys.stderr)
            print(f"[TRITON_PERF]   python start → triton import           : {proc_to_import*1000:10.1f}ms", file=_sys.stderr)
            print(f"[TRITON_PERF]   triton import → first Triton API call  : {before_triton*1000:10.1f}ms", file=_sys.stderr)
        else:
            print(f"[TRITON_PERF] Wall time since triton import             : {wall_since_import*1000:10.1f}ms", file=_sys.stderr)
            print(f"[TRITON_PERF]   before first Triton API call            : {before_triton*1000:10.1f}ms", file=_sys.stderr)
        print(f"[TRITON_PERF]   Triton JITFunction.run (top-level)      : {instrumented*1000:10.1f}ms", file=_sys.stderr)
        print(f"[TRITON_PERF]   between Triton calls (host/test work)   : {between_calls*1000:10.1f}ms", file=_sys.stderr)
        print(f"[TRITON_PERF]   after last Triton API call              : {after_triton*1000:10.1f}ms", file=_sys.stderr)
        # Print milestone timeline relative to process start
        if self._milestones and self._process_start is not None:
            print(f"[TRITON_PERF] --- Timeline (from process start) ---", file=_sys.stderr)
            for mname, mtime in sorted(self._milestones, key=lambda x: x[1]):
                offset = (mtime - self._process_start) * 1000
                print(f"[TRITON_PERF]   {offset:10.1f}ms  {mname}", file=_sys.stderr)
            if self._first_log_time is not None:
                print(f"[TRITON_PERF]   {(self._first_log_time - self._process_start)*1000:10.1f}ms  first JITFunction.run() started", file=_sys.stderr)
            print(f"[TRITON_PERF]   {(self._last_log_time - self._process_start)*1000:10.1f}ms  last JITFunction.run() ended", file=_sys.stderr)
        print("[TRITON_PERF] === End Summary ===", file=_sys.stderr, flush=True)


perf_log = _PerfLogger()

_IS_WINDOWS = sys.platform == "win32"
SUBPROCESS_DECODE_ARGS = (locale.getpreferredencoding(), ) if _IS_WINDOWS else ()


def is_xpu():
    import torch
    return torch.xpu.is_available()


def _cc_cmd(cc, src, out, include_dirs, library_dirs, libraries):
    if "cl.EXE" in cc or "clang-cl" in cc or "icx-cl" in cc:
        cc_cmd = [cc, "/Zc:__cplusplus", "/std:c++17", src, "/nologo", "/O2", "/LD", "/wd4996", "/MD", "/EHsc"]
        cc_cmd += [f"/I{dir}" for dir in include_dirs]
        cc_cmd += [f"/Fo{os.path.join(os.path.dirname(out), 'main.obj')}"]
        cc_cmd += ["/link"]
        cc_cmd += [f"/OUT:{out}"]
        cc_cmd += [f"/IMPLIB:{os.path.join(os.path.dirname(out), 'main.lib')}"]
        cc_cmd += [f"/PDB:{os.path.join(os.path.dirname(out), 'main.pdb')}"]
        cc_cmd += [f"/LIBPATH:{dir}" for dir in library_dirs]
        cc_cmd += [f'{lib}.lib' for lib in libraries]
    else:
        cc_cmd = [cc, src, "-O3", "-shared", "-Wno-psabi"]
        if os.name != "nt":
            cc_cmd += ["-fPIC"]
        else:
            cc_cmd += ["-Wno-deprecated-declarations"]
        cc_cmd += [f'-l:{lib}' if '.so' in lib else f'-l{lib}' for lib in libraries]
        cc_cmd += [f"-L{dir}" for dir in library_dirs]
        cc_cmd += [f"-I{dir}" for dir in include_dirs]
        cc_cmd += ["-o", out]

    return cc_cmd


@functools.lru_cache()
def _find_compiler(language: str) -> str:
    if language == "c":
        cc = os.environ.get("CC")
        if cc is not None:
            return cc
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cl = shutil.which("cl") if os.name == "nt" else None
        cc = cl or gcc or clang
        if cc is not None:
            return cc
        raise RuntimeError(
            "Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.")

    assert language == "c++"
    cxx = os.environ.get("CXX")
    if cxx is not None:
        return cxx

    clangxx = shutil.which("clang++")
    gxx = shutil.which("g++")
    cxx = gxx if gxx is not None else clangxx
    if cxx is not None:
        return cxx

    raise RuntimeError(
        "Failed to find C++ compiler. Please specify via CXX environment variable or set triton.knobs.build.impl.")


def _language_from_filename(source_name: str) -> str:
    ext = Path(source_name).suffix
    if ext == ".c":
        return "c"
    if ext in {".cc", ".cpp", ".cxx"}:
        return "c++"
    print(source_name)
    raise ValueError(f"Unrecognized file extension: {source_name}")


def _build(name: str, src: str, srcdir: str, library_dirs: list[str], include_dirs: list[str], libraries: list[str],
           ccflags: list[str], language: str = "c") -> str:
    if impl := knobs.build.impl:
        return impl(name, src, srcdir, library_dirs, include_dirs, libraries, ccflags)
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, f'{name}{suffix}')
    cc = _find_compiler(language)
    scheme = sysconfig.get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    custom_backend_dirs = knobs.build.backend_dirs
    include_dirs = include_dirs + [srcdir, py_include_dir, *custom_backend_dirs]

    if is_xpu():
        icpx = shutil.which("icpx")
        cxx = shutil.which(os.environ.get("CXX", "shutil-dummy-value"))
        if cxx is None:
            clangpp = shutil.which("clang++")
            gxx = shutil.which("g++")
            cl = shutil.which("cl")
            cxx = icpx or cl if os.name == "nt" else icpx or clangpp or gxx
            if cxx is None:
                raise RuntimeError("Failed to find C++ compiler. Please specify via CXX environment variable.")
        cc = cxx

        if cxx is icpx:
            ccflags += ["-fsycl"]
        else:
            if os.name != "nt":
                ccflags += ["--std=c++17"]
            if os.environ.get("TRITON_SUPPRESS_GCC_HOST_CODE_DEPRECATION_WARNINGS", "1") == "1":
                ccflags += ["-Wno-deprecated-declarations"]
            if os.environ.get("TRITON_SUPPRESS_SYCL_DISABLE_FSYCL_SYCLHPP_WARNING", "1") == "1":
                ccflags += ["-DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING"]
        if os.name == "nt":
            library_dirs = library_dirs + [
                os.path.abspath(os.path.join(sysconfig.get_paths(scheme=scheme)["stdlib"], "..", "libs"))
            ]
    else:
        cc_cmd = [cc]

    # for -Wno-psabi, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111047
    cc_cmd = _cc_cmd(cc, src, so, include_dirs, library_dirs, libraries)
    if language == "c++":
        cc_cmd.insert(3, "-std=c++17")
    cc_cmd += ccflags

    if os.getenv("VERBOSE"):
        print(" ".join(cc_cmd))

    try:
        subprocess.run(cc_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        output = e.stdout.decode(*SUBPROCESS_DECODE_ARGS)
        raise RuntimeError(output)
    return so


def _library_flag(lib: str) -> str:
    # Match .so files with optional version numbers (e.g., .so, .so.1, .so.513.50.1)
    if re.search(r'\.so(\.\d+)*$', lib) or lib.endswith(".a"):
        return f"-l:{lib}"
    return f"-l{lib}"


@functools.lru_cache
def platform_key() -> str:
    from platform import machine, system, architecture
    return ",".join([machine(), system(), *architecture()])


def _get_file_extension(language):
    if language == "c":
        return ".c"
    if language == "c++":
        return ".cpp"
    raise ValueError(f"Unexpected languange: {language}")


def _load_module_from_path(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load newly compiled {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_cache_manager(src: bytes, config: dict[str, list[str] | None]):
    digest = hashlib.sha256()
    digest.update(src)
    digest.update(platform_key().encode("utf-8"))
    for k, vs in config.items():
        if vs is None:
            continue
        digest.update(k.encode("utf-8"))
        for v in vs:
            digest.update(v.encode("utf-8"))
            digest.update(b":")
    key = digest.hexdigest()
    return get_cache_manager(key)


def _compile_so(src: bytes, src_path: str, name: str, library_dirs: list[str] | None, include_dirs: list[str] | None,
                libraries: list[str] | None, ccflags: list[str] | None, load_module: bool, language: str):
    config = dict(language=[language], library_dirs=library_dirs, include_dirs=include_dirs, libraries=libraries,
                  ccflags=ccflags)
    cache = _get_cache_manager(src, config=config)
    t0 = time.perf_counter() if perf_log.enabled else 0
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    cache_path = cache.get_file(f"{name}{suffix}")
    if cache_path is not None:
        if not load_module:
            return cache_path
        try:
            mod = _load_module_from_path(name, cache_path)
            if perf_log.enabled:
                perf_log.log("compile_module_from_src", f"{name} [cache HIT]", time.perf_counter() - t0)
            return mod
        except (RuntimeError, ImportError):
            log = logging.getLogger(__name__)
            log.warning(f"Triton cache error: compiled module {name}.so could not be loaded")

    with tempfile.TemporaryDirectory() as tmpdir:
        so = _build(name, src_path, tmpdir, library_dirs or [], include_dirs or [], libraries or [], ccflags or [],
                    language=language)
        with open(so, "rb") as f:
            cache_path = cache.put(f.read(), f"{name}{suffix}", binary=True)

    if perf_log.enabled:
        perf_log.log("compile_module_from_src", f"{name} [cache MISS, compiled]", time.perf_counter() - t0)
    return _load_module_from_path(name, cache_path) if load_module else cache_path


def _compile_so_from_file(src_path: str, name: str, library_dirs: list[str] | None, include_dirs: list[str] | None,
                          libraries: list[str] | None, ccflags: list[str] | None, load_module: bool):
    src_path = os.path.abspath(src_path)
    src_name = os.path.basename(src_path)
    with open(src_path, "rb") as f:
        src = f.read()

    language = _language_from_filename(src_name)
    return _compile_so(src=src, src_path=src_path, name=name, library_dirs=library_dirs, include_dirs=include_dirs,
                       libraries=libraries, ccflags=ccflags, language=language, load_module=load_module)


def _compile_so_from_src(src: str, name: str, library_dirs: list[str] | None, include_dirs: list[str] | None,
                         libraries: list[str] | None, ccflags: list[str] | None, language, load_module: bool):
    src_bytes = src.encode("utf-8")
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, f"{name}{_get_file_extension(language)}")
        with open(src_path, "wb") as f:
            f.write(src_bytes)
        return _compile_so(src=src_bytes, src_path=src_path, name=name, library_dirs=library_dirs,
                           include_dirs=include_dirs, libraries=libraries, ccflags=ccflags, language=language,
                           load_module=load_module)


def compile_so_from_file(src_path: str, name: str, library_dirs: list[str] | None = None,
                         include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                         ccflags: list[str] | None = None) -> str:
    return _compile_so_from_file(src_path, name, library_dirs, include_dirs, libraries, ccflags, load_module=False)


def compile_so_from_src(src: str, name: str, library_dirs: list[str] | None = None,
                        include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                        ccflags: list[str] | None = None, language="c") -> str:
    return _compile_so_from_src(src, name, library_dirs, include_dirs, libraries, ccflags, language, load_module=False)


def compile_module_from_file(src_path: str, name: str, library_dirs: list[str] | None = None,
                             include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                             ccflags: list[str] | None = None) -> ModuleType:
    return _compile_so_from_file(src_path, name, library_dirs, include_dirs, libraries, ccflags, load_module=True)


def compile_module_from_src(src: str, name: str, library_dirs: list[str] | None = None,
                            include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                            ccflags: list[str] | None = None, language="c") -> ModuleType:
    return _compile_so_from_src(src, name, library_dirs, include_dirs, libraries, ccflags, language, load_module=True)
