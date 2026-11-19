#!/usr/bin/env python3
"""
Diagnose why ``import torch`` fails on a Windows XPU wheel.

The failure looks like ``ImportError: DLL load failed while importing _C: The specified module could not be
found.`` and does not name the DLL that cannot be resolved. This script gathers, for one installed torch wheel:

* ``snapshot``  - machine facts (OS, driver, ``PATH``, VC++ redistributable, oneAPI, ``pip freeze``).
* ``walk``      - a static walk of the PE import tables, starting at ``torch/_C*.pyd``. Every import is resolved
                  against the directories the *new* loader searches (no ``PATH``) and, separately, against the
                  directories the *old* preload loop effectively searched (importer directory and ``PATH``).
                  Runs on any OS (needs ``pefile``).
* ``probe``     - Windows only. ``LoadLibraryExW`` each resolved module in dependency order and print the Win32
                  error of the first failure, with the search flags of the old and the new loader.
* ``variants``  - Windows only. ``import torch`` in a fresh interpreter under several loader variants (plain,
                  ``PATH`` prefixes (torch/lib only, Library/bin only, both), extra DLL directories, the removed
                  preload loop, system oneAPI on ``PATH``), followed by an XPU smoke test for every variant that
                  imports.
* ``run``       - all of the above for one virtual environment, written to ``<out>/<label>.json``.
* ``summarize`` - merge the ``run`` results into a Markdown table (and ``$GITHUB_STEP_SUMMARY``).

See ``.github/workflows/windows-torch-import-diagnostics.yml``.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

# Search flags of ``LoadLibraryExW``.
LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000
FLAGS_NEW = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
FLAGS_OLD = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR

# Windows system DLLs that are never part of a wheel or a pip runtime package.
SYSTEM_DLLS = {
    "kernel32.dll", "kernelbase.dll", "ntdll.dll", "user32.dll", "gdi32.dll", "advapi32.dll", "shell32.dll",
    "shlwapi.dll", "ole32.dll", "oleaut32.dll", "ws2_32.dll", "bcrypt.dll", "crypt32.dll", "dbghelp.dll",
    "psapi.dll", "version.dll", "iphlpapi.dll", "rpcrt4.dll", "setupapi.dll", "cfgmgr32.dll", "winmm.dll",
    "userenv.dll", "secur32.dll", "ncrypt.dll", "imm32.dll", "comdlg32.dll", "opengl32.dll", "dxgi.dll",
    "d3d11.dll", "d3d12.dll", "shcore.dll", "wintrust.dll", "mswsock.dll", "normaliz.dll", "powrprof.dll",
    "pdh.dll", "netapi32.dll", "dwmapi.dll", "uxtheme.dll", "ucrtbase.dll", "msvcrt.dll", "imagehlp.dll",
}  # yapf: disable
# Shipped by the Visual C++ redistributable, not by Windows itself.
VC_REDIST_PREFIXES = ("msvcp140", "vcruntime140", "concrt140", "vcomp140", "vccorlib140")
PROBE_TIMEOUT = 600


def is_api_set(name: str) -> bool:
    return name.startswith(("api-ms-", "ext-ms-"))


def is_vc_redist(name: str) -> bool:
    return name.startswith(VC_REDIST_PREFIXES)


# ----------------------------------------------------------------------------------------------------------------
# Static import walk.
# ----------------------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def list_dir(directory: Path) -> dict[str, Path]:
    """Case-insensitive ``name -> path`` listing of the files in a directory (empty if unreadable)."""
    try:
        return {p.name.lower(): p for p in directory.iterdir() if p.is_file()}
    except OSError:
        return {}


def find_in(directory: Path, name: str) -> Path | None:
    return list_dir(directory).get(name.lower())


def read_imports(path: Path) -> tuple[list[str], list[str]]:
    """Return the lower-cased (static, delay-load) imports of a PE file."""
    try:
        import pefile  # pylint: disable=import-outside-toplevel
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pefile is required: pip install pefile") from e
    pe = pefile.PE(str(path), fast_load=True)  # raises pefile.PEFormatError for a non-PE file
    try:
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
        ])
        static = sorted({e.dll.decode().lower() for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])})
        delay = sorted({e.dll.decode().lower() for e in getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", [])})
    finally:
        pe.close()
    return static, delay


def torch_dll_dirs(venv_root: Path, site_packages: Path) -> list[Path]:
    """Directories ``torch/__init__.py`` registers with ``os.add_dll_directory`` in a venv."""
    candidates = [site_packages / "torch" / "lib", venv_root / "Library" / "bin", venv_root / "bin"]
    return [p for p in candidates if p.exists()]


@dataclass
class WalkOptions:
    python_dir: Path | None = None  # directory of the base interpreter (python3xx.dll)
    path_env: str | None = None  # PATH used in legacy mode (default: the current one)
    system32: Path | None = None  # directory standing in for System32 (testing off Windows)


class ImportWalker:  # pylint: disable=too-many-instance-attributes  # accumulates the findings of one walk
    """
    Resolve the transitive import closure of ``torch/_C*.pyd``.

    mode ``new``:    ``AddDllDirectory`` directories (see ``torch_dll_dirs``), the Python directory and System32.
    mode ``legacy``: ``new`` plus the importing module's own directory and ``PATH`` - what the removed preload loop
                     (``LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR`` and the ``PATH`` fallback) made resolvable.
    """

    def __init__(self, venv_root: Path, site_packages: Path, mode: str, options: WalkOptions | None = None) -> None:
        if mode not in ("new", "legacy"):
            raise ValueError(mode)
        options = options or WalkOptions()
        self.mode = mode
        self.torch_dir = site_packages / "torch"
        self.roots = sorted(self.torch_dir.glob("_C*.pyd"))
        if not self.roots:
            raise SystemExit(f"no torch/_C*.pyd under {self.torch_dir}")
        self.base_dirs = torch_dll_dirs(venv_root, site_packages)
        if options.python_dir:
            self.base_dirs.append(options.python_dir)
        path_env = options.path_env if options.path_env is not None else os.environ.get("PATH", "")
        self.path_dirs = [Path(p) for p in path_env.split(os.pathsep) if p] if mode == "legacy" else []
        self.sys_dir = options.system32 or (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32")
        self.check_system = sys.platform == "win32" or options.system32 is not None
        self.unresolved: dict[str, list[str]] = {}
        self.runtime_package_dirs: dict[str, str] = {}
        self.assumed_system: set[str] = set()
        self.unparsable: dict[str, str] = {}
        self.deps: dict[str, list[str]] = {}  # module -> modules it resolved to (for the load order)

    def resolve(self, name: str, importer: Path) -> Path | None:
        search = list(self.base_dirs)
        if self.mode == "legacy":
            search += [importer.parent, *self.path_dirs]
        for directory in search:
            hit = find_in(directory, name)
            if hit:
                return hit
        return find_in(self.sys_dir, name) if self.check_system else None

    def visit(self, mod: Path, queue: deque[Path]) -> None:
        try:
            static, delay = read_imports(mod)
        except Exception as e:  # pylint: disable=broad-exception-caught  # a corrupt file must not abort the run
            self.unparsable[mod.name] = f"{type(e).__name__}: {e}"
            return
        self.deps[mod.name] = []
        for bare, delayed in [(n, False) for n in static] + [(n, True) for n in delay]:
            if is_api_set(bare):
                continue
            hit = self.resolve(bare, mod)
            if hit is None and not self.check_system and (bare in SYSTEM_DLLS or is_vc_redist(bare)):
                self.assumed_system.add(bare)  # cannot verify System32 / the redistributable off Windows
            elif hit is None:
                self.unresolved.setdefault(bare + (" (delay)" if delayed else ""), []).append(mod.name)
            else:
                if self.torch_dir not in hit.parents and hit.parent != self.sys_dir:
                    self.runtime_package_dirs[bare] = str(hit.parent)
                if hit.suffix.lower() in (".dll", ".pyd") and not delayed:
                    self.deps[mod.name].append(hit.name)
                    queue.append(hit)

    def load_order(self) -> list[str]:
        """Dependencies first (depth-first post-order from the root)."""
        order: list[str] = []
        done: set[str] = set()

        def dfs(name: str) -> None:
            if name in done:
                return
            done.add(name)
            for dep in self.deps.get(name, []):
                dfs(dep)
            order.append(name)

        dfs(self.roots[0].name)
        return order

    def run(self) -> dict:
        seen: set[str] = set()
        queue: deque[Path] = deque(self.roots)
        while queue:
            mod = queue.popleft()
            if str(mod).lower() in seen:
                continue
            seen.add(str(mod).lower())
            self.visit(mod, queue)
        return {
            "mode": self.mode,
            "root": self.roots[0].name,
            "modules_visited": len(seen),
            "unresolved": {k: sorted(set(v))
                           for k, v in sorted(self.unresolved.items())},
            "from_runtime_packages": dict(sorted(self.runtime_package_dirs.items())),
            "assumed_system": sorted(self.assumed_system),
            "unparsable": self.unparsable,
            "dll_dirs": [str(p) for p in self.base_dirs],
            "load_order": self.load_order(),
            "edges": self.deps,
        }


def walk_imports(venv_root: Path, site_packages: Path, mode: str, options: WalkOptions | None = None) -> dict:
    return ImportWalker(venv_root, site_packages, mode, options).run()


# ----------------------------------------------------------------------------------------------------------------
# Windows loader probe.
# ----------------------------------------------------------------------------------------------------------------


def probe_load(venv_root: Path, site_packages: Path, flags: int) -> dict:
    """``LoadLibraryExW`` every module of the import closure, dependencies first, with the given search flags."""
    if sys.platform != "win32":
        return {"skipped": "not Windows"}
    import ctypes  # pylint: disable=import-outside-toplevel

    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    kernel32.LoadLibraryExW.restype = ctypes.c_void_p
    kernel32.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
    dirs = torch_dll_dirs(venv_root, site_packages)
    handles = [os.add_dll_directory(str(d)) for d in dirs]
    results: list[dict] = []
    for name in walk_imports(venv_root, site_packages, "legacy")["load_order"]:
        path = next((hit for hit in (find_in(d, name) for d in [*dirs, site_packages / "torch"]) if hit), None)
        if path is None:
            continue
        handle = kernel32.LoadLibraryExW(str(path), None, flags)
        err = 0 if handle else ctypes.get_last_error()
        results.append(
            {"module": name, "ok": bool(handle), "winerror": err, "message": ctypes.FormatError(err) if err else ""})
    for h in handles:
        h.close()
    failed = [r for r in results if not r["ok"]]
    return {"flags": hex(flags), "loaded": len(results) - len(failed), "failed": failed}


# ----------------------------------------------------------------------------------------------------------------
# Loader variants.
# ----------------------------------------------------------------------------------------------------------------

# Mirrors the loop removed from torch/__init__.py by pytorch/pytorch#202206.
OLD_PRELOAD = r"""
import ctypes, glob, importlib.util, os, sys
_root = os.path.dirname(importlib.util.find_spec("torch").origin)
_lib = os.path.join(_root, "lib")
_paths = [p for p in (_lib, os.path.join(sys.exec_prefix, "Library", "bin"), os.path.join(sys.exec_prefix, "bin"))
          if os.path.exists(p)]
for _p in _paths:
    os.add_dll_directory(_p)
_k = ctypes.WinDLL("kernel32.dll", use_last_error=True)
_k.LoadLibraryExW.restype = ctypes.c_void_p
_k.LoadLibraryW.restype = ctypes.c_void_p
_patched = False
for _dll in glob.glob(os.path.join(_lib, "*.dll")):
    _res = _k.LoadLibraryExW(_dll, None, 0x00001100)
    _err = ctypes.get_last_error()
    if _res is None and _err != 126:
        raise ctypes.WinError(_err)
    if _res is None:
        if not _patched:
            os.environ["PATH"] = ";".join(_paths + [os.environ["PATH"]])
            _patched = True
        if _k.LoadLibraryW(_dll) is None:
            raise ctypes.WinError(ctypes.get_last_error())
"""

ADD_ALL_DLL_DIRS = r"""
import os, sys
for _base in (sys.exec_prefix,):
    for _dirpath, _dirs, _files in os.walk(_base):
        if any(f.lower().endswith(".dll") for f in _files):
            try:
                os.add_dll_directory(_dirpath)
            except OSError:
                pass
"""

SMOKE = r"""
import json, torch
r = {"torch": torch.__version__, "xpu_available": bool(torch.xpu.is_available())}
if r["xpu_available"]:
    a = torch.randn(256, 256, device="xpu")
    steps = {
        "add": lambda: a + 1,
        "matmul": lambda: a @ a,
        "fft": lambda: torch.fft.fft(a),
        "linalg_inv": lambda: torch.linalg.inv(a + 64 * torch.eye(256, device="xpu")),
        "svd": lambda: torch.linalg.svdvals(a),
    }
    for name, fn in steps.items():
        try:
            fn(); torch.xpu.synchronize(); r[name] = "ok"
        except Exception as e:
            r[name] = f"{type(e).__name__}: {str(e)[:200]}"
print("SMOKE_JSON " + json.dumps(r))
"""


def oneapi_bin_dirs() -> list[str]:
    roots = [os.environ.get("ONEAPI_ROOT", ""), r"C:\Program Files (x86)\Intel\oneAPI"]
    found: list[str] = []
    for root in filter(None, roots):
        base = Path(root)
        if base.is_dir():
            found += [str(p) for p in base.glob("*/*/bin") if p.is_dir()] + [str(p) for p in base.glob("*/bin")]
    return sorted(set(found))


def run_python(python: str, code: str, env_path_prefix: list[str] | None = None, timeout: int = PROBE_TIMEOUT) -> dict:
    env = dict(os.environ)
    if env_path_prefix:
        env["PATH"] = os.pathsep.join(env_path_prefix + [env.get("PATH", "")])
    with tempfile.TemporaryDirectory() as cwd:  # not the repo: nothing local can shadow torch
        try:
            p = subprocess.run([python, "-I", "-c", code], capture_output=True, text=True, timeout=timeout, env=env,
                               cwd=cwd, check=False)
        except subprocess.TimeoutExpired:
            return {"ok": False, "returncode": None, "error": f"timeout after {timeout}s"}
    tail = "\n".join((p.stderr or "").strip().splitlines()[-6:])
    return {
        "ok": p.returncode == 0, "returncode": p.returncode, "error": tail if p.returncode else "", "stdout":
        p.stdout.strip()[-2000:]
    }


def run_variants(python: str, venv_root: Path, site_packages: Path) -> dict:
    lib = str(site_packages / "torch" / "lib")
    libbin = str(venv_root / "Library" / "bin")
    variants: dict[str, tuple[str, list[str]]] = {
        "baseline": ("import torch", []),
        "path_torch_lib_only": ("import torch", [lib]),
        "path_library_bin_only": ("import torch", [libbin]),
        "path_prefix": ("import torch", [libbin, lib]),
        "add_all_dll_dirs": (ADD_ALL_DLL_DIRS + "import torch", []),
        "old_preload_emulation": (OLD_PRELOAD + "import torch", []),
        "system_oneapi_on_path": ("import torch", oneapi_bin_dirs()),
    }
    out: dict[str, dict] = {}
    for name, (code, prefix) in variants.items():
        if name == "system_oneapi_on_path" and not prefix:
            out[name] = {"skipped": "no oneAPI directory found"}
            continue
        res = run_python(python, code, prefix)
        if res["ok"]:
            smoke = run_python(python, code.replace("import torch", "") + SMOKE, prefix)
            line = next((ln for ln in smoke.get("stdout", "").splitlines() if ln.startswith("SMOKE_JSON ")), None)
            res["smoke"] = json.loads(line[len("SMOKE_JSON "):]) if line else {"error": smoke.get("error", "no output")}
        out[name] = res
    return out


# ----------------------------------------------------------------------------------------------------------------
# Machine snapshot.
# ----------------------------------------------------------------------------------------------------------------


def powershell(cmd: str) -> str:
    exe = shutil.which("pwsh") or shutil.which("powershell")
    if not exe:
        return ""
    p = subprocess.run([exe, "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=120, check=False)
    return p.stdout.strip()


def snapshot(python: str | None) -> dict:
    names = [
        "sycl9.dll", "ur_loader.dll", "ur_win_proxy_loader.dll", "libmmd.dll", "pti_view-1.dll", "vcruntime140.dll",
        "vcruntime140_1.dll", "msvcp140.dll", "ze_loader.dll", "libiomp5md.dll"
    ]
    path_dirs = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    on_path: dict[str, list[str]] = {}
    for n in names:
        hits = [d for d in path_dirs if (Path(d) / n).is_file()]
        if hits:
            on_path[n] = hits
    snap: dict = {
        "platform": platform.platform(),
        "python": sys.version,
        "path": path_dirs,
        "dlls_on_path": on_path,
        "oneapi_bin_dirs": oneapi_bin_dirs(),
    }
    if sys.platform == "win32":
        import winreg  # pylint: disable=import-outside-toplevel

        redist = {}
        for arch in ("X64", "X86"):
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                    rf"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\{arch}") as k:
                    redist[arch] = winreg.QueryValueEx(k, "Version")[0]
            except OSError:
                redist[arch] = None
        snap["vc_redist"] = redist
        snap["gpu"] = powershell("Get-CimInstance Win32_VideoController | "
                                 "Select-Object Name,DriverVersion,DriverDate | ConvertTo-Json -Compress")
        snap["os_build"] = powershell("(Get-CimInstance Win32_OperatingSystem | "
                                      "Select-Object Caption,Version,BuildNumber) | ConvertTo-Json -Compress")
    if python:
        p = subprocess.run([python, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
        snap["pip_freeze"] = p.stdout.splitlines()
    return snap


# ----------------------------------------------------------------------------------------------------------------
# Orchestration and reporting.
# ----------------------------------------------------------------------------------------------------------------


def default_site_packages(venv_root: Path) -> Path:
    win = venv_root / "Lib" / "site-packages"
    return win if win.exists() else next((venv_root / "lib").glob("python*/site-packages"), win)


def cmd_run(args: argparse.Namespace) -> None:
    venv_root = Path(args.venv_root)
    site_packages = Path(args.site_packages) if args.site_packages else default_site_packages(venv_root)
    python = args.venv_python
    result: dict = {"label": args.label}
    version = run_python(python, "import importlib.metadata as m; print(m.version('torch'))", timeout=60)
    result["torch_version"] = version.get("stdout") or args.label
    # python311.dll lives next to the base interpreter, not next to the venv's launcher.
    base = run_python(python, "import sys; print(sys.base_prefix)", timeout=60).get("stdout")
    py_dir = Path(base) if base else Path(python).parent
    path_env = os.environ.get("PATH", "")
    for mode in ("new", "legacy"):
        w = walk_imports(venv_root, site_packages, mode, WalkOptions(python_dir=py_dir, path_env=path_env))
        w.pop("edges")
        result[f"walk_{mode}"] = w
    result["probe_new_flags"] = probe_load(venv_root, site_packages, FLAGS_NEW)
    result["probe_old_flags"] = probe_load(venv_root, site_packages, FLAGS_OLD)
    result["variants"] = run_variants(python, venv_root, site_packages)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: result[k]
                    for k in ("torch_version", )} | {
                        "baseline_ok": result["variants"]["baseline"]["ok"],
                        "unresolved_new": list(result["walk_new"]["unresolved"]),
                    }, indent=2))


def first_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""


def summarize(results_dir: Path) -> str:
    rows = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json")) if p.name != "snapshot.json"]
    install_failed = sorted(
        p.name.removesuffix(".install-failed.txt") for p in results_dir.glob("*.install-failed.txt"))
    if not rows:
        return "No results.\n" + "".join(f"\n- `{label}`: pip install failed\n" for label in install_failed)
    variant_names = list(rows[0]["variants"])
    lines = [
        "## `import torch` diagnostics", "",
        "| torch | " + " | ".join(variant_names) + " | unresolved (new loader) | unresolved (old loader) |",
        "|---|" + "---|" * (len(variant_names) + 2)
    ]
    for r in rows:
        cells = []
        for v in variant_names:
            res = r["variants"].get(v, {})
            if "skipped" in res:
                cells.append("skipped")
            elif res.get("ok"):
                smoke = res.get("smoke", {})
                bad = [k for k, val in smoke.items() if isinstance(val, str) and val != "ok" and k != "torch"]
                cells.append("pass" + (f" (smoke fails: {', '.join(bad)})" if bad else ""))
            else:
                cells.append("**FAIL**")
        new = ", ".join(r["walk_new"]["unresolved"]) or "-"
        old = ", ".join(r["walk_legacy"]["unresolved"]) or "-"
        lines.append(f"| {r['torch_version']} | " + " | ".join(cells) + f" | {new} | {old} |")
    lines += [
        f"| {label} | " + " | ".join(["pip install failed"] + [""] * (len(variant_names) + 1)) + " |"
        for label in install_failed
    ]
    lines += ["", "### Details", ""]
    for r in rows:
        lines.append(f"<details><summary>{r['torch_version']}</summary>\n")
        for v, res in r["variants"].items():
            if not res.get("ok") and "skipped" not in res:
                lines.append(f"- `{v}`: `{first_line(res.get('error', ''))}`")
        for key in ("probe_new_flags", "probe_old_flags"):
            failed = r[key].get("failed") or []
            lines.append(f"- `{key}`: " +
                         ("; ".join(f"{f['module']}: [{f['winerror']}] {f['message']}"
                                    for f in failed) if failed else r[key].get("skipped", "all loaded")))
        lines.append(f"- DLLs taken from pip runtime packages: `{', '.join(r['walk_new']['from_runtime_packages'])}`")
        lines.append("\n</details>\n")
    return "\n".join(lines) + "\n"


def cmd_summarize(args: argparse.Namespace) -> None:
    text = summarize(Path(args.results))
    print(text)
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as f:
            f.write(text)


def cmd_snapshot(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = snapshot(args.venv_python)
    (out / "snapshot.json").write_text(json.dumps(data, indent=2))
    print(json.dumps({k: v for k, v in data.items() if k not in ("pip_freeze", "path", "python")}, indent=2))


def cmd_walk(args: argparse.Namespace) -> None:
    venv_root = Path(args.venv_root)
    site_packages = Path(args.site_packages) if args.site_packages else default_site_packages(venv_root)
    options = WalkOptions(
        Path(args.python_dir) if args.python_dir else None, args.path_env,
        Path(args.system32) if args.system32 else None)
    w = walk_imports(venv_root, site_packages, args.mode, options)
    if not args.edges:
        w.pop("edges")
    print(json.dumps(w, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("snapshot")
    p.add_argument("--out", required=True)
    p.add_argument("--venv-python")
    p.set_defaults(fn=cmd_snapshot)

    p = sub.add_parser("walk")
    p.add_argument("--venv-root", required=True)
    p.add_argument("--site-packages")
    p.add_argument("--mode", choices=("new", "legacy"), default="new")
    p.add_argument("--python-dir")
    p.add_argument("--path-env", help="PATH to use in legacy mode (default: the current PATH)")
    p.add_argument("--system32", help="directory standing in for System32 (testing)")
    p.add_argument("--edges", action="store_true", help="include the full import graph")
    p.set_defaults(fn=cmd_walk)

    p = sub.add_parser("run")
    p.add_argument("--venv-root", required=True)
    p.add_argument("--venv-python", required=True)
    p.add_argument("--site-packages")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("summarize")
    p.add_argument("--results", required=True)
    p.set_defaults(fn=cmd_summarize)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
