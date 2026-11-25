"""TEMPORARY diagnostics for the Windows abi3 build. Do not merge."""
import json
import sys
import zipfile
from pathlib import Path

import pefile


def check_wheel(wheel):
    print(f"== wheel: {wheel.name}")
    bad = False
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if not name.endswith((".pyd", ".dll")):
                continue
            pe = pefile.PE(data=z.read(name), fast_load=True)
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
            deps = sorted({e.dll.decode().lower() for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])})
            py = [d for d in deps if d.startswith("python")]
            print(f"{name}: python deps = {py}")
            if any(d != "python3.dll" for d in py):
                bad = True
    return bad


def check_compile_commands(root):
    for cc in Path(root).rglob("compile_commands.json"):
        print(f"== compile_commands: {cc}")
        entries = json.loads(cc.read_text())
        for e in entries:
            f = e["file"].replace("\\", "/")
            cmd = e.get("command") or " ".join(e.get("arguments", []))
            if "third_party/proton" in f or "nanobind" in f or f.endswith("python/src/specialize.cc"):
                print(f"{'OK ' if 'Py_LIMITED_API' in cmd else 'MISSING'} Py_LIMITED_API  {f}")


if __name__ == "__main__":
    bad = False
    for w in Path(sys.argv[1]).glob("*.whl"):
        bad |= check_wheel(w)
    check_compile_commands(sys.argv[2])
    sys.exit(1 if bad else 0)
