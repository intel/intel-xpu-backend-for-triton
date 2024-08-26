from os import chdir, makedirs
from tempfile import TemporaryDirectory
from subprocess import run

with TemporaryDirectory() as tmpdir:
    pkg = "intel_extension_for_pytorch"
    version = "2.4.0+noop"

    chdir(tmpdir)
    makedirs(pkg, exist_ok=True)

    files = {
        f"{pkg}/__init__.py":
        f"__version__ = \"{version}\"", "setup.py":
        ("from setuptools import setup, find_packages\n"
         f"setup(name='{pkg}', version='{version}', packages=find_packages())\n"), "project.toml":
        ("[build-system]\n"
         "requires = [\"setuptools\", \"wheel\"]\n"
         "build-backend = \"setuptools.build_meta\"\n")
    }
    for file, content in files.items():
        with open(file, "w") as f:
            f.write(content)
    cmds = [
        f"pip uninstall -y {pkg}", "pip install build", "python -m build .",
        f"pip install dist/{pkg}-{version}-py3-none-any.whl"
    ]
    for cmd in cmds:
        run(cmd.split(), check=True)
