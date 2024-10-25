import os
import shutil
import subprocess
import sys

# TODO: update once there is replacement for clean:
#  https://github.com/pypa/setuptools/discussions/2838
from distutils import log  # pylint: disable=[deprecated-module]
from distutils.dir_util import remove_tree  # pylint: disable=[deprecated-module]
from distutils.command.clean import clean as _clean  # pylint: disable=[deprecated-module]

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

import torch


class CMakeExtension(Extension):

    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class CMakeBuild():

    def __init__(self, debug=False, dry_run=False):
        self.current_dir = os.path.abspath(os.path.dirname(__file__))
        self.build_temp = self.current_dir + "/build/temp"
        self.extdir = self.current_dir + "/triton_kernels_benchmark"
        self.build_type = self.get_build_type(debug)
        self.cmake_prefix_paths = [torch.utils.cmake_prefix_path]
        self.use_ipex = False
        self.dry_run = dry_run

    def get_build_type(self, debug):
        DEBUG_OPTION = os.getenv("DEBUG", "0")
        return "Debug" if debug or (DEBUG_OPTION == "1") else "Release"

    def run(self):
        self.check_ipex()
        self.build_extension()

    def check_ipex(self):
        self.use_ipex = os.getenv("USE_IPEX", "1") == "1"
        if not self.use_ipex:
            return
        try:
            import intel_extension_for_pytorch
        except ImportError:
            log.warn("ipex is not installed trying to build without ipex")
            self.use_ipex = False
            return
        self.cmake_prefix_paths.append(intel_extension_for_pytorch.cmake_prefix_path)

    def check_call(self, *popenargs, **kwargs):
        log.info(" ".join(popenargs[0]))
        if not self.dry_run:
            subprocess.check_call(*popenargs, **kwargs)

    def build_extension(self):
        ninja_dir = shutil.which("ninja")
        # create build directories
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        cmake_args = [
            "-G",
            "Ninja",  # Ninja is much faster than make
            "-DCMAKE_MAKE_PROGRAM=" +
            ninja_dir,  # Pass explicit path to ninja otherwise cmake may cache a temporary path
            "-DCMAKE_PREFIX_PATH=" + ";".join(self.cmake_prefix_paths),
            f"-DUSE_IPEX={int(self.use_ipex)}",
            "-DCMAKE_INSTALL_PREFIX=" + self.extdir,
            "-DPython3_ROOT_DIR:FILEPATH=" + sys.exec_prefix,
            "-DCMAKE_VERBOSE_MAKEFILE=TRUE",
            "-DCMAKE_C_COMPILER=icx",
            "-DCMAKE_CXX_COMPILER=icpx",
            "-DCMAKE_BUILD_TYPE=" + self.build_type,
            "-S",
            self.current_dir,
            "-B",
            self.build_temp,
        ]

        max_jobs = os.getenv("MAX_JOBS", str(2 * os.cpu_count()))
        build_args = [
            "--build",
            self.build_temp,
            "-j" + max_jobs,
        ]

        install_args = [
            "--build",
            self.build_temp,
            "--target",
            "install",
        ]

        env = os.environ.copy()
        self.check_call(["cmake"] + cmake_args, env=env)
        self.check_call(["cmake"] + build_args)
        self.check_call(["cmake"] + install_args)

    def clean(self):
        if os.path.exists(self.build_temp):
            remove_tree(self.build_temp, dry_run=self.dry_run)
        else:
            log.warn("'%s' does not exist -- can't clean it", os.path.relpath(self.build_temp,
                                                                              os.path.dirname(__file__)))


class build_ext(_build_ext):

    def run(self):
        cmake = CMakeBuild(debug=self.debug, dry_run=self.dry_run)
        cmake.run()
        super().run()


class clean(_clean):

    def run(self):
        cmake = CMakeBuild(dry_run=self.dry_run)
        cmake.clean()
        super().run()


def get_git_commit_hash(length=8):
    try:
        cmd = ["git", "rev-parse", f"--short={length}", "HEAD"]
        return f"+git{subprocess.check_output(cmd).strip().decode('utf-8')}"
    except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
    ):
        return ""


setup(
    name="triton-kernels-benchmark",
    version="3.1.0" + get_git_commit_hash(),
    packages=["triton_kernels_benchmark"],
    install_requires=[
        "torch",
        "pandas",
        "tabulate",
        "matplotlib",
    ],
    package_dir={"triton_kernels_benchmark": "triton_kernels_benchmark"},
    package_data={"triton_kernels_benchmark": ["xetla_kernel.cpython-*.so"]},
    cmdclass={
        "build_ext": build_ext,
        "clean": clean,
    },
    ext_modules=[CMakeExtension("triton_kernels_benchmark")],
    extra_require={
        "ipex": ["numpy<=2.0", "intel-extension-for-pytorch=2.1.10"],
        "pytorch": ["torch>=2.6"],
    },
)
