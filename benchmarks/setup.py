import os
import shutil
import subprocess
import sys
import logging

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext

import torch


class CMakeExtension(Extension):

    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class CMakeBuild():

    def __init__(self, build_lib, build_temp, debug=False, dry_run=False):
        self.current_dir = os.path.abspath(os.path.dirname(__file__))
        self.build_temp = build_temp
        self.extdir = build_lib + "/triton_kernels_benchmark"
        self.build_type = self.get_build_type(debug)
        self.cmake_prefix_paths = [torch.utils.cmake_prefix_path]
        self.dry_run = dry_run

    def get_build_type(self, debug):
        DEBUG_OPTION = os.getenv("DEBUG", "0")
        return "Debug" if debug or (DEBUG_OPTION == "1") else "Release"

    def run(self):
        self.build_extension()

    def check_call(self, *popenargs, **kwargs):
        logging.info(" ".join(popenargs[0]))
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
            "-DCMAKE_INSTALL_PREFIX=" + self.extdir,
            "-DPython3_ROOT_DIR:FILEPATH=" + sys.exec_prefix,
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


class build_ext(_build_ext):

    def run(self):
        cmake = CMakeBuild(
            build_lib=self.build_lib,
            build_temp=self.build_temp,
            debug=self.debug,
            dry_run=self.dry_run,
        )
        cmake.run()
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
    version="3.6.0" + get_git_commit_hash(),
    packages=find_packages(),
    install_requires=[
        "torch>=2.6",
        "pandas",
        "scipy",
        "psutil",
        "tabulate",
        "matplotlib",
    ],
    package_dir={"triton_kernels_benchmark": "triton_kernels_benchmark"},
    package_data={
        "triton_kernels_benchmark": [
            "xetla_kernel.cpython-*.so",
            "cutlass_kernel.cpython-*.so",
            "onednn_kernel.cpython-*.so",
        ]
    },
    cmdclass={
        "build_ext": build_ext,
    },
    ext_modules=[
        CMakeExtension("triton_kernels_benchmark.xetla_kernel"),
        CMakeExtension("triton_kernels_benchmark.cutlass_kernel"),
        CMakeExtension("triton_kernels_benchmark.onednn_kernel"),
    ],
    entry_points={
        "console_scripts": [
            "triton-benchmarks = triton_kernels_benchmark.benchmark_utils:main",
        ],
    },
)
