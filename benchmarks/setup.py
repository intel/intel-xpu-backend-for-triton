import os
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.build_ext import build_ext as build_ext_orig

import torch


class CMakeBuild():

    def __init__(self, build_type="Debug"):
        self.current_dir = os.path.abspath(os.path.dirname(__file__))
        self.build_temp = self.current_dir + "/build/temp"
        self.extdir = self.current_dir + "/triton_kernels_benchmark"
        self.build_type = build_type
        self.cmake_prefix_paths = [torch.utils.cmake_prefix_path]
        self.use_ipex = False

    def run(self):
        self.check_ipex()
        self.build_extension()

    def check_ipex(self):
        self.use_ipex = os.getenv("USE_IPEX", "1") == "1"
        if not self.use_ipex:
            return
        import intel_extension_for_pytorch
        self.cmake_prefix_paths.append(intel_extension_for_pytorch.cmake_prefix_path)

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
            "-DUSE_IPEX=" + ("1" if self.use_ipex else "0"),
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
        print(" ".join(["cmake"] + cmake_args))
        subprocess.check_call(["cmake"] + cmake_args, env=env)
        print(" ".join(["cmake"] + build_args))
        subprocess.check_call(["cmake"] + build_args)
        print(" ".join(["cmake"] + install_args))
        subprocess.check_call(["cmake"] + install_args)


class build_ext(build_ext_orig):

    def run(self):
        self.build_cmake()
        super().run()

    def build_cmake(self):
        DEBUG_OPTION = os.getenv("DEBUG", "0")
        build_type = "Debug" if self.debug or DEBUG_OPTION == "1" else "Release"
        cmake = CMakeBuild(build_type)
        cmake.run()


setup(name="triton-kernels-benchmark", packages=[
    "triton_kernels_benchmark",
], package_dir={
    "triton_kernels_benchmark": "triton_kernels_benchmark",
}, package_data={"triton_kernels_benchmark": ["xetla_kernel.cpython-*.so"]}, cmdclass={
    "build_ext": build_ext,
})
