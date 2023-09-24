import math
import os
from pathlib import Path
import re
import tempfile
import torch
from triton.runtime.driver import DriverBase
from triton.common.backend import BaseBackend, register_backend  # noqa:E402
from triton.compiler.make_launcher import make_so_cache_key  # noqa:E402
from triton.runtime.cache import get_cache_manager  # noqa:E402
from triton.runtime.jit import version_key  # noqa:E402
import subprocess
from jinja2 import Environment, FileSystemLoader

# TODO(jgong5): set via env
TRITON_SHARED_OPT = os.environ.get("TRITON_SHARED_OPT", "triton-shared-opt")
MLIR_OPT = os.environ.get("MLIR_OPT", "mlir-opt")
MLIR_TRANSLATE = os.environ.get("MLIR_TRANSLATE", "mlir-translate")
LLVM_LLC = os.environ.get("LLVM_LLC", "llc")

def run_command(command, stdin=""):
    try:
        print(f"Running command: {command}")
        print("Input:")
        print(stdin)
        # Launch the external process and communicate with it
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        stdout, stderr = process.communicate(input=stdin)

        # Check for any errors in stderr
        if process.returncode != 0:
            raise Exception(f"Error executing command: {stderr}")

        # Return the stdout of the process
        print("Output:")
        print(stdout)
        return stdout

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def ttir_to_linalg(mod):
    # launch process "TRITON_SHARED_OPT --triton-to-linalg"
    # feed `mod` as stdin to the process 
    # return the stdout of it

    # Command to launch the external process
    command = f"{TRITON_SHARED_OPT} --triton-to-linalg"
    return run_command(command, str(mod))


def linalg_to_llir(mod, arch):
    command = f'{MLIR_OPT} -empty-tensor-to-alloc-tensor -one-shot-bufferize="allow-return-allocs" --test-lower-to-llvm'
    return run_command(command, str(mod))


def llir_to_llvm(mod, arch):
    command = f"{MLIR_TRANSLATE} --mlir-to-llvmir"
    return run_command(command, str(mod))


def llvm_to_x86(mod, arch):
    # build llvm file to the shared library
    # return the binary content
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_obj = os.path.join(tmpdir, "tmp.o")
        tmp_so = os.path.join(tmpdir, "tmp.so")
        command = f"{LLVM_LLC} -filetype=obj -O3 -o {tmp_obj} --"
        run_command(command, str(mod))
        command = f"g++ -shared -o {tmp_so} {tmp_obj}"
        run_command(command)
        # read tmp_so as bytes and return
        with open(tmp_so, "rb") as f:
            return f.read()


def llvm_get_kernel_name(mod):
    # extract the kerne name from the string pattern below:
    # define void @kerne_name(
    pattern = r'^define void @(\w+)\('
    lines = str(mod).strip().split('\n')
    for line in lines:
        match = re.search(pattern, line)
        if match:
            kernel_name = match.group(1)
            print(f"kernel name: {kernel_name}")
            return kernel_name
    raise Exception("Cannot find kerne name!")


class CPUBackend(BaseBackend):
    def add_stages(self, arch, extern_libs, stages):
        """
        Custom the arch, extern_libs and stages per backend specific requirement
        """
        filter_in_stages = ["ast", "ttir"]
        filter_out_stages = []
        for key, _ in stages.items():
            if key not in filter_in_stages:
                filter_out_stages.append(key)
        for filter_out_key in filter_out_stages:
            stages.pop(filter_out_key)

        stages["linalg"] = (lambda path: Path(path).read_text(),
                            lambda src: ttir_to_linalg(src))
        stages["llvm-mlir"] = (lambda path: None,
                          lambda src: linalg_to_llir(src, arch))
        stages["llvm"] = (lambda path: None,
                          lambda src: llir_to_llvm(src, arch))
        stages["x86"] = (lambda path: None,
                         lambda src: llvm_to_x86(src, arch))

    def add_meta_info(self, ir, cur_module, next_module, metadata, asm):
        """
        Custom the ir, module, metadata and asm per backend specific requirement
        """
        if ir == "llvm":
            metadata["name"] = llvm_get_kernel_name(next_module)

        if ir == "x86":
            asm["so"] = next_module

    def get_load_binary_fn(self):
        """
        Return a callable to load binary
        """
        def _load_binary_fn(kernel_name, binary, shared_size, device):
            import ctypes
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_so = os.path.join(tmpdir, "tmp.so")
                with open(tmp_so, "wb") as f:
                    f.write(binary)
                mod = ctypes.CDLL(tmp_so)
                return mod, getattr(mod, kernel_name), 0, 0

        return _load_binary_fn

    def get_driver(self) -> DriverBase:
        """
        Get the backend driver. Please refer to "DriverBase" for more details
        """
        raise NotImplementedError

    def get_stream(self):
        """
        Get stream for current device
        """
        return None

    def get_device_properties(self, device):
        # TODO(jgong5): add more property for CPU like number of cores, cache size, etc.
        return {"max_shared_mem": math.inf}

    def get_current_device(self):
        """
        Get current device
        """
        return 0

    def set_current_device(self, device):
        """
        Set current device as the given device
        """
        pass

    def get_kernel_bin(self):
        return "x86bin"

    def make_launcher_stub(self, name, signature, constants, ids):
        """
        Generate the launcher stub to launch the kernel
        """

        def ty_to_cpp(ty):
            if ty[0] == '*':
                return "void*"
            return {
                "i1": "int32_t",
                "i8": "int8_t",
                "i16": "int16_t",
                "i32": "int32_t",
                "i64": "int64_t",
                "u32": "uint32_t",
                "u64": "uint64_t",
                "fp16": "float",
                "bf16": "float",
                "fp32": "float",
                "f32": "float",
                "fp64": "double",
            }[ty]

        def _extracted_type_pybind11(ty):
            if ty[0] == '*':
                return "py::object"
            return {
                'i1': 'int32_t',
                'i32': 'int32_t',
                'i64': 'int64_t',
                'u32': 'uint32_t',
                'u64': 'uint64_t',
                'fp32': 'float',
                'f32': 'float',
                'fp64': 'double',
            }[ty]

        def generate_launcher():
            # load jinja template from the same folder as the module
            env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
            template = env.get_template('launcher.j2')
            context = {
                "args_decl": ', '.join(f"{_extracted_type_pybind11(ty)} arg{i}" for i, ty in signature.items()),
                "func_args_decl": ', '.join(f"{ty_to_cpp(ty)}" for _, ty in signature.items()),
                "args": ', '.join(f"getPointer(arg{i},{i})" if ty[0] == "*" else f"arg{i}" for i, ty in signature.items()),
            }
            rendered_template = template.render(context)
            print("Launcher:")
            print(rendered_template)
            return rendered_template

        def build_launcher(so_name, src_path, tmpdir):
            from torch.utils import cpp_extension
            import sysconfig

            so_path = os.path.join(tmpdir, so_name)
            # invoke GCC to build the launcher
            ipaths = cpp_extension.include_paths() + [sysconfig.get_path("include")]
            command = f"g++ {' '.join(['-I' + p for p in ipaths])} -shared -fPIC -fopenmp -O3 -o {so_path} {src_path}"
            print(f"Running command: {command}")
            subprocess.run(command, shell=True, check=True)
            return so_path

        # name of files that are cached
        so_cache_key = make_so_cache_key(version_key(), signature, constants, ids)
        so_cache_manager = get_cache_manager(so_cache_key)
        so_name = f"{name}.so"
        # retrieve stub from cache if it exists
        cache_path = so_cache_manager.get_file(so_name)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src = generate_launcher()
                src_path = os.path.join(tmpdir, "main.cpp")
                with open(src_path, "w") as f:
                    f.write(src)
                so = build_launcher(so_name, src_path, tmpdir)
                with open(so, "rb") as f:
                    return so_cache_manager.put(f.read(), so_name, binary=True)
        else:
            return cache_path

    def get_architecture_descriptor(self, **kwargs):
        """
        Get the architecture descriptor the backend
        """
        return {"num_warps": torch.get_num_threads(), "threads_per_warp": 1, "num_stages": 1}

register_backend("cpu", CPUBackend)