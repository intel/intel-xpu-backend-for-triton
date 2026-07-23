import importlib.metadata
import os
import json
import sys
import re
import hashlib
import shutil
import ctypes
import subprocess
import sysconfig
import tempfile
from pathlib import Path
from functools import cached_property, lru_cache

from triton import knobs
import triton
from triton.runtime.build import _build, platform_key, _load_module_from_path
from triton.runtime.cache import get_cache_manager
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase, decompose_descriptor
from triton.backends.driver import expand_signature, wrap_handle_tensordesc_impl

# A hard-coded cache version that can be updated when we know that the cached file is invalid and
# there are no other ways to detect that the runtime environment has changed. For example, a shared
# library has been updated as a result of updated dependencies.
# See https://github.com/intel/intel-xpu-backend-for-triton/issues/3095.
__CACHE_VERSION = "1"

PyKernelArg = None
ARG_CONSTEXPR = None
ARG_KERNEL = None
ARG_TUPLE = None


def find_sycl_icpx(include_dir: list[str]) -> tuple[list[str], list[str]]:
    include_dir = include_dir.copy()
    icpx_path = shutil.which("icpx")
    if icpx_path:
        # only `icpx` compiler knows where sycl runtime binaries and header files are
        compiler_root = os.path.abspath(f"{icpx_path}/../..")
        include_dir += [os.path.join(compiler_root, "include"), os.path.join(compiler_root, "include/sycl")]
        sycl_dir = os.path.join(compiler_root, "lib")
        return include_dir, [sycl_dir]

    oneapi_root = os.getenv("ONEAPI_ROOT")
    if oneapi_root:
        include_dir += [
            os.path.join(oneapi_root, "compiler/latest/include"),
            os.path.join(oneapi_root, "compiler/latest/include/sycl")
        ]
        sycl_dir = os.path.join(oneapi_root, "compiler/latest/lib")
        return include_dir, [sycl_dir]

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
        if any(map(lambda el: el in f.name, ("libsycl.so", "sycl.lib"))):
            sycl_dir = f.locate().parent.resolve()
            if os.name == "nt":
                # for sycl8.dll loading on Windows
                dll_path = sycl_dir.parent.joinpath("bin")
                sycl_dirs.append(str(dll_path))
                _ = os.add_dll_directory(str(dll_path))
            sycl_dirs.append(str(sycl_dir))

    return include_dir, sycl_dirs


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


def find_sycl(include_dir: list[str]) -> tuple[list[str], list[str]]:
    """
    Looks for the sycl library in known places.

    Arguments:
      include_dir: list of include directories to pass to compiler.

    Returns:
      enriched include_dir and libsycl.so location.

    Raises:
      AssertionError: if library was not found.
    """

    sycl_dirs = []
    csycl = knobs.intel.sycl_compiler
    if not csycl or csycl == "icpx":
        include_dir, sycl_dirs = find_sycl_icpx(include_dir)
    if len(sycl_dirs) == 0 and (not csycl or csycl.startswith("dpclang")):
        include_dir, sycl_dirs = find_sycl_dpclang(include_dir)
    if len(sycl_dirs) == 0:
        raise AssertionError("sycl headers not found, please install `icpx` compiler, "
                             "or provide `ONEAPI_ROOT` environment "
                             "or install `intel-sycl-rt>=2025.0.0` wheel"
                             "or instal `dpclang` compiler (experimental)")

    return include_dir, sycl_dirs


class CompilationHelper:
    _library_dir: list[str]
    _include_dir: list[str]
    libraries: list[str]

    def __init__(self):
        self._library_dir = None
        self._include_dir = None
        self._libsycl_dir = None
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
        include_dir, self._libsycl_dir = find_sycl(include_dir)
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


COMPILATION_HELPER = CompilationHelper()


class ArchParser:

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
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
        # driver.active.utils.load_binary(self.name, self.kernel, self.metadata.shared, self.metadata.build_flags, device) ->
        # driver.active.utils.load_binary((self.name, self.kernel, self.metadata.shared, self.metadata.build_flags, device))
        # PTSS-overflow detection happens at the C level (driver.c
        # tryRaisePTSSOutOfResources): when zeModuleCreate fails and the
        # IGC build log carries a PTSS marker, OutOfResources is raised
        # directly so triton.runtime.autotuner can skip the offending tile.
        # No per-launch overhead on the success path.
        try:
            return self.shared_library.load_binary(args)
        except Exception as e:
            from triton.runtime.errors import IntelGPUError, OutOfResources
            if isinstance(e, OutOfResources):
                raise
            if str(e).startswith("ZE_"):
                raise IntelGPUError("Error during Intel load_binary: " + str(e)) from e
            else:
                raise e

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


_VERSION_PATTERN = re.compile(r'(\d+)\.(\d+)\.(\d+)(?:\+(\d+))?')


def is_lts(ver) -> bool:
    if not ver:
        return True
    m = _VERSION_PATTERN.match(ver)
    if not m:
        return True
    return tuple(int(x) if x is not None else 0 for x in m.groups()) < (1, 6, 35096, 9)


@lru_cache
def get_hasher_common(is_lts: bool = False):
    hasher = hashlib.sha256((__CACHE_VERSION + platform_key()).encode("utf-8"))
    # Include libsycl_dir in the hash to prevent cache collisions across
    # environments with different oneAPI versions (e.g. 2025.3 vs 2026.0).
    # The compiled .so has libsycl_dir baked in as RPATH; without this,
    # two envs with identical extension_utils.c but different oneAPI stacks
    # share the same cache entry and load an incompatible .so.
    if COMPILATION_HELPER.libsycl_dir:
        hasher.update(str(COMPILATION_HELPER.libsycl_dir).encode("utf-8"))
    if is_lts:
        hasher.update("is_lts=True".encode("utf-8"))
    return hasher


def compile_module_from_src(src: str, name: str, is_lts: bool = False):
    hasher = get_hasher_common(is_lts).copy()
    hasher.update(src.encode("utf-8"))
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
            libraries = list(COMPILATION_HELPER.libraries)
            library_dir = list(COMPILATION_HELPER.library_dir)
            if COMPILATION_HELPER.libsycl_dir:
                if os.name == "nt":
                    extra_compiler_args += ["/LIBPATH:" + dir for dir in COMPILATION_HELPER.libsycl_dir]
                else:
                    extra_compiler_args += ["-Wl,-rpath," + dir for dir in COMPILATION_HELPER.libsycl_dir]

            if COMPILATION_HELPER.inject_pytorch_dep and name == "spirv_utils":
                if os.name == "nt":
                    extra_compiler_args += ["/DTRITON_INTEL_INJECT_PYTORCH=1"]
                else:
                    extra_compiler_args += ["-DTRITON_INTEL_INJECT_PYTORCH=1"]

            if name == "spirv_utils":
                # Build and load arch_utils first, then link spirv_utils against it.
                arch_utils = get_arch_utils_module()
                arch_utils_dir = os.path.dirname(arch_utils.cache_path)
                arch_utils_lib = os.path.basename(arch_utils.cache_path)
                libraries += [arch_utils_lib]
                library_dir += [arch_utils_dir]
                if os.name == "nt":
                    extra_compiler_args += [f"/LIBPATH:{arch_utils_dir}"]
                else:
                    extra_compiler_args += [f"-Wl,-rpath,{arch_utils_dir}"]

            if name == "spirv_utils" and not is_lts:
                if os.name == "nt":
                    extra_compiler_args += ["/DENABLE_EXPERIMENTAL_EVENTLESS_SUBMIT"]
                else:
                    extra_compiler_args += ["-DENABLE_EXPERIMENTAL_EVENTLESS_SUBMIT"]

            so = _build(name, src_path, tmpdir, library_dir, COMPILATION_HELPER.include_dir, libraries,
                        ccflags=extra_compiler_args)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}{suffix}", binary=True)

    if name == 'arch_utils':
        return ArchParser(cache_path)
    if name == 'spirv_utils':
        return SpirvUtils(cache_path)
    if name == 'extension_utils_impl':
        return ExtensionUtils(cache_path)
    if name == 'proton_utils':
        return cache_path

    return _load_module_from_path(name, cache_path)


@lru_cache
def get_arch_utils_module():
    dirname = os.path.dirname(os.path.realpath(__file__))
    src = Path(os.path.join(dirname, "arch_parser.c")).read_text()
    return compile_module_from_src(src=src, name="arch_utils", is_lts=False)


@lru_cache
def get_spirv_utils_module():
    dirname = os.path.dirname(os.path.realpath(__file__))
    # Use class-level capability probe; this helper can be called outside any XPUUtils instance.
    is_lts_driver = XPUUtils._is_lts()
    mod = compile_module_from_src(
        src=Path(os.path.join(dirname, "driver.c")).read_text(),
        name="spirv_utils",
        is_lts=is_lts_driver,
    )
    return mod


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
        # we save `spirv_utils` module so that the destructor is not called prematurely, which will unload the dll
        # and can cause `Fatal Python error: Segmentation fault`
        mod = get_spirv_utils_module()
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

    @classmethod
    def get_current_device(cls):
        try:
            from torch._C import _xpu_getDevice
            return _xpu_getDevice()
        except ImportError:
            import torch
            return torch.xpu.current_device()

    def get_sycl_queue(self):
        import torch
        return torch.xpu.current_stream().sycl_queue

    @classmethod
    @lru_cache
    def _get_device_capability(cls, device):
        import torch
        return torch.xpu.get_device_capability(device)

    @classmethod
    @lru_cache
    def _is_lts(cls):
        device = cls.get_current_device()
        properties = cls._get_device_capability(device)
        return is_lts(properties.get('driver_version'))

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
        "fp16": "double",
        "bf16": "double",
        "fp32": "double",
        "f32": "double",
        "fp64": "double",
    }[ty]


def make_kernel_signature(signature):
    """
    Creates a kernel signature in C to be able to efficiently extract
    arguments in the launcher.
    """

    def _flatten_signature(sig, output):
        # Flatten tuples
        if isinstance(sig, tuple):
            for x in sig:
                _flatten_signature(x, output)
        else:
            output.append(sig)

    flat_signature = []
    for sig in signature:
        _flatten_signature(sig, flat_signature)
    kernel_signature = [x for x in flat_signature if x != "constexpr"]

    return triton.runtime.driver.active.utils.build_signature_metadata((kernel_signature, ))


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
        args_dict['spv_name'] = f"{arg.name}.spv"
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
        launcher = triton.runtime.driver.active.utils.launch
        expanded_signature = expand_signature(signature.values(), tensordesc_meta=None, descriptor_type="*i8")
        self.arg_annotations = annotate_arguments(expanded_signature)
        self.kernel_signature = make_kernel_signature(expanded_signature)
        self.launch = wrap_handle_tensordesc(launcher, signature, tensordesc_meta=[])

        # Serialize KernelArguments for SPIR-V Runner
        self.serialize_kernel_args = knobs.intel.enable_dump_spirv_kernel_args
        self.cache_dir = metadata.cache_dir
        self.dump_dir = self._resolve_dump_dir(metadata.cache_dir)
        self.print_dump_spirv_kernel_args_info = knobs.intel.print_dump_spirv_kernel_args_info
        self.constants = constants
        self.signature = signature

    def _resolve_dump_dir(self, cache_dir):
        dump_dir_root = knobs.intel.dump_spirv_kernel_args_dir
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
        if self.serialize_kernel_args:
            if self.print_dump_spirv_kernel_args_info:
                print(
                    f"Triton kernel dump info: kernel_name={kernel_metadata.name}, cache_dir={self.cache_dir}, dump_dir={self.dump_dir}"
                )
            serialize_args((gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                            launch_exit_hook, *args), self.constants, self.signature, self.dump_dir)

        if os.environ.get("TRITON_DUMP_LAUNCH_PARAMS") == "1":
            # This function does not cover all cases, for example when the arguments are tuple,
            # but it is sufficient for llama 3.1 kernels
            self._dump_launch_params((gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata,
                                      launch_enter_hook, launch_exit_hook, *args), self.constants, self.signature)

        self.launch(gridX, gridY, gridZ, stream, function, kernel_metadata, launch_metadata, launch_enter_hook,
                    launch_exit_hook, self.arg_annotations, self.kernel_signature, args)


class XPUDriver(DriverBase):

    def __init__(self):
        self.launcher_cls = XPULauncher
        super().__init__()

    def __getattr__(self, name):
        # Lazily initialize utils to avoid unnecessary XPU runtime invocations.
        # See https://github.com/intel/intel-xpu-backend-for-triton/issues/624
        if name == "utils":
            self.utils = XPUUtils()
            return self.utils
        else:
            raise AttributeError

    def get_current_device(self):
        return self.utils.get_current_device()

    def get_current_stream(self, device):
        try:
            from torch._C import _xpu_getCurrentRawStream
            return _xpu_getCurrentRawStream(device)
        except ImportError:
            import torch
            return torch.xpu.current_stream().sycl_queue

    @lru_cache
    def _construct_target(self, device):
        from triton.backends.intel.extension_utils import query_device_extensions

        dev_property = self.utils._get_device_capability(device)

        def update_device_arch(dev_property):
            if not (arch := knobs.intel.device_arch):
                arch = get_arch_utils_module().parse_device_arch(dev_property["architecture"])
            dev_property["arch"] = arch

        # All GPUs with the same device_id have the same extensions, so we just
        # need to query any GPU device
        device_id = dev_property.get("device_id")
        extensions = query_device_extensions(device_id)
        dev_property.update(extensions)
        dev_property["__intel_already_queried_extensions__"] = True
        update_device_arch(dev_property)

        return GPUTarget("xpu", dev_property, warp_size=32)

    def get_current_target(self):
        device = self.get_current_device()
        return self._construct_target(device)

    def build_proton_help_lib(self):
        from triton.backends.intel.driver import compile_module_from_src

        dirname = os.path.dirname(os.path.realpath(__file__))
        return compile_module_from_src(src=Path(dirname).joinpath("proton_utils.cpp").read_text(), name="proton_utils")

    def get_active_torch_device(self):
        import torch
        return torch.device("xpu", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.xpu

    @staticmethod
    def is_active():
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
