import copy
import errno
import os
import shutil
import sys
from typing import List

import setuptools  # noqa: F401
import torch
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import _TORCH_PATH

IS_WINDOWS = sys.platform == "win32"
SHARED_FLAG = "/DLL" if IS_WINDOWS else "-shared"
SYCL_FLAG = "-fsycl"

COMMON_DPCPP_FLAGS = ["-fPIC"]

TORCH_LIB_PATH = os.path.join(_TORCH_PATH, "lib")


def use_profile():
    if os.getenv("TRITON_XPU_PROFILE") is not None and (os.getenv("TRITON_XPU_PROFILE").lower() == 'on' or os.getenv("TRITON_XPU_PROFILE").lower() == '1'):
        return True
    else:
        return False


def get_dpcpp_complier():
    # build cxx via dpcpp
    dpcpp_cmp = shutil.which("icpx")
    if dpcpp_cmp is None:
        raise RuntimeError("Failed to find compiler path from OS PATH")
    _cxxbin = os.getenv("CXX")
    if _cxxbin is not None:
        dpcpp_cmp = _cxxbin
    return dpcpp_cmp


def get_icx_complier():
    # build cc via icx
    icx_cmp = shutil.which("icx")
    if icx_cmp is None:
        raise RuntimeError("Failed to find compiler path from OS PATH")
    return icx_cmp


def _is_cpp_file(path: str) -> bool:
    valid_ext = [".cpp", ".hpp"]
    return os.path.splitext(path)[1] in valid_ext


def _is_c_file(path: str) -> bool:
    valid_ext = [".c", ".h"]
    return os.path.splitext(path)[1] in valid_ext


class SYCLBuildExtension(build_ext, object):
    r"""
    A custom :mod:`setuptools` build extension .
    This class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++17``) for icpx compilation.

    When using :class:`SYCLBuildExtension`, it is allowed to supply a
    dictionary for ``extra_compile_args`` (rather than the usual list) that
    maps from languages (``cxx``) to a list of additional compiler flags to
    supply to the compiler.

    ``no_python_abi_suffix`` (bool): If ``no_python_abi_suffix`` is
    ``False`` (default), then we attempt to build module with python abi
    suffix, example: output module name:
    module_name.cpython-37m-x86_64-linux-gnu.so
    , the ``cpython-37m-x86_64-linux-gnu`` is append python abi suffix.
    """

    @classmethod
    def with_options(cls, **options):
        r"""
        Returns a subclass with alternative constructor that extends original
        keyword arguments to the original constructor with the given options.
        """

        class cls_with_options(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        super(SYCLBuildExtension, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

    def finalize_options(self) -> None:
        super().finalize_options()

    def build_extensions(self) -> None:
        dpcpp_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not dpcpp_ext and extension:
            extension = next(extension_iter, None)

        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx' when
            # extra_compile_args is a dict. Otherwise, default torch
            # flags do not get passed. Necessary when only one of 'cxx' is
            # passed to extra_compile_args in SYCLExtension, i.e.
            #   SYCLExtension(..., extra_compile_args={'cxx': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ["cxx"]:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            if use_profile():
                self._add_compile_flag(extension, "-DTORCH_API_INCLUDE_EXTENSION_H")
                # See note [Pybind11 ABI constants]
                for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
                    val = getattr(torch._C, f"_PYBIND11_{name}")
                    if val is not None and not IS_WINDOWS:
                        self._add_compile_flag(extension, f'-DPYBIND11_{name}="{val}"')

            self._add_gnu_cpp_abi_flag(extension)

        # Save the original _compile method for later.
        if self.compiler.compiler_type == "msvc":
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile
            original_spawn = self.compiler.spawn

        def append_std17_if_no_std_present(cflags) -> None:
            cpp_format_prefix = (
                "/{}:" if self.compiler.compiler_type == "msvc" else "-{}="
            )
            cpp_flag_prefix = cpp_format_prefix.format("std")
            cpp_flag = cpp_flag_prefix + "c++17"
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_dpcpp_flags(cflags):
            cflags = COMMON_DPCPP_FLAGS + cflags
            return cflags

        def unix_wrap_single_compile(
            obj, src, ext, cc_args, extra_postargs, pp_opts
        ) -> None:
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cpp_file(src):
                    _cxxbin = get_dpcpp_complier()
                    self.compiler.set_executable("compiler_so", _cxxbin)
                    if isinstance(cflags, dict):
                        cflags = cflags['cxx'] + COMMON_DPCPP_FLAGS
                    else:
                        cflags = unix_dpcpp_flags(cflags)
                elif _is_c_file(src):
                    _ccbin = get_icx_complier()
                    self.compiler.set_executable("compiler_so", _ccbin)
                    if isinstance(cflags, dict):
                        cflags = cflags['cxx'] + COMMON_DPCPP_FLAGS
                    else:
                        cflags = unix_dpcpp_flags(cflags)
                elif isinstance(cflags, dict):
                    cflags = cflags["cxx"]
                append_std17_if_no_std_present(cflags)

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable("compiler_so", original_compiler)

        def _gen_link_lib_cmd_line(
            linker,
            objects,
            target_name,
            library_dirs,
            runtime_library_dirs,
            libraries,
            extra_postargs,
        ):
            cmd_line = []

            library_dirs_args = []
            library_dirs_args += [f"-L{x}" for x in library_dirs]

            runtime_library_dirs_args = []
            runtime_library_dirs_args += [f"-L{x}"
                                          for x in runtime_library_dirs]

            libraries_args = []
            libraries_args += [f"-l{x}" for x in libraries]
            common_args = [SHARED_FLAG] + [SYCL_FLAG]

            """
            link command formats:
            cmd = [LD common_args objects library_dirs_args
            runtime_library_dirs_args libraries_args -o
            target_name extra_postargs]
            """

            cmd_line += [linker]
            cmd_line += common_args
            cmd_line += objects
            cmd_line += library_dirs_args
            cmd_line += runtime_library_dirs_args
            cmd_line += libraries_args
            cmd_line += ["-o"]
            cmd_line += [target_name]
            cmd_line += extra_postargs

            return cmd_line

        def create_parent_dirs_by_path(filename):
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

        def unix_wrap_single_link_shared_object(
            objects,
            output_libname,
            output_dir=None,
            libraries=None,
            library_dirs=None,
            runtime_library_dirs=None,
            export_symbols=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            build_temp=None,
            target_lang=None,
        ):
            # create output directories avoid linker error.
            create_parent_dirs_by_path(output_libname)

            _cxxbin = get_dpcpp_complier()
            cmd = _gen_link_lib_cmd_line(
                _cxxbin,
                objects,
                output_libname,
                library_dirs,
                runtime_library_dirs,
                libraries,
                extra_postargs,
            )

            return original_spawn(cmd)

        if self.compiler.compiler_type == "msvc":
            raise "Not implemented"
        else:
            self.compiler._compile = unix_wrap_single_compile
            self.compiler.link_shared_object = unix_wrap_single_link_shared_object  # noqa: E501

        build_ext.build_extensions(self)

    def _add_compile_flag(self, extension, flag):
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)  # noqa: E501
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    def _add_gnu_cpp_abi_flag(self, extension):
        self._add_compile_flag(
            extension,
            "-D_GLIBCXX_USE_CXX11_ABI=1"
        )


def _prepare_compile_flags(extra_compile_args):
    if isinstance(extra_compile_args, List):
        extra_compile_args.append(SYCL_FLAG)
    elif isinstance(extra_compile_args, dict):
        cl_flags = extra_compile_args.get("cxx", [])
        cl_flags.append(SYCL_FLAG)
        extra_compile_args["cxx"] = cl_flags

    return extra_compile_args


def get_pytorch_lib_dir():
    return [os.path.join(_TORCH_PATH, "lib")]


def library_paths() -> List[str]:
    r"""
    Get the lib paths include PyTorch lib, IPEX lib and oneDNN lib.

    Returns:
        A list of lib path strings.
    """
    paths = []
    paths += get_pytorch_lib_dir()
    paths += get_one_api_help().get_library_dirs()

    return paths


def _prepare_ldflags(extra_ldflags, verbose, is_standalone):

    oneapi_link_args = ["-lsycl", ]

    if use_profile():
        if IS_WINDOWS:
            python_path = os.path.dirname(sys.executable)
            python_lib_path = os.path.join(python_path, "libs")

            extra_ldflags.append('c10.lib')
            extra_ldflags.append('torch_cpu.lib')
            extra_ldflags.append('torch.lib')
            if not is_standalone:
                extra_ldflags.append("torch_python.lib")
                extra_ldflags.append(f"/LIBPATH:{python_lib_path}")

        else:
            extra_ldflags.append('-lc10')
            extra_ldflags.append('-ltorch_cpu')
            extra_ldflags.append('-ltorch')
            if not is_standalone:
                extra_ldflags.append("-ltorch_python")

            if is_standalone and "TBB" in torch.__config__.parallel_info():
                extra_ldflags.append("-ltbb")

            if is_standalone:
                extra_ldflags.append(f"-Wl,-rpath,{TORCH_LIB_PATH}")

        library_dirs = library_paths()
        oneapi_link_args += [f"-L{x}" for x in library_dirs]
        oneapi_link_args += ["-Wl", "-ldnnl", "-lOpenCL", "-lpthread", "-lm", "-ldl"]
        oneapi_link_args += ['-lintel-ext-pt-gpu']

    extra_ldflags += oneapi_link_args
    return extra_ldflags


def _get_dpcpp_root():
    # TODO: Need to decouple with toolchain env scripts
    dpcpp_root = os.getenv("CMPLR_ROOT")
    return dpcpp_root


class _one_api_help:
    __dpcpp_root = None
    __ipex_root = None

    def __init__(self):
        import intel_extension_for_pytorch
        self.__dpcpp_root = _get_dpcpp_root()
        self.__ipex_root = os.path.dirname(intel_extension_for_pytorch.__file__)
        self.check_dpcpp_cfg()

    def check_dpcpp_cfg(self):
        if self.__dpcpp_root is None:
            raise "Didn't detect dpcpp root. Please source <oneapi_dir>/compiler/<version>/env/vars.sh "

    def get_ipex_lib_dir(self):
        return [os.path.join(self.__ipex_root, "lib")]

    def get_dpcpp_include_dir(self):
        return [
            os.path.join(self.__dpcpp_root, "linux", "include"),
            os.path.join(self.__dpcpp_root, "linux", "include", "sycl"),
        ]

    def get_library_dirs(self):
        library_dirs = []
        library_dirs += [f"{x}" for x in self.get_ipex_lib_dir()]
        return library_dirs

    def get_include_dirs(self):
        include_dirs = []
        include_dirs += [f"{x}" for x in self.get_dpcpp_include_dir()]
        return include_dirs


def get_pytorch_ipex_onemkl_include_dir():
    import intel_extension_for_pytorch
    paths = intel_extension_for_pytorch.xpu.cpp_extension.include_paths()
    return paths


def get_one_api_help():
    oneAPI = _one_api_help()
    return oneAPI


def include_paths() -> List[str]:
    """
    Get the include paths required to build an extension.

    Returns:
        A list of include path strings.
    """
    if use_profile():
        # add pytorch include directories
        paths = []
        paths += get_pytorch_ipex_onemkl_include_dir()

        # add oneAPI include directories
        paths += get_one_api_help().get_include_dirs()
    else:
        import torch.utils.cpp_extension
        paths = torch.utils.cpp_extension.include_paths()

    return paths


def SYCLExtension(name, sources, *args, **kwargs):
    r"""
    Creates a :class:`setuptools.Extension` for SYCL
    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a SYCL
    extension.
    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.
    """

    library_dirs = kwargs.get("library_dirs", [])
    kwargs["library_dirs"] = library_dirs

    libraries = kwargs.get("libraries", [])
    kwargs["libraries"] = libraries

    include_dirs = kwargs.get("include_dirs", [])
    include_dirs += include_paths()
    kwargs["include_dirs"] = include_dirs

    kwargs["language"] = "c++"

    extra_compile_args = kwargs.get("extra_compile_args", {})
    extra_link_args = kwargs.get("extra_link_args", [])
    # add oneapi link parameters
    extra_link_args = _prepare_ldflags(extra_link_args, False, False)
    extra_compile_args = _prepare_compile_flags(extra_compile_args)

    # todo: add dpcpp parameter support.
    kwargs["extra_link_args"] = extra_link_args
    kwargs["extra_compile_args"] = extra_compile_args

    return setuptools.Extension(name, sources, *args, **kwargs)
