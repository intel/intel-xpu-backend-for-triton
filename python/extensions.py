import copy
import errno
import os
import shutil
import sys
from typing import List

import pybind11  # noqa: F401
import setuptools
from setuptools.command.build_ext import build_ext

IS_WINDOWS = sys.platform == "win32"
SHARED_FLAG = "/DLL" if IS_WINDOWS else "-shared"
SYCL_FLAG = "-fsycl"

COMMON_DPCPP_FLAGS = ["-fPIC"]


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


def _prepare_ldflags(extra_ldflags, verbose, is_standalone):
    if IS_WINDOWS:
        python_path = os.path.dirname(sys.executable)
        python_lib_path = os.path.join(python_path, "libs")

        if not is_standalone:
            extra_ldflags.append(f"/LIBPATH:{python_lib_path}")
    else:
        if is_standalone:
            extra_ldflags.append("-Wl,-rpath")

    oneapi_link_args = []
    oneapi_link_args += ["-lsycl"]
    extra_ldflags += oneapi_link_args

    return extra_ldflags


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
