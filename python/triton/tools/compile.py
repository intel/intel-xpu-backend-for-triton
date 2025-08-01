import binascii
import hashlib
import importlib.util
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List

import triton
from triton._internal_testing import is_xpu
import triton.backends


@dataclass
class CompileArgs:
    '''
    A class to contain arguments from command-line parser.
    '''
    path: str = ''
    kernel_name: str = ''
    signature: str = ''
    grid: str = ''
    grf_mode: str = ''
    generate_native_code: bool = False
    target: str | None = None
    num_warps: int = 1
    threads_per_warp: int = 32
    num_stages: int = 3
    out_name: str | None = None
    out_path: Path | None = None


desc = """
Triton ahead-of-time compiler:

This program compiles the kernel with name `kernel-name` in the file at the
provided `path` into self-contained C source-code that embeds the `cubin`
data along with utilities to load, unload and launch the kernel.

signature is provided as a list of (optionally divisibility-hinted) types
or constexpr values, e.g.

`compile.py --kernel-name kernel --signature "*fp32:16, i32:16, 1024, i32" --out-name kernel /path/to/kernel.py`

will compile triton.JITFunction of name `kernel` inside the file `/path/to/kernel.py`.
Said kernel will be specialized such that argument 0, 1 are assumed to be multiple of 16,
and argument 2 is assumed to be a compile-time constant of value 1024, i.e. it won't be part of the generated prototype.

The resulting entry point will have signature

CUresult kernel_{specialization_suffix}(CUstream stream, unsigned gX, unsigned gY, unsigned gZ, float* arg0, int32_t arg1, int32_t arg2)

Different such specialized entry points can be combined using the `linker.py` script.

NOTE: when resolving the scope of /path/to/kernel.py, the file will be executed from within its parent directory with the python interpreter
used to run this `compile.py` script
"""


def main():
    # command-line arguments
    parser = ArgumentParser(description=desc)
    parser.add_argument("path",
                        help="Path to Python source containing desired kernel in its scope. File will be executed.")
    parser.add_argument("--kernel-name", "-n", type=str, default="", help="Name of the kernel to compile",
                        required=True)
    parser.add_argument(
        "--target", "-t", type=str, default=None,
        help="The target to compile towards, in format of '<backend>:<arch>:<warp-size>'; "
        "e.g., 'cuda:80:32', 'hip:gfx942:64'. Default to None, which means using current machine's GPU target")
    parser.add_argument("--num-warps", "-w", type=int, default=1, help="Number of warps to launch the kernel")
    parser.add_argument("--threads-per-warp", "-tpw", type=int, default=32, help="Number of theads per warp")
    parser.add_argument("--num-stages", "-ns", type=int, default=3,
                        help="Number of stages (meta-parameter of the kernel)")
    parser.add_argument("--out-name", "-on", type=str, default=None, help="Out name for the compiled kernel")
    parser.add_argument("--out-path", "-o", type=Path, default=None, help="Out filename")
    parser.add_argument("--signature", "-s", type=str, help="Signature of the kernel", required=True)
    parser.add_argument("--grid", "-g", type=str, help="Launch grid of the kernel", required=True)
    parser.add_argument("--grf-mode", "-gm", type=str, default="large", help="Detemine spv build flags")
    parser.add_argument("--generate-native-code", "-gnc", action="store_true",
                        help="Generate native binary instead of SPV for XPU")
    cli_args = parser.parse_args()
    args = CompileArgs(**vars(cli_args))  # A sanity check to ensure class CompileArgs is updated as well.
    compile_kernel(args)


def compile_kernel(args: CompileArgs):
    out_name = args.out_name if args.out_name else args.kernel_name
    out_path = args.out_path if args.out_path else Path(out_name)

    # execute python sources and extract functions wrapped in JITFunction
    arg_path = Path(args.path)
    sys.path.insert(0, str(arg_path.parent))
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, args.kernel_name)
    grid = args.grid.split(",")
    assert len(grid) == 3

    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), args.signature.split(",")))

    def hash_signature(signature: List[str]):
        m = hashlib.sha256()
        m.update(" ".join(signature).encode())
        return m.hexdigest()[:8]

    meta_sig = f"warps{args.num_warps}xstages{args.num_stages}"
    sig_hash = hash_signature(signature + [meta_sig])

    def constexpr(s):
        try:
            ret = int(s)
            return ret
        except ValueError:
            pass
        try:
            ret = float(s)
            return ret
        except ValueError:
            pass
        return None

    hints = {(i, ): constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constants = {kernel.arg_names[i]: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    for key, value in hints.items():
        if value == 1:
            constants[kernel.arg_names[key[0]]] = value
    signature = {kernel.arg_names[i]: s.split(":")[0] for i, s in enumerate(signature)}
    for key in constants:
        signature[key] = 'constexpr'
    const_sig = 'x'.join([str(v) for v in constants.values()])
    doc_string = [f"{k}={v}" for k, v in constants.items()]
    doc_string += [f"num_warps={args.num_warps}", f"num_stages={args.num_stages}"]
    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    attrs = {k: [["tt.divisibility", 16]] for k, v in hints.items() if v == 16}
    src = triton.compiler.ASTSource(fn=kernel, constexprs=constants, signature=signature, attrs=attrs)

    target = triton.backends.compiler.GPUTarget(*args.target.split(":")) \
        if args.target else triton.runtime.driver.active.get_current_target()
    backend = triton.compiler.make_backend(target)
    kwargs = {"num_warps": args.num_warps, "num_stages": args.num_stages}
    if is_xpu():
        kwargs = {
            "num_warps": args.num_warps, "num_stages": args.num_stages, "threads_per_warp": args.threads_per_warp,
            "grf_mode": args.grf_mode, "generate_native_code": args.generate_native_code
        }
    options = backend.parse_options(kwargs)
    ccinfo = triton.compile(src, target=target, options=options.__dict__)
    args.threads_per_warp = ccinfo.metadata.threads_per_warp

    if getattr(ccinfo.metadata, "global_scratch_size", 0) > 0:
        raise RuntimeError("AOT compiling kernels with global scratch requirements is not yet implemented")

    arg_names = []
    arg_types = []
    arg_names_not_1 = []
    arg_types_not_1 = []
    for i, arg_name in enumerate(kernel.arg_names):
        if arg_name not in constants:
            arg_names.append(arg_name)
            arg_types.append(signature[arg_name])
            arg_names_not_1.append(arg_name)
            arg_types_not_1.append(signature[arg_name])
        elif hints.get((i, ), None) == 1:
            arg_names.append(arg_name)
            arg_types.append("i32")

    # dump C stub code
    suffix = ''
    for i, ty in enumerate(signature.values()):
        suffix += str(i)
        if hints.get((i, ), None) == 1:
            suffix += 'c'
        if hints.get((i, ), None) == 16:
            suffix += 'd'
    func_name = '_'.join([out_name, sig_hash, suffix])
    asm = ccinfo.asm[backend.binary_ext]  # store binary data once

    hex_ = str(binascii.hexlify(asm))[2:-1]

    ty_to_cpp = triton.runtime.driver.active.map_python_to_cpp_type

    params = {
        "kernel_name": func_name,
        "triton_kernel_name": args.kernel_name,
        "bin_size": len(asm),
        "bin_data": ", ".join([f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])]),
        "signature": ", ".join([f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names_not_1, arg_types_not_1)]),
        "full_signature": ", ".join([f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names, arg_types)]),
        "arg_pointers": ", ".join([f"&{arg}" for arg in arg_names_not_1] + ["&global_scratch"]),
        "num_args": len(arg_names_not_1) + 1,
        "kernel_docstring": doc_string,
        "shared": ccinfo.metadata.shared,
        "num_warps": args.num_warps,
        "algo_info": '_'.join([const_sig, meta_sig]),
        "gridX": grid[0],
        "gridY": grid[1],
        "gridZ": grid[2],
        "_placeholder": "",
    }
    if is_xpu():
        if args.generate_native_code:
            format_name = "native"
        else:
            format_name = "spirv"
        params |= {
            "arg_types": ", ".join(ty_to_cpp(arg) for arg in arg_types_not_1),
            "grf_mode": args.grf_mode,
            "build_flags": ccinfo.metadata.build_flags,
            "threads_per_warp": args.threads_per_warp,
            "format_name": format_name,
        }
    output_files = []
    backend_name = target.backend
    if is_xpu():
        # instead of "xpu"
        backend_name = "intel"
    template_dir = Path(__file__).parent / "extra" / backend_name
    for template_path in template_dir.glob('compile.*'):
        ext = template_path.suffix
        output_file = out_path.with_suffix(f".{sig_hash}_{suffix}{ext}")
        with output_file.open("w") as fp:
            fp.write(template_path.read_text().format(**params))
        output_files.append(output_file)

    return func_name, output_files


if __name__ == "__main__":
    main()
