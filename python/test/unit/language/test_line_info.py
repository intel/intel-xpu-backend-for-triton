import subprocess
import tempfile

import pytest
import torch
import intel_extension_for_pytorch  # type: ignore # noqa: F401

import triton
import triton.language as tl


@triton.jit
def kernel_single(X,
                  Y,
                  BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def device_inline(x):
    return x + x


@triton.jit
def kernel_call(X,
                Y,
                BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = device_inline(x)
    tl.store(Y + tl.arange(0, BLOCK), y)


@triton.jit(noinline=True)
def device_noinline(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = x + x
    tl.store(Y + tl.arange(0, BLOCK), y)


@triton.jit
def kernel_call_noinline(X, Y, BLOCK: tl.constexpr):
    device_noinline(X, Y, BLOCK)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4),
    ],
    key=[],
)
@triton.jit
def kernel_autotune(X, Y, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    for i in range(0, SIZE, BLOCK):
        x = tl.load(X + i + tl.arange(0, BLOCK))
        tl.store(Y + i + tl.arange(0, BLOCK), x)


# AddIOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)
# Since the + symbol will take effect in the dot op after combination,
# it seems making sense to annotate with the same line as dot.
@triton.jit
def kernel_dot_combine(x):
    c = tl.full((32, 32), 4, dtype=tl.int8)
    a = (tl.arange(0, 32)[:, None] + tl.arange(0, 32)[None, :]).to(tl.int8)
    d = tl.dot(a, a)
    d = d + c
    tl.device_print("", d)


def spv_extract_file_lines(spv, command):
    fd, path = tempfile.mkstemp()
    with open(fd, 'wb') as spvbin:
        spvbin.write(spv)
    spv = subprocess.check_output(command + [path]).decode("utf-8")
    lines = spv.splitlines()

    # Collect string variables (pairs of [varname, string]). One should contain the file name.
    id_and_strings = []
    for line in lines:
        if "OpString" not in line:
            continue
        entries = line[line.index("%"):].split(" ")
        id_and_strings.append((entries[0].strip(), entries[3].strip()))

    # Collect pairs of [fileName, lineNo].
    file_and_lines = []
    for line in lines:
        if "OpLine" not in line:
            continue
        entries = line[line.index("OpLine"):].split(" ")
        var, lineNo = (entries[1].strip(), entries[2].strip())
        for id, string in id_and_strings:
            if var == id:
                file_and_lines.append((string, lineNo))
                break

    return file_and_lines

def get_disassembler_command_and_debug_line_format():
    """Gets backend specific disassembler information.

    Returns a tuple: (object file kind, disassembler tool command,
    debug line anchor, debug line file and line number separator).
    """
    backend = triton.runtime.driver.active.get_current_target().backend

    if backend == "cuda":
        from triton.backends.nvidia.compiler import _path_to_binary
        nvdisasm, _ = _path_to_binary("nvdisasm")
        return ("cubin", [nvdisasm, "-g"], "## File", ",")

    if backend == "hip":
        import shutil
        # Try to find llvm-objdump from the current PATH to disassmble hsaco.
        tool = shutil.which("llvm-objdump")
        if tool is not None:
            return ("hsaco", [tool, "-D", "-l", "--arch=amdgcn"], ";", ":")
        raise RuntimeError("llvm-objdump not found in PATH")

    if backend == "xpu":
        from triton.backends.intel.compiler import _path_to_binary
        dis, _ = _path_to_binary("spirv-dis")
        return ("spvbin", [dis], "", "")

    raise RuntimeError(f"unknown backend {backend}")


def extract_file_lines(command, anchor, separator, asm):
    fd, path = tempfile.mkstemp()
    with open(fd, 'wb') as cubin:
        cubin.write(asm)
    asm = subprocess.check_output(command + [path]).decode("utf-8")
    file_lines = []
    lines = asm.splitlines()
    for line in lines:
        # We are looking for an anchor string and a separator between the file name and line number.
        if anchor in line and separator in line:
            entries = line[line.index(anchor):].split(separator)
            if len(entries) == 2 and all(len(e) != 0 for e in entries):
                file_lines.append((entries[0].strip(), entries[1].strip()))
    return file_lines


def check_file_lines(file_lines, file_name, lineno, should_contain=True):
    """
    Check if the file name and line number is in the file_lines

    Args:
        file_lines: list of (file_name, line_number)
        file_name: file name
        lineno: line number, -1 means do not check line number
        should_contain: whether the file name and line number should be in the file_lines
    """
    for file, line in file_lines:
        if lineno == -1:
            if file_name in file:
                return True
        if file_name in file and str(lineno) in line:
            return should_contain
    return not should_contain


func_types = ["single", "call", "call_noinline", "autotune", "dot_combine"]


@pytest.mark.parametrize("func", func_types)
def test_line_info(func: str):
    try:
        obj_kind, command, anchor, separator = get_disassembler_command_and_debug_line_format()
    except BaseException:
        pytest.skip("disassembler is not available")

    shape = (128, )
    kernel_info = {}
    if func == "single":
        kernel_info = kernel_single.warmup(torch.float32, torch.float32, BLOCK=shape[0], grid=(1,))
    elif func == "call":
        kernel_info = kernel_call.warmup(torch.float32, torch.float32, BLOCK=shape[0], grid=(1,))
    elif func == "call_noinline":
        kernel_info = kernel_call_noinline.warmup(torch.float32, torch.float32, BLOCK=shape[0], grid=(1,))
    elif func == "autotune":
        kernel_info = kernel_autotune.warmup(torch.float32, torch.float32, SIZE=shape[0], grid=(1,))[0]
    elif func == "dot_combine":
        kernel_info = kernel_dot_combine.warmup(20, grid=(1,))

    if obj_kind == "spvbin":
        file_lines = spv_extract_file_lines(kernel_info.asm["spv"], command)
    else:
        file_lines = extract_file_lines(command, anchor, separator, kernel_info.asm[obj_kind])
    if func == "single":
        assert (check_file_lines(file_lines, "test_line_info.py", 16))
        assert (check_file_lines(file_lines, "test_line_info.py", 17))
    elif func == "call":
        assert (check_file_lines(file_lines, "test_line_info.py", 29))
        assert (check_file_lines(file_lines, "test_line_info.py", 22))
        assert (check_file_lines(file_lines, "test_line_info.py", 31))
    elif func == "call_noinline":
        assert (check_file_lines(file_lines, "test_line_info.py", 43))
        assert (check_file_lines(file_lines, "test_line_info.py", 36))
        assert (check_file_lines(file_lines, "test_line_info.py", 37))
        assert (check_file_lines(file_lines, "test_line_info.py", 38))
    elif func == "autotune":
        assert (check_file_lines(file_lines, "test_line_info.py", 55))
        assert (check_file_lines(file_lines, "test_line_info.py", 56))
        assert (check_file_lines(file_lines, "test_line_info.py", 54))
    elif func == "dot_combine":
        assert (check_file_lines(file_lines, "test_line_info.py", 66))
        assert (check_file_lines(file_lines, "test_line_info.py", 67, should_contain=False))
