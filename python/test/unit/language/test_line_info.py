import subprocess
import tempfile

import pytest
import torch
import intel_extension_for_pytorch  # type: ignore # noqa: F401

import triton
import triton.language as tl
from triton.backends.intel.compiler import _path_to_binary


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


def extract_file_lines(spv):
    dis, _ = _path_to_binary("spirv-dis")
    fd, path = tempfile.mkstemp()
    with open(fd, 'wb') as spvbin:
        spvbin.write(spv)
    spv = subprocess.check_output([dis, path]).decode("utf-8")
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
        _, _ = _path_to_binary("spirv-dis")
    except BaseException:
        pytest.skip("spirv-dis is not available")

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

    file_lines = extract_file_lines(kernel_info.asm["spv"])

    if func == "single":
        assert (check_file_lines(file_lines, "test_line_info.py", 17))
        assert (check_file_lines(file_lines, "test_line_info.py", 18))
    elif func == "call":
        assert (check_file_lines(file_lines, "test_line_info.py", 30))
        assert (check_file_lines(file_lines, "test_line_info.py", 23))
        assert (check_file_lines(file_lines, "test_line_info.py", 32))
    elif func == "call_noinline":
        assert (check_file_lines(file_lines, "test_line_info.py", 44))
        assert (check_file_lines(file_lines, "test_line_info.py", 37))
        assert (check_file_lines(file_lines, "test_line_info.py", 38))
        assert (check_file_lines(file_lines, "test_line_info.py", 39))
    elif func == "autotune":
        assert (check_file_lines(file_lines, "test_line_info.py", 56))
        assert (check_file_lines(file_lines, "test_line_info.py", 57))
        assert (check_file_lines(file_lines, "test_line_info.py", 55))
    elif func == "dot_combine":
        assert (check_file_lines(file_lines, "test_line_info.py", 66))
        assert (check_file_lines(file_lines, "test_line_info.py", 68, should_contain=False))
