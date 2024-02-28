import importlib.util
import itertools
import os
import shutil
import tempfile

import pytest
import torch
import intel_extension_for_pytorch  # type: ignore # noqa: F401

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction

tmpdir = ".tmp"


@triton.jit
def function_1(i):
    i = i + 1
    i = function_2(i)
    return i


@triton.jit
def function_2(i):
    i = i + 1
    return i


@triton.jit
def combine_fn(a, b):
    return COMBINE_OP  # noqa: F821


@triton.jit
def kernel(X, i, BLOCK: tl.constexpr):
    i = i + 1
    i = function_1(i)
    tl.store(X, i)


@triton.jit(do_not_specialize=["i"])
def kernel_nospec(X, i, BLOCK: tl.constexpr):
    i = i + 1
    i = function_1(i)
    tl.store(X, i)


@triton.jit
def kernel_with_combine_fn(X, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    i = REDUCE_OR_SCAN(i, 0, combine_fn)  # noqa: F821
    tl.store(X, i)


def apply_src_change(target, old, new):
    kernel.hash = None
    function_1.hash = None
    function_2.hash = None
    function_1.src = function_1.src.replace(old, new)
    target.src = target.src.replace(old, new)
    ret = target.cache_key
    target.src = target.src.replace(new, old)
    return ret


def test_nochange():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 1')
    assert baseline == updated


def test_toplevel_change():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 2')
    assert baseline != updated


def test_nested1_change():
    baseline = kernel.cache_key
    updated = apply_src_change(function_1, 'i + 1', 'i + 2')
    assert baseline != updated


def test_combine_fn_change():
    # Test that tl.reduce and associative_scan calls include
    # the combine_fn in the hash

    orig_combine_fn_src = combine_fn.src
    orig_kernel_src = kernel_with_combine_fn.src
    seen_keys = set()

    for reduce_or_scan, combine_op in itertools.product(
        ["tl.reduce", "tl.associative_scan"],
        ["a + b", "a * b"],
    ):
        combine_fn.src = orig_combine_fn_src.replace("COMBINE_OP", combine_op)
        kernel_with_combine_fn.src = orig_kernel_src.replace("REDUCE_OR_SCAN", reduce_or_scan)
        try:
            key = kernel_with_combine_fn.cache_key
        finally:
            combine_fn.src = orig_combine_fn_src
            kernel_with_combine_fn.src = orig_kernel_src

            kernel_with_combine_fn.hash = None
            combine_fn.hash = None

        assert key not in seen_keys
        seen_keys.add(key)


def write_and_load_module(code, num_extra_lines):
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.py') as f:
        f.write(('# extra line\n' * num_extra_lines) + code)
        f.flush()
        spec = importlib.util.spec_from_file_location("module.name", f.name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def test_changed_line_numbers_invalidate_cache():
    from textwrap import dedent
    code = dedent("""
        import triton
        @triton.jit
        def test_kernel(i):
            i = i + 1
    """)
    orig_mod = write_and_load_module(code, 0)
    orig_cache_key = orig_mod.test_kernel.cache_key

    updated_mod = write_and_load_module(code, 1)
    updated_cache_key = updated_mod.test_kernel.cache_key
    assert orig_cache_key != updated_cache_key


def reset_tmp_dir():
    os.environ["TRITON_CACHE_DIR"] = tmpdir
    if os.path.exists(tmpdir):
        # https://stackoverflow.com/questions/303200/how-do-i-remove-delete-a-folder-that-is-not-empty
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_reuse():
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    JITFunction.cache_hook = inc_counter
    reset_tmp_dir()
    x = torch.empty(1, dtype=torch.int32, device='xpu')
    for i in range(10):
        kernel[(1, )](x, 1, BLOCK=1024)
    assert counter == 1


@pytest.mark.parametrize('mode', ['enable', 'disable'])
def test_specialize(mode):
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    JITFunction.cache_hook = inc_counter
    reset_tmp_dir()
    x = torch.empty(1, dtype=torch.int32, device='xpu')
    function = {'enable': kernel, 'disable': kernel_nospec}[mode]
    target = {'enable': 3, 'disable': 1}[mode]
    for i in [1, 2, 4, 8, 16, 32]:
        function[(1, )](x, i, BLOCK=512)
    assert counter == target


def test_annotation():

    @triton.jit
    def kernel(X, i: tl.int32):
        tl.store(X, i)

    x = torch.empty(1, dtype=torch.int32, device='xpu')

    device = torch.xpu.current_device()
    kernel[(1, )](x, 1)
    kernel[(1, )](x, 8)
    kernel[(1, )](x, 16)
    kernel[(1, )](x, 17)
    assert len(kernel.cache[device]) == 3


def test_constexpr_not_callable() -> None:

    @triton.jit
    def kernel(X, c: tl.constexpr):
        tl.store(X, 2)

    x = torch.empty(1, dtype=torch.int32, device='xpu')
    error = False
    try:
        kernel[(1, )](x, c="str")
    except BaseException:
        error = True
    assert error is False
    # try and catch
    try:
        kernel[(1, )](x, c=tl.abs)
    except BaseException:
        error = True
    assert error is True


def test_jit_warmup_cache() -> None:

    @triton.jit
    def kernel_add(a, b, o, N: tl.constexpr):
        idx = tl.arange(0, N)
        tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))

    args = [
        torch.randn(32, dtype=torch.float32, device="xpu"),
        torch.randn(32, dtype=torch.float32, device="xpu"),
        torch.randn(32, dtype=torch.float32, device="xpu"),
        32,
    ]
    device = torch.xpu.current_device()
    assert len(kernel_add.cache[device]) == 0
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add.cache[device]) == 1
    kernel_add.warmup(*args, grid=(1, ))
    assert len(kernel_add.cache[device]) == 1
    kernel_add.warmup(*args, grid=(1, ))
    assert len(kernel_add.cache[device]) == 1


def test_jit_debug() -> None:

    @triton.jit
    def kernel_add(a, b, o, N: tl.constexpr):
        idx = tl.arange(0, N)
        tl.device_assert(idx < 32, "idx < 32")
        tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))

    device = torch.xpu.current_device()
    assert len(kernel_add.cache[device]) == 0
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add.cache[device]) == 1
    kernel_add.debug = False
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add.cache[device]) == 2
    kernel_add.debug = True
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add.cache[device]) == 3
    bins = list(kernel_add.cache[device].values())
    assert bins[2].asm['ttir'] != bins[1].asm['ttir']


@triton.jit
def add_fn(a, b, o, N: tl.constexpr):
    idx = tl.arange(0, N)
    tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))


def test_jit_noinline() -> None:

    @triton.jit
    def kernel_add_device(a, b, o, N: tl.constexpr):
        add_fn(a, b, o, N)

    device = torch.xpu.current_device()
    assert len(kernel_add_device.cache[device]) == 0
    kernel_add_device.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add_device.cache[device]) == 1
    bins = list(kernel_add_device.cache[device].values())
    inline_ttir = bins[0].asm['ttir']
    add_fn.noinline = True
    add_fn.hash = None
    kernel_add_device.hash = None
    kernel_add_device.cache[device].clear()
    kernel_add_device.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add_device.cache[device]) == 1
    bins = list(kernel_add_device.cache[device].values())
    noinline_ttir = bins[0].asm['ttir']
    assert inline_ttir != noinline_ttir


def test_memory_leak() -> None:

    @triton.jit
    def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
        xnumel = 10
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)


def test_preload() -> None:

    @triton.jit
    def kernel_add(a, b, o, N: tl.constexpr, type: tl.constexpr):
        idx = tl.arange(0, N)
        tl.device_assert(idx < 32, "idx < 32")
        tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))

    device = torch.xpu.current_device()

    # get the serialized specialization data
    specialization_data = None

    def cache_hook(*args, **kwargs):
        nonlocal specialization_data
        specialization_data = kwargs["compile"]["specialization_data"]

    JITFunction.cache_hook = cache_hook
    pre_compile = kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, tl.float32, grid=(1, ))
    hash = pre_compile.hash
    assert specialization_data is not None

    # clear the cache
    reset_tmp_dir()
    kernel_add.cache[device].clear()

    # preload the kernel
    kernel_preload = kernel_add.preload(specialization_data)
    assert kernel_preload.hash == hash
    assert len(kernel_add.cache[device]) == 1

    # we should hit the cache and not compile anything
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    JITFunction.cache_hook = inc_counter
    final_kernel = kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, tl.float32, grid=(1, ))
    JITFunction.cache_hook = None
    assert counter == 0
    assert len(kernel_add.cache[device]) == 1
    assert final_kernel.hash == hash
