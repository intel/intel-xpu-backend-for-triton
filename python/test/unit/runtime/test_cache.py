import expecttest
import importlib.util
import itertools
import multiprocessing
import os
import re
import time
import gc
import pathlib
from concurrent.futures import Executor, Future, ThreadPoolExecutor

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_hip
from triton.runtime.cache import FileCacheManager, RemoteCacheManager


def test_file_cache_manager_get_group_rejects_missing_child(fresh_knobs, tmp_path):
    fresh_knobs.cache.dir = str(tmp_path)
    manager = FileCacheManager("key")
    metadata_path = manager.put("{}", "kernel.json", binary=False)
    artifact_path = manager.put("binary", "kernel.cubin", binary=False)

    manager.put_group("kernel.json", {
        "kernel.json": metadata_path,
        "kernel.cubin": artifact_path,
    })
    assert manager.get_group("kernel.json") == {
        "kernel.json": metadata_path,
        "kernel.cubin": artifact_path,
    }

    os.remove(artifact_path)
    assert manager.get_group("kernel.json") is None


def test_remote_cache_manager_get_group_rejects_missing_child(fresh_knobs, tmp_path):

    class DictRemoteCacheBackend:
        data = {}

        def __init__(self, key):
            self.key = key

        def get(self, filenames):
            return {filename: self.data[filename] for filename in filenames if filename in self.data}

        def put(self, filename, data):
            self.data[filename] = data

    fresh_knobs.cache.dir = str(tmp_path)
    fresh_knobs.cache.remote_manager_class = DictRemoteCacheBackend
    DictRemoteCacheBackend.data = {}

    manager = RemoteCacheManager("key")
    manager.put("{}", "kernel.json", binary=False)
    manager.put(b"binary", "kernel.cubin")
    manager.put_group("kernel.json", {
        "kernel.json": "unused-local-path",
        "kernel.cubin": "unused-local-path",
    })

    group = manager.get_group("kernel.json")
    assert group is not None
    assert set(group) == {"kernel.json", "kernel.cubin"}

    del DictRemoteCacheBackend.data["kernel.cubin"]
    assert manager.get_group("kernel.json") is None


@triton.jit
def function_0(i):
    return i + 1


@triton.jit
def function_1(i):
    i = i + 1
    cond: tl.constexpr = True
    if cond:
        FN: tl.constexpr = function_2
    else:
        FN: tl.constexpr = function_0
    return FN(i)


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


@triton.jit(do_not_specialize_on_alignment=["i"])
def kernel_nospec_on_alignment(X, i, BLOCK: tl.constexpr):
    i = i + 1
    i = function_1(i)
    tl.store(X, i)


@triton.jit
def kernel_with_combine_fn(X, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    i = REDUCE_OR_SCAN(i, 0, combine_fn)  # noqa: F821
    tl.store(X, i)


def apply_src_change(target, old, new, to_modify):
    kernel.hash = None
    function_0.hash = None
    function_1.hash = None
    function_2.hash = None
    to_modify._unsafe_update_src(to_modify.src.replace(old, new))
    ret = target.cache_key
    to_modify._unsafe_update_src(to_modify.src.replace(new, old))
    return ret


def test_nochange():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 1', function_1)
    assert baseline == updated


def test_toplevel_change():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 2', function_1)
    assert baseline != updated


def test_keyword_only_default_dependency_change():

    @triton.jit
    def with_default(i, *, function_1: tl.constexpr = function_1):
        return function_1(i)

    baseline = with_default.cache_key
    with_default.hash = None
    updated = apply_src_change(with_default, 'i + 1', 'i + 2', function_1)
    assert baseline != updated


def test_nested1_change():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 2', function_2)
    assert baseline != updated


def test_nested2_change():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 2', function_0)
    assert baseline != updated


ORDER_DEPENDENT_CONSTEXPR = tl.constexpr(42)


@triton.jit
def order_dependent_inner():
    return ORDER_DEPENDENT_CONSTEXPR


@triton.jit
def order_dependent_outer():
    order_dependent_inner()


def test_cache_key_independent_of_dependency_hash_order():
    functions = (order_dependent_inner, order_dependent_outer)

    for function in functions:
        function.hash = None
        function.used_global_vals = {}
    cold_key = order_dependent_outer.cache_key

    for function in functions:
        function.hash = None
        function.used_global_vals = {}
    order_dependent_inner.cache_key
    prehashed_key = order_dependent_outer.cache_key

    assert prehashed_key == cold_key


def test_cache_key_independent_of_globals_dict_identity():

    def make_child(value):
        shared = tl.constexpr(value)

        @triton.jit
        def child():
            return shared

        return child

    child_a = make_child(1)
    child_b = make_child(2)

    @triton.jit
    def parent():
        child_a()
        child_b()

    functions = (child_a, child_b, parent)

    def key_after_prehashing(first, second):
        for function in functions:
            function.hash = None
            function.used_global_vals = {}
        first.cache_key
        second.cache_key
        return parent.cache_key

    assert key_after_prehashing(child_a, child_b) == key_after_prehashing(child_b, child_a)


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
        combine_fn._unsafe_update_src(orig_combine_fn_src.replace("COMBINE_OP", combine_op))
        kernel_with_combine_fn._unsafe_update_src(orig_kernel_src.replace("REDUCE_OR_SCAN", reduce_or_scan))
        try:
            key = kernel_with_combine_fn.cache_key
        finally:
            combine_fn._unsafe_update_src(orig_combine_fn_src)
            kernel_with_combine_fn._unsafe_update_src(orig_kernel_src)

        assert key not in seen_keys
        seen_keys.add(key)


@triton.constexpr_function
def constexpr_flag_fn():
    return False


@triton.jit
def constexpr_fn_user(out):
    a: tl.constexpr = constexpr_flag_fn()
    tl.store(out, a)


def test_constexpr_fn_change():
    baseline = constexpr_fn_user.cache_key

    orig_src = constexpr_flag_fn.src
    new_src = orig_src.replace("False", "True")
    constexpr_flag_fn._unsafe_update_src(new_src)
    constexpr_fn_user.hash = None
    updated = constexpr_fn_user.cache_key
    assert baseline != updated

    constexpr_flag_fn._unsafe_update_src(orig_src)
    constexpr_fn_user.hash = None
    assert constexpr_fn_user.cache_key == baseline


@triton.constexpr_function
def invalid_constexpr_fn():
    return torch.cuda.get_device_capability()


def test_invalid_constexpr_fn():
    with pytest.raises(RuntimeError):
        invalid_constexpr_fn.cache_key


def write_and_load_module(temp_file: pathlib.Path, code, num_extra_lines):
    temp_file.write_text(('# extra line\n' * num_extra_lines) + code)
    spec = importlib.util.spec_from_file_location("module.name", str(temp_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_changed_line_numbers_invalidate_cache(tmp_path: pathlib.Path):
    from textwrap import dedent
    code = dedent("""
        import triton
        @triton.jit
        def test_kernel(i):
            i = i + 1
    """)
    temp_file0 = tmp_path / "test_changed_line_numbers_invalidate_cache0.py"
    orig_mod = write_and_load_module(temp_file0, code, 0)
    orig_cache_key = orig_mod.test_kernel.cache_key

    temp_file1 = tmp_path / "test_changed_line_numbers_invalidate_cache1.py"
    updated_mod = write_and_load_module(temp_file1, code, 1)
    updated_cache_key = updated_mod.test_kernel.cache_key
    assert orig_cache_key != updated_cache_key


def test_reuse(device, fresh_triton_cache):
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    triton.knobs.runtime.jit_cache_hook = inc_counter
    x = torch.empty(1, dtype=torch.int32, device=device)
    for i in range(10):
        kernel[(1, )](x, 1, BLOCK=1024)
    assert counter == 1
    device = getattr(torch, device).current_device()
    kernel.device_caches[device][0].clear()


@pytest.mark.parametrize('mode', ['enable', 'disable', 'disable_on_alignment'])
def test_specialize(mode, device, fresh_triton_cache):
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    triton.knobs.runtime.jit_cache_hook = inc_counter
    x = torch.empty(1, dtype=torch.int32, device=device)
    function = {'enable': kernel, 'disable': kernel_nospec, 'disable_on_alignment': kernel_nospec_on_alignment}[mode]
    target = {'enable': 3, 'disable': 1, 'disable_on_alignment': 2}[mode]
    for i in [1, 2, 4, 8, 16, 32]:
        function[(1, )](x, i, BLOCK=512)
    assert counter == target
    device = getattr(torch, device).current_device()
    kernel.device_caches[device][0].clear()
    kernel_nospec.device_caches[device][0].clear()
    kernel_nospec_on_alignment.device_caches[device][0].clear()


def test_annotation(device):

    @triton.jit
    def kernel(X, i: tl.int32):
        tl.store(X, i)

    x = torch.empty(1, dtype=torch.int32, device=device)

    device = getattr(torch, device).current_device()
    kernel[(1, )](x, 1)
    kernel[(1, )](x, 8)
    kernel[(1, )](x, 16)
    kernel[(1, )](x, 17)
    assert len(kernel.device_caches[device][0]) == 3


GLOBAL_DEFAULT_ARG = 1


def test_kernel_default_arg(device):
    global GLOBAL_DEFAULT_ARG

    @triton.jit
    def kernel(X, i: tl.constexpr = GLOBAL_DEFAULT_ARG):
        tl.store(X, i)

    x = torch.empty(1, dtype=torch.int32, device=device)
    kernel[(1, )](x)
    assert x == torch.ones_like(x)

    # Changing the global variable should not change the default argument in
    # `kernel`.  That value gets set at the time the function is declared.
    GLOBAL_DEFAULT_ARG = 2
    kernel[(1, )](x)
    assert x == torch.ones_like(x)

    device = getattr(torch, device).current_device()
    assert len(kernel.device_caches[device][0]) == 1


GLOBAL_VAR = tl.constexpr(1)


def test_kernel_global_var_change(device):
    global GLOBAL_VAR

    @triton.jit
    def kernel(X):
        tl.store(X, GLOBAL_VAR)

    x = torch.empty(1, dtype=torch.int32, device=device)
    kernel[(1, )](x)
    assert x == torch.ones_like(x)

    GLOBAL_VAR = 2
    with pytest.raises(RuntimeError) as e:
        kernel[(1, )](x)

    assert "global variable" in str(e.value).lower()


GLOBAL = 42  # noqa


def test_local_shadows_global():
    global GLOBAL

    @triton.jit
    def kernel():
        _, GLOBAL = 0, 0  # noqa
        a = GLOBAL  # noqa

    # No error because the `GLOBAL` we're modifying is not the same `GLOBAL` as
    # inside the kernel.
    GLOBAL = 42
    kernel[(1, )]()
    GLOBAL = 43
    kernel[(1, )]()


def test_keyword_only_shadows_global(monkeypatch):
    monkeypatch.setitem(globals(), "GLOBAL", 42)

    @triton.jit
    def kernel(*, GLOBAL: tl.constexpr):
        tl.static_assert(GLOBAL == 1)

    kernel[(1, )](GLOBAL=1)
    monkeypatch.setitem(globals(), "GLOBAL", 43)
    kernel[(1, )](GLOBAL=1)


CONSTEXPR_GLOBAL = tl.constexpr(42)


def test_local_does_not_shadow_global():
    global CONSTEXPR_GLOBAL

    @triton.jit
    def kernel():
        a = CONSTEXPR_GLOBAL  # noqa
        _, CONSTEXPR_GLOBAL = 0, 0  # noqa

    CONSTEXPR_GLOBAL = tl.constexpr(42)
    kernel[(1, )]()
    CONSTEXPR_GLOBAL = tl.constexpr(43)

    # Error because the `CONSTEXPR_GLOBAL` we're modifying is the same
    # `CONSTEXPR_GLOBAL` that's read inside `kernel`.  (Alternatively, we could
    # make this kernel an error altogether, as it is if it's a pure Python
    # function -- the fact that we store to `CONSTEXPR_GLOBAL` inside the kernel
    # makes the first read a read of the local variable, which doesn't exist
    # yet.)
    with pytest.raises(RuntimeError):
        kernel[(1, )]()


CONFLICTING_GLOBAL = tl.constexpr(0)


@triton.jit
def conflicting_global_inner():
    a = CONFLICTING_GLOBAL  # noqa


def test_conflicting_global_in_inner_function():
    global CONFLICTING_GLOBAL

    @triton.jit
    def kernel1():
        a = CONFLICTING_GLOBAL  # noqa
        conflicting_global_inner()

    @triton.jit
    def kernel2():
        a = CONFLICTING_GLOBAL  #noqa
        conflicting_global_inner()

    kernel1[(1, )]()

    # This should be an error because kernel2 calls conflicting_global_inner,
    # which saw a value for 42 for the global when it was first compiled.
    CONFLICTING_GLOBAL = 1

    with pytest.raises(RuntimeError) as e:
        kernel2[(1, )]()

    assert "Global variable CONFLICTING_GLOBAL has value" in str(e.value)


def test_use_builtin():

    @triton.jit
    def kernel():
        a = float(0)  # noqa

    # No error about the value of `float` changing.
    kernel[(1, )]()
    kernel[(1, )]()


def test_no_cache_module_as_global():

    @triton.jit
    def kernel():
        tl.arange(0, 16)

    kernel[(1, )]()
    # `tl` should not be entered into used_global_vals
    assert not kernel.used_global_vals


BUILTIN_AS_GLOBAL = tl.int32


def test_cache_builtin_as_global():
    global BUILTIN_AS_GLOBAL

    @triton.jit
    def kernel():
        x = BUILTIN_AS_GLOBAL  # noqa

    kernel[(1, )]()

    BUILTIN_AS_GLOBAL = tl.int64
    with pytest.raises(RuntimeError) as e:
        kernel[(1, )]()

    assert "global variable" in str(e.value).lower()


def test_cache_closure():

    def make_closure(cst):

        @triton.jit
        def closure():
            tl.full((16, ), cst, dtype=tl.int32)

        return closure

    cst = tl.constexpr(42)
    closure = make_closure(cst)

    closure[(1, )]()
    cst.value = 43
    with pytest.raises(RuntimeError) as e:
        closure[(1, )]()

    assert "cst has changed since we compiled this kernel, from constexpr[42] to constexpr[43]" in str(e.value)


CLOSURE_SHADOW_GLOBAL = tl.constexpr(3)


def test_cache_closure_shadows_global():

    def make_closure(value):
        CLOSURE_SHADOW_GLOBAL = value

        @triton.jit
        def closure():
            return CLOSURE_SHADOW_GLOBAL

        return closure

    first = make_closure(tl.constexpr(7))
    second = make_closure(tl.constexpr(9))
    same_as_first = make_closure(tl.constexpr(7))
    captures_none = make_closure(None)

    first_key = first.cache_key
    second_key = second.cache_key
    same_as_first_key = same_as_first.cache_key
    captures_none.cache_key

    def tracked_values(fn):
        return {name: value for (name, _), (value, _) in fn.used_global_vals.items()}

    assert first_key != second_key
    assert first_key == same_as_first_key
    assert tracked_values(first)["CLOSURE_SHADOW_GLOBAL"].value == 7
    assert tracked_values(second)["CLOSURE_SHADOW_GLOBAL"].value == 9
    assert "CLOSURE_SHADOW_GLOBAL" not in tracked_values(captures_none)


@triton.jit
def no_cache_callable_inner():
    pass


def test_no_cache_callable():

    @triton.jit
    def kernel():
        no_cache_callable_inner()

    kernel[(1, )]()
    # `no_cache_callable_inner` should not be entered into used_global_vals.
    assert not kernel.used_global_vals


def test_constexpr_cache_invalidation_recreated(device):

    def test_run(val):
        VAL = tl.constexpr(val)

        @triton.jit
        def kernel(out):
            tl.store(out, VAL)

        out = torch.zeros(1, device=device)
        kernel[(1, )](out)
        return out.item()

    assert test_run(123) == 123
    assert test_run(123) == 123
    assert test_run(1234) == 1234
    assert test_run(1234) == 1234


def test_jit_warmup_cache(device) -> None:

    @triton.jit
    def kernel_add(a, b, o, N: tl.constexpr):
        idx = tl.arange(0, N)
        tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))

    args = [
        torch.randn(32, dtype=torch.float32, device=device),
        torch.randn(32, dtype=torch.float32, device=device),
        torch.randn(32, dtype=torch.float32, device=device),
        32,
    ]
    device = getattr(torch, device).current_device()
    assert len(kernel_add.device_caches[device][0]) == 0
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add.device_caches[device][0]) == 1
    kernel_add.warmup(*args, grid=(1, ))
    assert len(kernel_add.device_caches[device][0]) == 1
    kernel_add.warmup(*args, grid=(1, ))
    assert len(kernel_add.device_caches[device][0]) == 1


def test_jit_debug(device) -> None:

    @triton.jit
    def kernel(tmp):
        tl.device_assert(tl.load(tmp) == 1, "tmp == 1")

    device = getattr(torch, device).current_device()
    tmp = torch.tensor([1], dtype=torch.int32, device=device)
    assert len(kernel.device_caches[device][0]) == 0
    kernel[(1, )](tmp, debug=False)
    assert len(kernel.device_caches[device][0]) == 1
    kernel[(1, )](tmp, debug=True)
    assert len(kernel.device_caches[device][0]) == 2
    bins = list(kernel.device_caches[device][0].values())
    assert bins[0].asm['ttir'] != bins[1].asm['ttir']


@triton.jit
def add_fn(a, b, o, N: tl.constexpr):
    idx = tl.arange(0, N)
    tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))


def test_jit_noinline(device) -> None:

    @triton.jit
    def kernel_add_device(a, b, o, N: tl.constexpr):
        add_fn(a, b, o, N)

    device = getattr(torch, device).current_device()
    assert len(kernel_add_device.device_caches[device][0]) == 0
    kernel_add_device.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add_device.device_caches[device][0]) == 1
    bins = list(kernel_add_device.device_caches[device][0].values())
    inline_ttir = bins[0].asm['ttir']
    add_fn.noinline = True
    add_fn.hash = None
    kernel_add_device.hash = None
    kernel_add_device.device_caches[device][0].clear()
    kernel_add_device.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add_device.device_caches[device][0]) == 1
    bins = list(kernel_add_device.device_caches[device][0].values())
    noinline_ttir = bins[0].asm['ttir']
    assert inline_ttir != noinline_ttir


def test_preload(device, fresh_triton_cache) -> None:

    @triton.jit
    def kernel_add(a, b, o, N: tl.constexpr, type: tl.constexpr):
        idx = tl.arange(0, N)
        tl.device_assert(idx < 32, "idx < 32")
        tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))

    @triton.jit
    def kernel_sub(a, b, o, N: tl.constexpr, type: tl.constexpr):
        idx = tl.arange(0, N)
        tl.device_assert(idx < 32, "idx < 32")
        tl.store(o + idx, tl.load(a + idx) - tl.load(b + idx))

    device = getattr(torch, device).current_device()

    # get the serialized specialization data
    specialization_data = None

    def cache_hook(*args, **kwargs):
        nonlocal specialization_data
        specialization_data = kwargs["compile"]["specialization_data"]

    triton.knobs.runtime.jit_cache_hook = cache_hook
    pre_compile = kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, tl.float32, grid=(1, ))
    hash = pre_compile.hash
    assert specialization_data is not None

    # clear the cache
    kernel_add.device_caches[device][0].clear()

    # preload the kernel
    kernel_preload = kernel_add.preload(specialization_data)
    assert kernel_preload.hash == hash
    assert len(kernel_add.device_caches[device][0]) == 1

    # we should hit the cache and not compile anything
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    triton.knobs.runtime.jit_cache_hook = inc_counter
    final_kernel = kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, tl.float32, grid=(1, ))
    assert counter == 0
    assert len(kernel_add.device_caches[device][0]) == 1
    assert final_kernel.hash == hash

    # test that we can't preload a mismatched kernel
    with pytest.raises(RuntimeError, match="Specialization data is for"):
        kernel_sub.preload(specialization_data)

    specialization_data_unknown_target = re.sub(r'("target"\s*:\s*\{[^{}]*"backend"\s*:\s*)"(.*?)"',
                                                r'\1"unknown_target"', specialization_data, count=1)

    with pytest.raises(RuntimeError, match="Specialization data is for {'backend': 'unknown_target'"):
        kernel_add.preload(specialization_data_unknown_target)


@triton.jit
def sequence_offset(idx, offsets: tl.constexpr):
    tl.static_assert(len(offsets) == 2)
    tl.static_assert(len(offsets[0]) == 2)
    tl.static_assert(len(offsets[1]) == 1)
    return idx + offsets[0][0] + offsets[0][1] + offsets[1][0]


@triton.jit
def tuple_call_kernel(out_ptr, offsets: tl.constexpr):
    tl.static_assert(len(offsets) == 2)
    idx = tl.arange(0, 1)
    tl.store(out_ptr + idx, sequence_offset(idx, offsets))


def test_preload_constexpr_tuple_arg(device, fresh_triton_cache, fresh_knobs) -> None:
    device = getattr(torch, device).current_device()
    offsets = ((2, 3), (5, ))
    specialization_data = None

    def cache_hook(*args, **kwargs):
        nonlocal specialization_data
        specialization_data = kwargs["compile"]["specialization_data"]

    fresh_knobs.runtime.jit_cache_hook = cache_hook
    tuple_call_kernel.device_caches[device][0].clear()
    pre_compile = tuple_call_kernel.warmup(torch.int32, offsets, grid=(1, ))
    hash = pre_compile.hash
    assert specialization_data is not None

    tuple_call_kernel.device_caches[device][0].clear()

    kernel_preload = tuple_call_kernel.preload(specialization_data)
    assert kernel_preload.hash == hash
    assert len(tuple_call_kernel.device_caches[device][0]) == 1

    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    fresh_knobs.runtime.jit_cache_hook = inc_counter
    final_kernel = tuple_call_kernel.warmup(torch.int32, offsets, grid=(1, ))
    assert counter == 0
    assert len(tuple_call_kernel.device_caches[device][0]) == 1
    assert final_kernel.hash == hash


def test_hooks(device, fresh_triton_cache) -> None:

    @triton.jit
    def kernel_add(a, b, o, N: tl.constexpr, type: tl.constexpr):
        idx = tl.arange(0, N)
        tl.device_assert(idx < 32, "idx < 32")
        tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))

    # get the serialized specialization data
    specialization_data = None
    is_warmup = False
    key = 0
    name = None

    def cache_hook(*args, **kwargs):
        nonlocal specialization_data
        specialization_data = kwargs["compile"]["specialization_data"]
        nonlocal is_warmup
        is_warmup = kwargs["compile"]["is_warmup"]
        nonlocal key
        key = kwargs["compile"]["key"]
        nonlocal name
        name = kwargs["fn"].name

    specialization_data_compiled = None

    def compiled_hook(*args, **kwargs):
        nonlocal specialization_data_compiled
        specialization_data_compiled = kwargs["compile"]["specialization_data"]

    triton.knobs.runtime.jit_cache_hook = cache_hook
    triton.knobs.runtime.jit_post_compile_hook = compiled_hook
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, tl.float32, grid=(1, ))
    assert specialization_data is not None and specialization_data_compiled == specialization_data
    assert is_warmup is True
    assert key in kernel_add.device_caches[getattr(torch, device).current_device()][0]
    assert name == "test_hooks.<locals>.kernel_add"


@pytest.mark.xfail(reason="within_2g is a HIP specific optimization", condition=not is_hip(), run=False)
def test_within_2gb(device, fresh_triton_cache) -> None:
    default_buffer_ops = os.environ.get("AMDGCN_USE_BUFFER_OPS", "0")
    try:
        use_buffer_ops_opts = ["1", "0"]
        # The ranges should only be available when buffer ops are enabled
        pointer_ranges = [[(0, )], []]
        for use_buffer_ops, pointer_range in zip(use_buffer_ops_opts, pointer_ranges):
            # Set AMDGCN_USE_BUFFER_OPS
            os.environ["AMDGCN_USE_BUFFER_OPS"] = use_buffer_ops

            @triton.jit
            def kernel_add(a):
                tl.load(a)

            # This is the attribute we want to test
            pointer_range_32 = None

            def cache_hook(*args, **kwargs):
                nonlocal pointer_range_32
                pointer_range_32 = [
                    k for k, v in kwargs["compile"]["configs"][0].items() if ["tt.pointer_range", 32] in v
                ]

            triton.knobs.runtime.jit_cache_hook = cache_hook
            # In warmup we assume that the pointer range is 32 bits
            kernel_add.warmup(torch.float32, grid=(1, ))
            assert pointer_range_32 == pointer_range
            # Torch tensor > 2GB
            kernel_add[(1, 0)](torch.empty(2**31, dtype=torch.int8, device=device))
            assert len(pointer_range_32) == 0
            # Torch tensor <= 2GB
            kernel_add[(1, 0)](torch.empty(2**31 - 1, dtype=torch.int8, device=device))
            assert pointer_range_32 == pointer_range
    finally:
        os.environ["AMDGCN_USE_BUFFER_OPS"] = default_buffer_ops


def test_function_arguments(device):

    @triton.jit
    def func1():
        return 1

    @triton.jit
    def func2():
        return 2

    @triton.jit
    def func3(x):
        return x

    @triton.jit
    def func4(x, y):
        return x + y

    @triton.jit
    def kernel(Y, fn: tl.constexpr, fn_args):
        tl.store(Y, fn(*fn_args))

    y = torch.zeros((5, ), dtype=torch.int32, device=device)
    kernel[(1, )](y[0], func1, tuple())
    kernel[(1, )](y[1], func2, tuple())
    kernel[(1, )](y[2], func3, (3, ))
    kernel[(1, )](y[3], func4, (3, 4))
    kernel[(1, )](y[4], func1, tuple())

    device = getattr(torch, device).current_device()
    assert len(kernel.device_caches[device][0]) == 4
    assert y.tolist() == [1, 2, 3, 7, 1]


class MockThreadPool(Executor):

    def __init__(self):
        self.work_queue = []

    def submit(self, fn, *args, **kwargs):
        future = Future()

        def task():
            if not future.set_running_or_notify_cancel():
                return

            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        self.work_queue.append(task)
        return future

    def run_one(self):
        task = self.work_queue.pop(0)
        task()

    def run_all(self):
        while self.work_queue:
            self.run_one()

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.run_all()


def test_async_compile_mock(device, fresh_triton_cache):

    @triton.jit
    def kernel(Y, a: tl.constexpr):
        tl.store(Y, a)

    with (
            MockThreadPool() as pool,
            triton.AsyncCompileMode(pool),
    ):
        a = torch.empty((16, 16), device=device)
        b = torch.empty((16, 16), dtype=torch.int32, device=device)
        kernel.warmup(a, 0, grid=(1, ))
        kernel.warmup(a, 1, grid=(1, ))
        kernel.warmup(b, 0, grid=(1, ))
        kernel.warmup(b, 1, grid=(1, ))

        device = getattr(torch, device).current_device()

        # Nothing has actually compiled yet
        assert len(kernel.device_caches[device][0]) == 4
        assert len(pool.work_queue) == 4

        # Duplicates are only submitted once
        kernel.warmup(a, 0, grid=(1, ))
        kernel.warmup(a, 1, grid=(1, ))
        assert len(kernel.device_caches[device][0]) == 4
        assert len(pool.work_queue) == 4

        pool.run_one()
        kernel[(1, )](a, 0)
        assert len(kernel.device_caches[device][0]) == 4
        assert a[0, 0] == 0.0

        pool.run_all()


def test_async_compile(device, fresh_triton_cache):

    @triton.jit
    def kernel(Y, a: tl.constexpr):
        tl.store(Y, a)

    with (
            ThreadPoolExecutor(2) as pool,
            triton.AsyncCompileMode(pool),
    ):
        a = torch.empty((16, 16), device=device)
        b = torch.empty((16, 16), dtype=torch.int32, device=device)
        kernel.warmup(a, 0, grid=(1, ))
        kernel.warmup(a, 1, grid=(1, ))
        kernel.warmup(b, 0, grid=(1, ))
        kernel.warmup(b, 1, grid=(1, ))

        device = getattr(torch, device).current_device()
        assert len(kernel.device_caches[device][0]) == 4

        kernel[(1, )](b, 1)
        assert b[0, 0] == 1
        kernel[(1, )](b, 0)
        assert b[0, 0] == 0
        kernel[(1, )](a, 0)
        assert a[0, 0] == 0
        kernel[(1, )](a, 1)
        assert a[0, 0] == 1
        kernel[(1, )](a, 2)
        assert a[0, 0] == 2


def test_async_compile_error(fresh_triton_cache):

    @triton.jit
    def fn(x: tl.constexpr):
        tl.static_assert(x == 2)

    with pytest.raises(triton.compiler.errors.CompileTimeAssertionFailure):
        with (
                ThreadPoolExecutor(2) as pool,
                triton.AsyncCompileMode(pool),
        ):
            assert triton.runtime._async_compile.active_mode.get() is not None
            fn.warmup(1, grid=(1, ))

            assert len(fn.device_caches[0][0]) == 1

    # After the AsyncCompileMode context manager exits, the active mode should
    # be set to None again, even if there was an error.
    assert triton.runtime._async_compile.active_mode.get() is None
    # Failed async placeholders must not stay cached; otherwise their Future
    # objects keep exception tracebacks alive.
    assert len(fn.device_caches[0][0]) == 0


def test_async_compile_multiple_errors(fresh_triton_cache):

    @triton.jit
    def bad_a(x: tl.constexpr):
        tl.static_assert(x == 111)

    @triton.jit
    def bad_b(x: tl.constexpr):
        tl.static_assert(x == 222)

    with pytest.raises(triton.compiler.errors.CompileTimeAssertionFailure):
        with (
                ThreadPoolExecutor(2) as pool,
                triton.AsyncCompileMode(pool),
        ):
            bad_a.warmup(1, grid=(1, ))
            bad_b.warmup(1, grid=(1, ))

            assert len(bad_a.device_caches[0][0]) == 1
            assert len(bad_b.device_caches[0][0]) == 1

    # The drain has to reach every pending compile, not stop at the first one
    # that raises: a leftover FutureKernel keeps its exception traceback, and
    # through it the whole compilation context, alive until interpreter exit.
    assert len(bad_a.device_caches[0][0]) == 0
    assert len(bad_b.device_caches[0][0]) == 0
    assert triton.runtime._async_compile.active_mode.get() is None


def _make_failed_future_kernel():
    future = Future()
    future.set_running_or_notify_cancel()
    try:
        raise ValueError("boom")
    except ValueError as exc:
        # Set from inside the handler so the future carries a real traceback.
        future.set_exception(exc)
    future_kernel = triton.FutureKernel(future)
    future_kernel.add_callbacks(lambda kernel: None, lambda fk: None)
    return future_kernel


def test_async_compile_failed_future_repeated_result():
    resolved_twice = _make_failed_future_kernel()
    with pytest.raises(ValueError, match="boom"):
        resolved_twice.result()
    with pytest.raises(RuntimeError, match="previously failed"):
        resolved_twice.result()

    ignored_twice = _make_failed_future_kernel()
    assert ignored_twice.result(ignore_errors=True) is None
    assert ignored_twice.result(ignore_errors=True) is None

    attribute_probe = _make_failed_future_kernel()
    assert attribute_probe.result(ignore_errors=True) is None
    with pytest.raises(RuntimeError, match="previously failed"):
        _ = attribute_probe.does_not_exist

    late_waiter = _make_failed_future_kernel()
    assert late_waiter.result(ignore_errors=True) is None
    drained = []
    late_waiter.add_callbacks(lambda kernel: drained.append("finalize"), lambda fk: drained.append("cleanup"))
    assert late_waiter.result(ignore_errors=True) is None
    assert drained == ["cleanup"]

    for future_kernel in (resolved_twice, ignored_twice, attribute_probe, late_waiter):
        assert not any(isinstance(value, BaseException) for value in vars(future_kernel).values())


def test_async_compile_base_exception_evicts_cache():
    cache = {}
    future = Future()
    future.set_running_or_notify_cancel()
    try:
        raise KeyboardInterrupt("worker interrupted")
    except KeyboardInterrupt as exc:
        future.set_exception(exc)

    def store(kernel):
        cache["K"] = kernel

    def evict(_future_kernel):
        cache.pop("K", None)

    future_kernel = triton.FutureKernel(future)
    future_kernel.add_callbacks(store, evict)
    cache["K"] = future_kernel

    # A worker can fail with a BaseException, and the bookkeeping still has to
    # run: a retained Future keeps the exception traceback and every compiler
    # frame behind it. ignore_errors covers compile errors only, so an
    # interpreter-level exception must still propagate.
    with pytest.raises(KeyboardInterrupt):
        future_kernel.result(ignore_errors=True)

    assert "K" not in cache
    assert future_kernel.future is None
    assert future_kernel._state == "failed"
    assert not any(isinstance(value, BaseException) for value in vars(future_kernel).values())

    with pytest.raises(RuntimeError, match="previously failed"):
        future_kernel.result()

    # ignore_errors only ever covers compilation errors, so the terminal state
    # of an interrupted compile has to keep raising rather than quietly
    # returning None on every later resolution.
    with pytest.raises(RuntimeError, match="previously failed"):
        future_kernel.result(ignore_errors=True)

    # Contrast: a compile that failed with a plain Exception stays ignorable.
    exception_future = Future()
    exception_future.set_running_or_notify_cancel()
    try:
        raise ValueError("compile blew up")
    except ValueError as exc:
        exception_future.set_exception(exc)

    ignorable = triton.FutureKernel(exception_future)
    ignorable.add_callbacks(store, evict)
    assert ignorable.result(ignore_errors=True) is None
    assert ignorable.result(ignore_errors=True) is None


def test_async_compile_base_exception_mid_drain_evicts_pending():
    INTERRUPT_DELAY = 0.05
    SLOW_COMPILE_DELAY = 1.5
    cache = {}

    def finalize(slot):

        def store(kernel):
            cache[slot] = kernel

        return store

    def cleanup(slot):

        def evict(_future_kernel):
            cache.pop(slot, None)

        return evict

    def interrupted_compile():
        time.sleep(INTERRUPT_DELAY)
        raise KeyboardInterrupt("simulated ctrl-c in worker")

    def slow_compile():
        time.sleep(SLOW_COMPILE_DELAY)
        return "kernel-b"

    # The pool is driven by hand rather than with `with`: its shutdown waits for
    # the slow compile, which would hide the very promptness being measured.
    pool = ThreadPoolExecutor(2)
    try:
        started = time.perf_counter()
        with pytest.raises(KeyboardInterrupt):
            with triton.AsyncCompileMode(pool) as mode:
                cache["a"] = mode.submit("KA", interrupted_compile, finalize("a"), cleanup("a"))
                abandoned = mode.submit("KB", slow_compile, finalize("b"), cleanup("b"))
                cache["b"] = abandoned
        elapsed = time.perf_counter() - started
    finally:
        pool.shutdown(wait=True)

    assert not isinstance(cache.get("a"), triton.FutureKernel)
    assert not isinstance(cache.get("b"), triton.FutureKernel)
    assert abandoned.future is None
    assert abandoned._state == "failed"
    assert not any(isinstance(value, BaseException) for value in vars(abandoned).values())
    # Aborting must not wait on compiles still in flight; without this the fix
    # could regress into draining everything before re-raising.
    assert elapsed < 1.0


def test_async_compile_interrupt_while_waiting_evicts_pending(monkeypatch):
    SENTINEL = object()
    cache = {}

    def interrupted_as_completed(_futures):
        raise KeyboardInterrupt("simulated ctrl-c while waiting for compiles")

    monkeypatch.setattr(triton.runtime._async_compile, "as_completed", interrupted_as_completed)

    def finalize(slot):

        def store(kernel):
            cache[slot] = kernel

        return store

    def cleanup(slot):

        def evict(_future_kernel):
            cache.pop(slot, None)

        return evict

    # A real Ctrl-C most likely lands while the thread is parked inside
    # as_completed(), which raises from the for statement rather than from
    # result(), so the abandon sweep has to cover the whole drain and not just
    # the resolution of one future.
    with MockThreadPool() as pool:
        with pytest.raises(KeyboardInterrupt):
            with triton.AsyncCompileMode(pool) as mode:
                cache["a"] = mode.submit("KA", lambda: SENTINEL, finalize("a"), cleanup("a"))
                cache["b"] = mode.submit("KB", lambda: SENTINEL, finalize("b"), cleanup("b"))

    assert not isinstance(cache.get("a"), triton.FutureKernel)
    assert not isinstance(cache.get("b"), triton.FutureKernel)


def test_async_compile_interrupt_cleanup_submit_evicts_everything(monkeypatch):
    SENTINEL = object()
    cache = {}
    queued = []

    def interrupted_as_completed(_futures):
        raise KeyboardInterrupt("simulated ctrl-c while waiting for compiles")

    monkeypatch.setattr(triton.runtime._async_compile, "as_completed", interrupted_as_completed)

    def finalize(slot):

        def store(kernel):
            cache[slot] = kernel

        return store

    def cleanup(slot):

        def evict(_future_kernel):
            cache.pop(slot, None)

        return evict

    with MockThreadPool() as pool:
        with pytest.raises(KeyboardInterrupt):
            with triton.AsyncCompileMode(pool) as mode:

                def evict_and_submit_new_key(_future_kernel):
                    cache.pop("a", None)
                    # Inserts a brand-new key while the abandon sweep is running
                    # its own cleanup callbacks, so a sweep that walked a
                    # snapshot taken up front could never visit it.
                    second = mode.submit("KB", lambda: SENTINEL, finalize("b"), cleanup("b"))
                    cache["b"] = second
                    queued.append(second)

                cache["a"] = mode.submit("KA", lambda: SENTINEL, finalize("a"), evict_and_submit_new_key)

    assert not isinstance(cache.get("a"), triton.FutureKernel)
    assert not isinstance(cache.get("b"), triton.FutureKernel)
    assert len(queued) == 1
    assert queued[0]._state == "failed"
    assert queued[0].future is None


def test_async_compile_body_interrupt_does_not_wait():
    cache = {}

    def slow_compile():
        time.sleep(1.5)
        return object()

    def store(kernel):
        cache["slow"] = kernel

    def evict(_future_kernel):
        cache.pop("slow", None)

    pool = ThreadPoolExecutor(2)
    try:
        started = time.perf_counter()
        with pytest.raises(KeyboardInterrupt):
            with triton.AsyncCompileMode(pool) as mode:
                cache["slow"] = mode.submit("slow", slow_compile, store, evict)
                raise KeyboardInterrupt("simulated ctrl-c in the with body")
        elapsed = time.perf_counter() - started
    finally:
        # Executor.__exit__ waits for the slow compile, which would fold that
        # wait into the measurement, so shut the pool down outside it.
        pool.shutdown(wait=True)

    assert not isinstance(cache.get("slow"), triton.FutureKernel)
    # An interrupt raised in the body must abort just as promptly as one coming
    # out of a worker, instead of draining every compile still in flight.
    assert elapsed < 1.0


def test_async_compile_future_kernel_dunder_probe_does_not_compile():
    never_completed = Future()
    future_kernel = triton.FutureKernel(never_completed)
    future_kernel.add_callbacks(lambda kernel: None, lambda fk: None)

    assert hasattr(future_kernel, "__deepcopy__") is False
    assert hasattr(future_kernel, "__copy__") is False
    assert hasattr(future_kernel, "__setstate__") is False
    # Probing must not have blocked on (or resolved) the pending compile.
    assert future_kernel.future is never_completed
    assert not never_completed.done()


def test_async_compile_future_kernel_forwards_private_kernel_api():

    class FakeKernel:

        def __init__(self):
            self.calls = []
            self._run = "launcher"

        def _init_handles(self):
            self.calls.append("_init_handles")

    fake = FakeKernel()
    future = Future()
    future.set_result(fake)
    future_kernel = triton.FutureKernel(future)
    future_kernel.add_callbacks(lambda kernel: None, lambda fk: None)

    # CompiledKernel exposes _init_handles/_run and callers reach for them
    # through whatever the JIT cache handed back, so the proxy has to forward
    # single-underscore names instead of treating them as its own privates.
    future_kernel._init_handles()
    assert fake.calls == ["_init_handles"]
    assert future_kernel._run == "launcher"
    assert future_kernel.kernel is fake


def test_async_compile_future_kernel_new_without_init():
    uninitialized = triton.FutureKernel.__new__(triton.FutureKernel)

    # Allocated without __init__, so there is no state to resolve. __getattr__
    # must refuse instead of recursing through result() to read the very
    # attribute it was asked for, which would hang or blow the stack.
    with pytest.raises(AttributeError):
        getattr(uninitialized, "anything")
    with pytest.raises(AttributeError):
        getattr(uninitialized, "_state")
    assert hasattr(uninitialized, "__deepcopy__") is False


def test_async_compile_future_kernel_forwards_getitem():
    launched = []

    class FakeKernel:

        def __getitem__(self, grid):
            launched.append(grid)
            return f"runner{grid}"

    future = Future()
    future.set_result(FakeKernel())
    future_kernel = triton.FutureKernel(future)
    future_kernel.add_callbacks(lambda kernel: None, lambda fk: None)

    # kernel[grid](...) is the launch idiom, and implicit dunder lookup goes to
    # the type rather than through __getattr__, so the proxy needs a real
    # __getitem__ for a warmed-up kernel to be launchable at all.
    assert future_kernel[(1, 2, 3)] == "runner(1, 2, 3)"
    assert future_kernel.__getitem__((4, 5, 6)) == "runner(4, 5, 6)"
    assert launched == [(1, 2, 3), (4, 5, 6)]
    assert hasattr(future_kernel, "__deepcopy__") is False


def test_async_compile_cleanup_base_exception_still_evicts():
    calls = []
    cache = {"a": "placeholder", "b": "placeholder", "c": "placeholder"}
    future = Future()
    future.set_running_or_notify_cancel()
    try:
        raise ValueError("compile blew up")
    except ValueError as exc:
        future.set_exception(exc)

    def cleanup(slot, interrupt=False):

        def evict(_future_kernel):
            calls.append(slot)
            cache.pop(slot, None)
            if interrupt:
                raise KeyboardInterrupt("simulated ctrl-c in a cleanup callback")

        return evict

    future_kernel = triton.FutureKernel(future)
    future_kernel.add_callbacks(lambda kernel: None, cleanup("a", interrupt=True))
    future_kernel.add_callbacks(lambda kernel: None, cleanup("b"))
    future_kernel.add_callbacks(lambda kernel: None, cleanup("c"))

    # A second Ctrl-C is plausible exactly while cleanup is running, and it must
    # not cost the remaining waiters their eviction: all of them run, and the
    # interrupt surfaces afterwards.
    with pytest.raises(KeyboardInterrupt):
        future_kernel.result()

    assert calls == ["a", "b", "c"]
    assert cache == {}
    assert future_kernel._state == "failed"
    assert future_kernel.future is None


def test_async_compile_callback_error_does_not_block_others():
    compiled = object()
    calls = []

    def record(name, raises=False):

        def callback(_):
            calls.append(name)
            if raises:
                raise RuntimeError(name)

        return callback

    with MockThreadPool() as pool:
        succeeding = triton.FutureKernel(pool.submit(lambda: compiled))
        succeeding.add_callbacks(record("f1", raises=True), record("c1"))
        succeeding.add_callbacks(record("f2"), record("c2"))
        succeeding.add_callbacks(record("f3"), record("c3"))
        pool.run_one()

        with pytest.raises(Exception) as finalize_failure:
            succeeding.result()
        assert calls == ["f1", "f2", "f3"]
        assert str(finalize_failure.value) == "f1"

        calls.clear()

        def failing_compile():
            raise ValueError("compile blew up")

        failing = triton.FutureKernel(pool.submit(failing_compile))
        failing.add_callbacks(record("f1"), record("c1", raises=True))
        failing.add_callbacks(record("f2"), record("c2"))
        failing.add_callbacks(record("f3"), record("c3"))
        pool.run_one()

        with pytest.raises(Exception) as compile_failure:
            failing.result()
        assert calls == ["c1", "c2", "c3"]
        assert isinstance(compile_failure.value, ValueError)
        assert str(compile_failure.value) == "compile blew up"


def test_async_compile_shared_key_late_waiter():
    SENTINEL = object()
    cache = {}

    def finalize(slot):

        def store(kernel):
            cache[slot] = kernel

        return store

    def cleanup(slot):

        def evict(_future_kernel):
            cache.pop(slot, None)

        return evict

    with MockThreadPool() as pool, triton.AsyncCompileMode(pool) as mode:
        # Mirrors jit.py: the cache slot is overwritten with the FutureKernel
        # right after submit() returns, so a late waiter that joins a resolved
        # compile has to be finalized by result().
        first = mode.submit("K", lambda: SENTINEL, finalize("a"), cleanup("a"))
        cache["a"] = first
        pool.run_one()
        assert first.result() is SENTINEL
        assert cache["a"] is SENTINEL

        late = mode.submit("K", lambda: SENTINEL, finalize("b"), cleanup("b"))
        cache["b"] = late
        assert late is first

    assert cache["b"] is SENTINEL
    assert not isinstance(cache["b"], triton.FutureKernel)


def test_async_compile_exit_drains_reentrant_submits():
    SENTINEL = object()
    cache = {}

    def finalize(slot):

        def store(kernel):
            cache[slot] = kernel

        return store

    def cleanup(slot):

        def evict(_future_kernel):
            cache.pop(slot, None)

        return evict

    with ThreadPoolExecutor(2) as pool, triton.AsyncCompileMode(pool) as mode:

        def finalize_and_submit_more(kernel):
            cache["a"] = kernel
            # Queued from inside a finalize callback, i.e. after the drain has
            # already started: __exit__ must notice the new work instead of
            # walking a one-shot snapshot of the future list.
            cache["b"] = mode.submit("KB", lambda: SENTINEL, finalize("b"), cleanup("b"))

        cache["a"] = mode.submit("KA", lambda: SENTINEL, finalize_and_submit_more, cleanup("a"))

    assert cache["a"] is SENTINEL
    assert cache["b"] is SENTINEL
    assert not isinstance(cache["b"], triton.FutureKernel)
    assert mode.raw_futures == []
    assert mode.future_kernels == {}


def test_async_compile_exit_drains_same_key_reentrant_submits():
    SENTINEL = object()
    cache = {}

    def finalize(slot):

        def store(kernel):
            cache[slot] = kernel

        return store

    def cleanup(slot):

        def evict(_future_kernel):
            cache.pop(slot, None)

        return evict

    with ThreadPoolExecutor(2) as pool, triton.AsyncCompileMode(pool) as mode:

        def finalize_and_resubmit_same_key(kernel):
            cache["a"] = kernel
            # Same key, so submit() takes the dedup path: this appends another
            # finalize callback but queues no new future, which leaves the
            # callback runner as the only thing that can ever drain it.
            cache["b"] = mode.submit("K", lambda: SENTINEL, finalize("b"), cleanup("b"))

        cache["a"] = mode.submit("K", lambda: SENTINEL, finalize_and_resubmit_same_key, cleanup("a"))

    assert cache["a"] is SENTINEL
    assert cache["b"] is SENTINEL
    assert not isinstance(cache["a"], triton.FutureKernel)
    assert not isinstance(cache["b"], triton.FutureKernel)
    assert mode.raw_futures == []
    assert mode.future_kernels == {}


def test_async_compile_reentrant_result_from_finalize_callback():
    SENTINEL = object()
    calls = []
    future = Future()
    future.set_result(SENTINEL)
    future_kernel = triton.FutureKernel(future)

    def finalize_reentrant(kernel):
        calls.append("reentrant")
        assert future_kernel.result() is kernel

    def finalize_plain(_kernel):
        calls.append("plain")

    def cleanup(_future_kernel):
        calls.append("cleanup")

    future_kernel.add_callbacks(finalize_reentrant, cleanup)
    future_kernel.add_callbacks(finalize_plain, cleanup)

    # Resolving from inside a finalize callback must not re-run any callback nor
    # leave the lifecycle half-applied, even though the outer resolution has not
    # reached its terminal assignments yet.
    assert future_kernel.result() is SENTINEL
    assert calls == ["reentrant", "plain"]
    assert future_kernel._state == "succeeded"
    assert future_kernel.future is None
    assert future_kernel.kernel is SENTINEL

    assert future_kernel.result() is SENTINEL
    assert calls == ["reentrant", "plain"]


def test_async_compile_same_key_submit_from_cleanup_callback():
    cache = {}

    def compile_fails():
        raise ValueError("compile blew up")

    def cleanup(slot):

        def evict(_future_kernel):
            cache.pop(slot, None)

        return evict

    with pytest.raises(ValueError, match="compile blew up"):
        with MockThreadPool() as pool, triton.AsyncCompileMode(pool) as mode:

            def resubmit_same_key(_future_kernel):
                cache.pop("a", None)
                # Same key from inside a cleanup callback: submit() dedups onto
                # this very FutureKernel, so only the cleanup runner can ever
                # reach the callback it appends.
                cache["b"] = mode.submit("K", compile_fails, lambda kernel: None, cleanup("b"))

            cache["a"] = mode.submit("K", compile_fails, lambda kernel: None, resubmit_same_key)
            pool.run_all()

    assert "a" not in cache
    assert "b" not in cache


def test_async_compile_mode_rejects_nesting():
    with MockThreadPool() as pool, triton.AsyncCompileMode(pool) as outer:
        assert triton.runtime._async_compile.active_mode.get() is outer

        with pytest.raises(RuntimeError, match="already active"):
            with triton.AsyncCompileMode(pool):
                pass

        # __enter__ raised before installing itself, so __exit__ never ran and
        # the outer mode must still be the active one.
        assert triton.runtime._async_compile.active_mode.get() is outer

    assert triton.runtime._async_compile.active_mode.get() is None


def test_async_compile_ignore_errors_mode_swallows_compile_failures():
    cache = {}

    def compile_fails():
        raise ValueError("compile blew up")

    def store(kernel):
        cache["K"] = kernel

    def evict(_future_kernel):
        cache.pop("K", None)

    with MockThreadPool() as pool, triton.AsyncCompileMode(pool, ignore_errors=True) as mode:
        cache["K"] = mode.submit("K", compile_fails, store, evict)
        pool.run_all()

    assert "K" not in cache
    assert triton.runtime._async_compile.active_mode.get() is None


def test_async_compile_submit_deduplicates_by_key():
    SENTINEL = object()
    cache = {}
    compiles = []

    def compile_once():
        compiles.append("compile")
        return SENTINEL

    def finalize(slot):

        def store(kernel):
            cache[slot] = kernel

        return store

    def cleanup(slot):

        def evict(_future_kernel):
            cache.pop(slot, None)

        return evict

    with MockThreadPool() as pool, triton.AsyncCompileMode(pool) as mode:
        first = mode.submit("K", compile_once, finalize("a"), cleanup("a"))
        cache["a"] = first
        second = mode.submit("K", compile_once, finalize("b"), cleanup("b"))
        cache["b"] = second

        assert second is first
        assert len(pool.work_queue) == 1
        assert len(mode.raw_futures) == 1
        pool.run_all()

    assert compiles == ["compile"]
    assert cache["a"] is SENTINEL
    assert cache["b"] is SENTINEL


def test_async_compile_body_exception_with_compile_failure():
    cache = {}

    def compile_fails():
        raise ValueError("compile blew up")

    def store(kernel):
        cache["K"] = kernel

    def evict(_future_kernel):
        cache.pop("K", None)

    with pytest.raises(ValueError, match="compile blew up") as compile_failure:
        with MockThreadPool() as pool, triton.AsyncCompileMode(pool) as mode:
            cache["K"] = mode.submit("K", compile_fails, store, evict)
            pool.run_all()
            raise RuntimeError("body blew up")

    # The drain's first error is what propagates, and the body exception has to
    # survive as its context so neither failure is lost.
    assert isinstance(compile_failure.value.__context__, RuntimeError)
    assert str(compile_failure.value.__context__) == "body blew up"
    assert "K" not in cache


def test_async_compile_finalize_error_leaves_terminal_state():
    SENTINEL = object()
    calls = []
    future = Future()
    future.set_result(SENTINEL)

    def finalize_raises(_kernel):
        calls.append("finalize")
        raise RuntimeError("finalize blew up")

    def cleanup(_future_kernel):
        calls.append("cleanup")

    future_kernel = triton.FutureKernel(future)
    future_kernel.add_callbacks(finalize_raises, cleanup)

    with pytest.raises(RuntimeError, match="finalize blew up"):
        future_kernel.result()

    assert future_kernel.kernel is SENTINEL
    assert future_kernel.future is None
    assert future_kernel._state == "succeeded"
    assert calls == ["finalize"]

    # A callback error must not leave the compile unresolved: resolving again
    # hands back the kernel and must not re-run the callback that raised.
    assert future_kernel.result() is SENTINEL
    assert calls == ["finalize"]


def test_async_compile_finalize_base_exception_leaves_terminal_state():
    SENTINEL = object()
    calls = []
    future = Future()
    future.set_result(SENTINEL)

    def finalize_interrupts(_kernel):
        calls.append("interrupt")
        raise KeyboardInterrupt("simulated ctrl-c in a finalize callback")

    def finalize_plain(_kernel):
        calls.append("plain")

    def cleanup(_future_kernel):
        calls.append("cleanup")

    future_kernel = triton.FutureKernel(future)
    future_kernel.add_callbacks(finalize_interrupts, cleanup)
    future_kernel.add_callbacks(finalize_plain, cleanup)

    # The compile itself succeeded, so an interrupt out of a callback must not
    # cost the remaining waiters their finalize nor abandon the terminal
    # transition: dropping it would lose the kernel and keep the future.
    with pytest.raises(KeyboardInterrupt):
        future_kernel.result()

    assert calls == ["interrupt", "plain"]
    assert future_kernel._state == "succeeded"
    assert future_kernel.future is None
    assert future_kernel.kernel is SENTINEL
    assert not any(isinstance(value, BaseException) for value in vars(future_kernel).values())


def test_higher_order_kernel(device, fresh_triton_cache, capsys):

    @triton.jit
    def fn_a():
        tl.static_print("Compiling with fn_a")
        return 0

    @triton.jit
    def kernel(out_ptr, FUNC: tl.constexpr) -> None:
        val = FUNC()
        tl.store(out_ptr, val)

    output = torch.empty((), device=device, dtype=torch.int32)
    kernel[(1, )](output, fn_a)
    assert output.item() == 0

    # Test we can update src in-place
    orig_src = fn_a.src
    new_src = orig_src.replace("with fn_a", "with fn_a after modification")
    new_src = new_src.replace("0", "1")
    fn_a._unsafe_update_src(new_src)
    kernel[(1, )](output, fn_a)
    assert output.item() == 1

    # Test that the on disc cache works
    kernel.device_caches.clear()
    kernel[(1, )](output, fn_a)
    assert output.item() == 1

    fn_a._unsafe_update_src(orig_src)
    kernel[(1, )](output, fn_a)
    assert output.item() == 0

    expecttest.assert_expected_inline(capsys.readouterr().out, """\
Compiling with fn_a
Compiling with fn_a after modification
""")


def test_preload_higher_order_kernels(device, fresh_triton_cache) -> None:

    @triton.jit
    def fn_a():
        return 17

    @triton.jit
    def fn_b():
        return 31

    @triton.jit
    def kernel(out_ptr, FUNC: tl.constexpr) -> None:
        val = FUNC()
        tl.store(out_ptr, val)

    device = getattr(torch, device).current_device()

    # get the serialized specialization data
    specialization_data = None

    def cache_hook(*args, **kwargs):
        nonlocal specialization_data
        specialization_data = kwargs["compile"]["specialization_data"]

    triton.knobs.runtime.jit_cache_hook = cache_hook
    output = torch.empty((), device=device, dtype=torch.int32)
    compiled_kernel = kernel[(1, )](output, fn_a)
    assert output.item() == 17
    hash = compiled_kernel.hash
    assert specialization_data is not None

    # clear the cache
    kernel.device_caches[device][0].clear()

    # preload the kernel
    kernel_preload = kernel.preload(specialization_data)
    assert kernel_preload.hash == hash
    assert len(kernel.device_caches[device][0]) == 1

    # we should hit the cache and not compile anything
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    triton.knobs.runtime.jit_cache_hook = inc_counter
    final_kernel = kernel[(1, )](output, fn_a)
    assert counter == 0
    assert len(kernel.device_caches[device][0]) == 1
    assert final_kernel.hash == hash

    # different function should compile and not hit the cache
    kernel[(1, )](output, fn_b)
    assert counter == 1
    assert output.item() == 31


def test_module_load_unload(device, fresh_knobs):

    @triton.jit
    def kernel(out_ptr, val) -> None:
        tl.store(out_ptr, val)

    # we should hit the kernel unload call to decrese the counter from 1 to 0
    counter = 1
    owner_pid = os.getpid()

    def kernel_unload(*args, **kwargs):
        nonlocal counter
        assert os.getpid() == owner_pid
        counter -= 1

    # turn off python garbage collector, so the callback is not called
    # in the garbage collector
    gc.disable()
    triton.knobs.runtime.kernel_unload_hook.add(kernel_unload)

    out = torch.randn(1, dtype=torch.float32, device=device)
    pre_compile = kernel.warmup(out, 1, grid=(1, ))
    pre_compile._init_handles()

    assert counter == 1
    assert pre_compile.module is not None

    if "fork" in multiprocessing.get_all_start_methods():
        child = multiprocessing.get_context("fork").Process(target=pre_compile.__del__)
        child.start()
        child.join()
        assert child.exitcode == 0
        assert counter == 1

    pre_compile.__del__()

    assert counter == 0
    assert pre_compile.module is None
    # turn on garbage collector
    gc.enable()
