"""
Helper module containing Triton kernels for test_debug.py.
These kernels are separated so they can be called from subprocesses.
"""
import torch
import triton
import triton.language as tl
import sys


def _run_and_catch(kernel_fn, device):
    try:
        kernel_fn()
        getattr(torch, device).synchronize()
        return 0
    except RuntimeError:
        return 1
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return 2


def run_device_assert_kernel(cond, mask, opt_flag, jit_flag, device):

    @triton.jit(debug=jit_flag)
    def _kernel(COND: tl.constexpr, MASK: tl.constexpr):
        tl.device_assert(COND, 'test', mask=MASK)

    kwargs = {}
    if opt_flag is not None:
        kwargs["debug"] = opt_flag

    return _run_and_catch(lambda: _kernel[(1, )](cond, mask, **kwargs), device)


@triton.jit
def _kernel_add(X, Y, Z):
    tl.store(Z, tl.load(X) + tl.load(Y))


@triton.jit
def _kernel_mul(X, Y, Z):
    tl.store(Z, tl.load(X) * tl.load(Y))


@triton.jit
def _kernel_sub(X, Y, Z):
    tl.store(Z, tl.load(X) - tl.load(Y))


def run_overflow_kernel(op, x, y, x_dtype, y_dtype, debug, device):
    ops = {"add": _kernel_add, "mul": _kernel_mul, "sub": _kernel_sub}
    tri_func = ops[op]
    x = torch.tensor([x], dtype=getattr(torch, x_dtype), device=device)
    y = torch.tensor([y], dtype=getattr(torch, y_dtype), device=device)
    z = torch.empty_like(x)
    return _run_and_catch(lambda: tri_func[(1, )](x, y, z, debug=debug), device)


if __name__ == "__main__":

    def parse_bool_or_none(arg_str):
        if arg_str == "None":
            return None
        return arg_str == "True"

    test_type = sys.argv[1]
    if test_type == "device_assert":
        cond = sys.argv[2] == "True"
        mask = parse_bool_or_none(sys.argv[3])
        opt_flag = parse_bool_or_none(sys.argv[4])
        jit_flag = sys.argv[5] == "True"
        device = sys.argv[6]
        triton.knobs.refresh_knobs()
        exit_code = run_device_assert_kernel(cond, mask, opt_flag, jit_flag, device)
        sys.exit(exit_code)

    elif test_type == "overflow":
        op = sys.argv[2]
        x = int(sys.argv[3])
        y = int(sys.argv[4])
        x_dtype = sys.argv[5]
        y_dtype = sys.argv[6]
        debug = sys.argv[7] == "True"
        device = sys.argv[8]
        exit_code = run_overflow_kernel(op, x, y, x_dtype, y_dtype, debug, device)
        sys.exit(exit_code)

    else:
        print(f"Unknown test type: {test_type}")
        sys.exit(3)
