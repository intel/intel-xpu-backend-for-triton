"""
Helper module containing Triton kernels for test_debug.py.
These kernels are separated so they can be called from subprocesses.
"""
import torch
import triton
import triton.language as tl
import sys
import os


def run_device_assert_kernel(cond, mask, opt_flag, jit_flag, device):

    @triton.jit(debug=jit_flag)
    def _kernel(COND: tl.constexpr, MASK: tl.constexpr):
        tl.device_assert(COND, 'test', mask=MASK)

    kwargs = {}
    if opt_flag is not None:
        kwargs["debug"] = opt_flag

    try:
        _kernel[(1, )](cond, mask, **kwargs)
        getattr(torch, device).synchronize()
        return 0
    except RuntimeError:
        return 1
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return 2


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
        env_var = sys.argv[7] == "True"

        os.environ["TRITON_DEBUG"] = str(int(env_var))
        triton.knobs.refresh_knobs()
        exit_code = run_device_assert_kernel(cond, mask, opt_flag, jit_flag, device)
        sys.exit(exit_code)

    else:
        print(f"Unknown test type: {test_type}")
        sys.exit(3)
