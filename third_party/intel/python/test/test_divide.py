# flake8: noqa: F821, F841
import os

import torch
import pytest

import triton
import triton.language as tl


@pytest.fixture(autouse=True)
def triton_predicated_load(monkeypatch):
    monkeypatch.setenv("TRITON_INTEL_PREDICATED_LOAD", "1")
    yield


aten = torch.ops.aten


def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    for key, value in to_replace.items():
        kernel._unsafe_update_src(kernel.src.replace(key, value))
    return kernel


@pytest.mark.parametrize("float_div", [True, False])
@pytest.mark.parametrize("floor", [True, False])
@pytest.mark.parametrize("trunc", [True, False])
def test_divide(float_div, floor, trunc, device):
    # regression test for various division cases
    assert os.environ["TRITON_INTEL_PREDICATED_LOAD"] == "1"

    @triton.jit
    def divide_kernel(a, b, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK: tl.constexpr):
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(a + (x0), xmask)
        tmp2 = tl.load(b + (x0), xmask)
        # custom bits
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 / tmp3
        tmp5 = tl.where((tmp0 < 0) != (tmp2 < 0), tl.where(tmp0 % tmp2 != 0, tmp0 // tmp2 - 1, tmp0 // tmp2),
                        tmp0 // tmp2)
        tmp6 = tmp0 // tmp2
        GENERATE_OUTPUTS_HERE

    torch.manual_seed(0)

    outputs_float_div = "tl.store(out_ptr0 + (x0), tmp4, xmask)\n    tl.store(out_ptr3 + (x0), tmp4, xmask)" if float_div else ""
    outputs_floor = "    tl.store(out_ptr1 + (x0), tmp5, xmask)\n    tl.store(out_ptr4 + (x0), tmp5, xmask)" if floor else ""
    outputs_trunc = "    tl.store(out_ptr2 + (x0), tmp6, xmask)" if trunc else ""

    divide_kernel = patch_kernel(divide_kernel,
                                 {"GENERATE_OUTPUTS_HERE": f"{outputs_float_div}\n{outputs_floor}\n{outputs_trunc}"})

    def launch_triton(a, b):
        output0 = torch.zeros_like(a)
        output1 = torch.zeros_like(a)
        output2 = torch.zeros_like(a)
        output3 = torch.zeros_like(a)
        output4 = torch.zeros_like(a)

        n_elements = output0.numel()

        grid = lambda meta: (triton.cdiv(n_elements, meta['XBLOCK']), )

        divide_kernel[grid](a, b, output0, output1, output2, output3, output4, n_elements, XBLOCK=128)

        return (output0, output1, output2, output3, output4)

    def launch_torch(a, b):
        return (
            aten.div(a, b, rounding_mode=None) if float_div is True else torch.zeros_like(a),
            aten.div(a, b, rounding_mode="floor") if floor is True else torch.zeros_like(a),
            aten.div(a, b, rounding_mode="trunc") if trunc is True else torch.zeros_like(a),
            a / b if float_div is True else torch.zeros_like(a),
            a // b if floor is True else torch.zeros_like(a),
        )

    a = torch.randint(2**32, 2**40, [100, 100], device=device)
    b = torch.randint(-10, -1, [100, 100], device=device)

    for iter in range(100):
        triton_result = launch_triton(a, b)
        torch_result = launch_torch(a, b)

        for i in range(5):
            torch.testing.assert_close(
                triton_result[i], torch_result[i], check_dtype=False, msg=lambda msg:
                f"Float: {float_div}, Floor: {floor}, Trunc: {trunc}\nIteration {iter}, {i} failed\n{msg}")
