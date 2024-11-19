import torch
aten = torch.ops.aten

import pytest 

import triton
import triton.language as tl 


def patch_kernel(template, to_replace):
    kernel = triton.JITFunction(template.fn)
    for key, value in to_replace.items():
        kernel.src = kernel.src.replace(key, value)
    return kernel
    
def test_divide(device):
    # regression test for various division cases 

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
        tmp5 = tl.where((tmp0 < 0) != (tmp2 < 0), tl.where(tmp0 % tmp2 != 0, tmp0 // tmp2 - 1, tmp0 // tmp2), tmp0 // tmp2)
        tmp6 = tmp0 // tmp2
        tl.store(out_ptr0 + (x0), tmp4, xmask)
        tl.store(out_ptr1 + (x0), tmp5, xmask)
        tl.store(out_ptr2 + (x0), tmp6, xmask)
        tl.store(out_ptr3 + (x0), tmp4, xmask)
        tl.store(out_ptr4 + (x0), tmp5, xmask)

    torch.manual_seed(0)

    def launch_triton(a, b):
        output0 = torch.empty_like(a)
        output1 = torch.empty_like(a)
        output2 = torch.empty_like(a)
        output3 = torch.empty_like(a)
        output4 = torch.empty_like(a)

        n_elements = output0.numel()

        grid = lambda meta: (triton.cdiv(n_elements, meta['XBLOCK']), )
 
        divide_kernel[grid](a, b, output0, output1, output2, output3, output4, n_elements, XBLOCK=128)

        return (output0, output1, output2, output3, output4)
    
    def launch_torch(a, b):
            return (
                aten.div(a, b, rounding_mode=None),
                aten.div(a, b, rounding_mode="floor"),
                aten.div(a, b, rounding_mode="trunc"),
                a / b,
                a // b,
            )

    a = torch.randint(2**32, 2**40, [100, 100], device=device)
    b = torch.randint(-10, -1, [100, 100], device=device)

    for iter in range(100):
        triton_result = launch_triton(a, b)
        torch_result = launch_torch(a, b)

        for i in range(5):
            torch.testing.assert_close(triton_result[i], torch_result[i], check_dtype=False, msg=lambda msg: f"Iteration {iter}, {i} failed\n{msg}")

    
