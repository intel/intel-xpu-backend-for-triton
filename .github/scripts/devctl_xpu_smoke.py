import torch
import torch.xpu
import triton
import triton.language as tl


@triton.jit
def _increment_kernel(x_ptr, y_ptr, size: tl.constexpr, block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < size
    values = tl.load(x_ptr + offsets, mask=mask, other=0)
    tl.store(y_ptr + offsets, values + 1, mask=mask)


def main():
    if not torch.xpu.is_available():
        raise RuntimeError("No Intel XPU is available in the DevCtl job")

    size = 1024
    source = torch.arange(size, device="xpu", dtype=torch.float32)
    result = torch.empty_like(source)
    _increment_kernel[(triton.cdiv(size, 256),)](source, result, size, 256)
    torch.xpu.synchronize()
    torch.testing.assert_close(result, source + 1)

    print(f"XPU: {torch.xpu.get_device_name(0)}")
    print(f"Triton: {triton.__version__}")
    print("DevCtl XPU Triton smoke passed")


if __name__ == "__main__":
    main()
