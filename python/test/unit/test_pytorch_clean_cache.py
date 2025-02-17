import torch
from torch._inductor.utils import fresh_inductor_cache
import os


def test_case():

    @torch.compile
    def my_add(a, b):
        return a + b

    device = "xpu"
    a = torch.rand(2, 3, device=device)
    b = torch.rand(2, 3, device=device)
    _ = my_add(a, b)


with fresh_inductor_cache():
    test_case()
    triton_cache_dir = os.environ["TRITON_CACHE_DIR"]

print("triton cache dir:", triton_cache_dir)
assert not os.path.exists(triton_cache_dir), f"{os.listdir(triton_cache_dir)}"
