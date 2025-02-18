import torch
from torch._inductor.utils import fresh_inductor_cache
import os
import sys


def test_case():
    with fresh_inductor_cache():

        @torch.compile
        def my_add(a, b):
            return a + b

        device = "xpu"
        a = torch.rand(2, 3, device=device)
        b = torch.rand(2, 3, device=device)
        _ = my_add(a, b)

        triton_cache_dir = os.environ["TRITON_CACHE_DIR"]

        for m_name in list(sys.modules.keys()):
            if m_name.startswith("torch._inductor.runtime.compile_tasks."):
                m = sys.modules[m_name]
                for attr_name in m.__dict__.keys():
                    if attr_name.startswith("triton_poi"):
                        kernel = getattr(m, attr_name)
                        kernel.launchers = []
                        kernel.compile_results = []
                        # make sure `spirv_utils` destructor is called
                        _mod = sys.modules['triton.runtime.driver']
                        del getattr(_mod, 'driver').active.utils
                        import gc
                        gc.collect()
                del sys.modules[m_name]

    print("triton cache dir:", triton_cache_dir)
    assert not os.path.exists(triton_cache_dir), f"{os.listdir(triton_cache_dir)}"
