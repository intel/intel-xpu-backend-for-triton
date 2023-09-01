# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/interpreter/test_interpreter.py

sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/language/assert_helper.py
sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/language/print_helper.py
sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/language/test_annotations.py
sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/language/test_block_pointer.py
sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/language/test_line_info.py
sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/language/test_random.py
sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/language/test_core.py
sed -i '/def test_abs_fp8(in_dtype, device):/ a\    pytest.skip("fp8 is not supported for xpu")' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/language/test_core.py
sed -i 's/MmaLayout(/# &/' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/language/test_core.py
sed -i 's/f.name/&, device_type=device/' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/language/test_core.py
sed -i '/pytest.skip("bfloat16 is only supported on NVGPU with cc >= 80")/ a\    if dtype in ["float8e4b15", "tl.float8e4b15x4", "float8e4nv", "float8e5"]:\n        pytest.skip("fp8 is not supported yet")' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/language/test_core.py
sed -i "/ptx = pgm.asm\['ptx'\]/,/^$/d" ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/language/test_core.py
sed -i '/capability = torch.cuda.get_device_capability.*/i \    check_cuda_only(device)' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/language/test_core.py
# TODO : This following change will affect the test_dot, we need to modify this after the dot is enabled
sed -i -r '/[h|pgm]\.asm\["ptx"\]/,/^$/d' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/language/test_core.py

# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/operators/test_blocksparse.py
# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/operators/test_cross_entropy.py
# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/operators/test_flash_attention.py
# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/operators/test_inductor.py
# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/operators/test_matmul.py

# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/runtime/test_autotuner.py
# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/runtime/test_cache.py
# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/runtime/test_launch.py
# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${HOME}/actions-runner/_work/intel-xpu-backend-for-triton/intel-xpu-backend-for-triton/triton/python/test/unit/runtime/test_subproc.py
