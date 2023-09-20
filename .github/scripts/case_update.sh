# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/interpreter/test_interpreter.py
TRITON_ROOT=${HOME}/triton-preci

sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/language/assert_helper.py
sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/language/print_helper.py
sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/language/test_annotations.py
sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/language/test_block_pointer.py
sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/language/test_line_info.py
sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/language/test_random.py
sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/language/test_core.py
sed -i '/def test_abs_fp8(in_dtype, device):/ a\    pytest.skip("fp8 is not supported for xpu")' ${TRITON_ROOT}/triton/python/test/unit/language/test_core.py
sed -i 's/MmaLayout(/# &/' ${TRITON_ROOT}/triton/python/test/unit/language/test_core.py
sed -i 's/f.name/&, device_type=device/' ${TRITON_ROOT}/triton/python/test/unit/language/test_core.py
sed -i '/pytest.skip("float8e4nv is only supported on NVGPU with cc >= 80")/ a\    if dtype in ["float8e4b15", "tl.float8e4b15x4", "float8e4nv", "float8e5"]:\n        pytest.skip("fp8 is not supported yet")' ${TRITON_ROOT}/triton/python/test/unit/language/test_core.py
sed -i "/ptx = pgm.asm\['ptx'\]/,/^$/d" ${TRITON_ROOT}/triton/python/test/unit/language/test_core.py
sed -i '/capability = torch.cuda.get_device_capability.*/i \    check_cuda_only(device)' ${TRITON_ROOT}/triton/python/test/unit/language/test_core.py
sed -i -r '/h\.asm\["ptx"\]/,/^$/d' ${TRITON_ROOT}/triton/python/test/unit/language/test_core.py
# TODO : This following change will affect the test_dot, we need to modify this after the dot is enabled
sed -i -r '/pgm\.asm\["ptx"\]/,/(assert "shared" not in red_code)|(^$)/d' ${TRITON_ROOT}/triton/python/test/unit/language/test_core.py
sed -i 's/slice_kernel\[(1,)](XBLOCK=32/slice_kernel\[(1,)](XBLOCK=32, device_type=device/g' ${TRITON_ROOT}/triton/python/test/unit/language/test_core.py
python ${TRITON_ROOT}/triton/third_party/intel_xpu_backend/.github/scripts/case_update.py ${TRITON_ROOT}/triton/python/test/unit/language/test_core.py

# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/operators/test_blocksparse.py
# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/operators/test_cross_entropy.py
# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/operators/test_flash_attention.py
# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/operators/test_inductor.py
# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/operators/test_matmul.py

# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/runtime/test_autotuner.py
# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/runtime/test_cache.py
# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/runtime/test_launch.py
# sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_ROOT}/triton/python/test/unit/runtime/test_subproc.py
