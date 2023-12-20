SCRIPTS_ROOT_DIR=$(dirname $0)
TRITON_SRC_ROOT_DIR=$(realpath -- $SCRIPTS_ROOT_DIR/../../../..)
# language
for file in ${TRITON_SRC_ROOT_DIR}/python/test/unit/language/*.py; do
  sed -i '/import torch/ a\import intel_extension_for_pytorch' "$file"
done
sed -i 's/cuda/xpu/g' ${TRITON_SRC_ROOT_DIR}/python/test/unit/language/print_helper.py
sed -i -E 's/device=["'\'']cuda["'\'']/device="xpu"/g' ${TRITON_SRC_ROOT_DIR}/python/test/unit/language/test_core.py
sed -i '/def test_abs_fp8(in_dtype, device):/ a\    pytest.skip("fp8 is not supported for xpu")' ${TRITON_SRC_ROOT_DIR}/python/test/unit/language/test_core.py
sed -i 's/MmaLayout(/# &/' ${TRITON_SRC_ROOT_DIR}/python/test/unit/language/test_core.py
sed -i 's/instr_shape=\[/# &/' ${TRITON_SRC_ROOT_DIR}/python/test/unit/language/test_core.py
sed -i 's/f.name/&, device_type=device/' ${TRITON_SRC_ROOT_DIR}/python/test/unit/language/test_core.py
sed -i '/pytest.skip("float8e4nv is only supported on NVGPU with cc >= 90")/ a\    if dtype in  [tl.float8e4b15, tl.float8e4b15x4, tl.float8e4nv, tl.float8e5, "float8e4b15", "tl.float8e4b15x4", "float8e4nv", "float8e5"]:\n        pytest.skip("fp8 is not supported yet")' ${TRITON_SRC_ROOT_DIR}/python/test/unit/language/test_core.py
sed -i "/ptx = pgm.asm\['ptx'\]/,/^$/d" ${TRITON_SRC_ROOT_DIR}/python/test/unit/language/test_core.py
sed -i '/capability = torch.cuda.get_device_capability.*/i \    check_cuda_only(device)' ${TRITON_SRC_ROOT_DIR}/python/test/unit/language/test_core.py
sed -i -r '/h\.asm\["ptx"\]/,/^$/d' ${TRITON_SRC_ROOT_DIR}/python/test/unit/language/test_core.py
# TODO : This following change will affect the test_dot, we need to modify this after the dot is enabled
sed -i -r '/pgm\.asm\["ptx"\]/,/(assert "shared" not in red_code)|(^$)/d' ${TRITON_SRC_ROOT_DIR}/python/test/unit/language/test_core.py
sed -i 's/slice_kernel\[(1,)](XBLOCK=32/slice_kernel\[(1,)](XBLOCK=32, device_type=device/g' ${TRITON_SRC_ROOT_DIR}/python/test/unit/language/test_core.py
python ${SCRIPTS_ROOT_DIR}/case_update.py ${TRITON_SRC_ROOT_DIR}/python/test/unit/language/test_core.py
# operators
for file in ${TRITON_SRC_ROOT_DIR}/python/test/unit/operators/*.py; do
  sed -i '/import torch/ a\import intel_extension_for_pytorch' "$file"
done
sed -i -E 's/device=["'\'']cuda["'\'']/device="xpu"/g' ${TRITON_SRC_ROOT_DIR}/python/test/unit/operators/test_blocksparse.py
sed -i '/capability = torch.cuda.get_device_capability/,+2 s/^/#/' ${TRITON_SRC_ROOT_DIR}/python/test/unit/operators/test_blocksparse.py
sed -i 's/cuda()/xpu()/g' ${TRITON_SRC_ROOT_DIR}/python/test/unit/operators/test_blocksparse.py
sed -i -E 's/device=["'\'']cuda["'\'']/device="xpu"/g' ${TRITON_SRC_ROOT_DIR}/python/test/unit/operators/test_cross_entropy.py
sed -i '/capability = torch.cuda.get_device_capability/,+2 s/^/#/' ${TRITON_SRC_ROOT_DIR}/python/test/unit/operators/test_cross_entropy.py
sed -i 's/cuda/xpu/g' ${TRITON_SRC_ROOT_DIR}/python/test/unit/operators/test_inductor.py
# runtime
for file in ${TRITON_SRC_ROOT_DIR}/python/test/unit/runtime/*.py; do
  sed -i '/import torch/ a\import intel_extension_for_pytorch' "$file"
done
sed -i 's/cuda/xpu/g' ${TRITON_SRC_ROOT_DIR}/python/test/unit/runtime/test_launch.py
# tools
sed -i '/import torch/ a\import intel_extension_for_pytorch' ${TRITON_SRC_ROOT_DIR}/python/test/unit/tools/test_aot.py
