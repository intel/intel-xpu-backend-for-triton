Triton Demo Performace Tuning BKC:


Branches:

[triton](https://github.com/intel/intel-xpu-backend-for-triton/tree/triton_perf_poc)  
[xelta](https://github.com/intel-innersource/libraries.gpu.xetla/tree/jiaxing/collecting_perf/tests/integration/gemm/bf16)
[xelta_stream-k case](https://github.com/intel-innersource/libraries.gpu.xetla/tree/jiaxing/collecting_perf/examples/11_stream_k_gemm)




How to collect data:(PVC MAX 1550 & Basekit 2024.1.0 & libigc1: 1:1.0.24514.15888-igc+releaseinternal1)

`xetla` data:
```
git clone -b jiaxing/collecting_perf https://github.com/intel-innersource/libraries.gpu.xetla.git
cd libraries.gpu.xetla
wget -O demo_xetla_gemm.diff https://raw.githubusercontent.com/intel/intel-xpu-backend-for-triton/triton_perf_poc/demo_xetla_gemm.diff
git apply demo_xetla_gemm.diff
source tools/scripts/env.sh
mkdir build && cd build
cmake ..
make -j
tests/integration/gemm/bf16/gemm_bf16
export SYCL_PROGRAM_COMPILE_OPTIONS=" -vc-codegen -vc-disable-indvars-opt -doubleGRF -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' "
examples/11_stream_k_gemm/stream_k_gemm 
```

`Triton` data:
```
git clone -b triton_perf_poc https://github.com/intel/intel-xpu-backend-for-triton.git
cd intel-xpu-backend-for-triton
scripts/compile-triton.sh --triton  --venv
source .venv/bin/activate
scripts/compile-pytorch-ipex.sh --pinned
cd python/tutorials 
bash collect.sh 4096 4096 4096
```