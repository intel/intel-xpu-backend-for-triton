#!/bin/bash

set -e

IGC_VectorizerAllowEXP2=1 IGC_VectorizerAllowMAXNUM=1 IGC_VectorizerAllowWAVEALL=1 IGC_VectorizerAllowCMP=1 IGC_VectorizerAllowSelect=1 TRITON_INTEL_ENABLE_BLOCK_IO_STORE_ON_REGULAR_PTR=1 TRITON_INTEL_ONE_MATRIX_PER_LOAD_BT=1 IGC_DisableCodeScheduling=0 \
  python run_llm_inductor_greedy.py -m /data/liyang/hf_models/llama-3.1-8b/0e9e39f249a16976918f6564b8830bc894c89659 --max-new-tokens 128 --input-tokens 1024 --num-warmup 2 --num-iter 7 --compile --profile | tee llama31.compile.xpu.profile.log

echo "llama profiling log is stored into $PWD/llama31.compile.xpu.profile.log"
