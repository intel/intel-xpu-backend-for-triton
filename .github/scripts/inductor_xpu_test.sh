#! /bin/bash
# This script work for xpu / cuda device inductor tests

SUITE=${1:-huggingface}     # huggingface / torchbench / timm_models
DT=${2:-float32}            # float32 / float16 / amp_bf16 / amp_fp16
MODE=${3:-inference}        # inference / training
SCENARIO=${4:-accuracy}     # accuracy / performance
DEVICE=${5:-xpu}            # xpu / cuda
CARD=${6:-0}                # 0 / 1 / 2 / 3 ...
SHAPE=${7:-static}          # static / dynamic

WORKSPACE=`pwd`
LOG_DIR=$WORKSPACE/inductor_log/${SUITE}/${DT}
mkdir -p ${LOG_DIR}
LOG_NAME=inductor_${SUITE}_${DT}_${MODE}_${DEVICE}_${SCENARIO}

Mode_extra=""
if [[ $MODE == "training" ]]; then
    echo "Testing with training mode."
    Mode_extra="--training "
fi

Shape_extra=""
if [[ $SHAPE == "dynamic" ]]; then
    echo "Testing with dynamic shapes."
    Shape_extra="--dynamic-shapes --dynamic-batch-only "
fi

ulimit -n 1048576
if [[ $DT == "amp_bf16" ]] || [[ $DT == "amp_fp16" ]]; then
    ZE_AFFINITY_MASK=${CARD} python benchmarks/dynamo/${SUITE}.py --${SCENARIO} --amp -d${DEVICE} -n50 --no-skip --dashboard ${Mode_extra} ${Shape_extra} --backend=inductor --timeout=4800 --output=${LOG_DIR}/${LOG_NAME}.csv 2>&1 | tee ${LOG_DIR}/${LOG_NAME}.log
else
    ZE_AFFINITY_MASK=${CARD} python benchmarks/dynamo/${SUITE}.py --${SCENARIO} --${DT} -d${DEVICE} -n50 --no-skip --dashboard ${Mode_extra} ${Shape_extra} --backend=inductor --timeout=4800 --output=${LOG_DIR}/${LOG_NAME}.csv 2>&1 | tee ${LOG_DIR}/${LOG_NAME}.log
fi
