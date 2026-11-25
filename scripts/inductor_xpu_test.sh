#!/bin/bash
# This script work for xpu / cuda device inductor tests

SUITE=${1:-huggingface}     # huggingface / torchbench / timm_models
DT=${2:-float32}            # float32 / float16 / amp_bf16 / amp_fp16
MODE=${3:-inference}        # inference / training
SCENARIO=${4:-accuracy}     # accuracy / performance
DEVICE=${5:-xpu}            # xpu / cuda
CARD=${6:-0}                # 0 / 1 / 2 / 3 ...
SHAPE=${7:-static}          # static / dynamic
NUM_SHARDS=${8}             # num test shards
SHARD_ID=${9}               # shard id
MODEL_ONLY=${10}            # GoogleFnet / T5Small

WORKSPACE=${WORKSPACE:-$PWD}
LOG_DIR=$WORKSPACE/inductor_log/${SUITE}/${DT}
mkdir -p ${LOG_DIR}
LOG_NAME=inductor_${SUITE}_${DT}_${MODE}_${DEVICE}_${SCENARIO}

Model_only_extra=""
if [[ -n "$MODEL_ONLY" ]]; then
    echo "Testing model ${MODEL_ONLY}"
    Model_only_extra="--only ${MODEL_ONLY}"
fi

if [[ $MODE == "inference" ]]; then
    # For PT >= 2.1
    # Remove --freezing cause feature not ready
    Mode_extra="--inference"
fi

if [[ $MODE == "inference-with-freezing" ]]; then
    Mode_extra="--inference --freezing "
fi

if [[ $MODE == "training" ]]; then
    echo "Testing with training mode."
    Mode_extra="--training "
fi

Shape_extra=""
if [[ $SHAPE == "dynamic" ]]; then
    echo "Testing with dynamic shapes."
    Shape_extra="--dynamic-shapes --dynamic-batch-only "
fi

partition_flags=""
if [[ -n "$NUM_SHARDS" && -n "$SHARD_ID" ]]; then
  partition_flags="--total-partitions $NUM_SHARDS --partition-id $SHARD_ID "
fi

if (( $EUID == 0 )); then
    ulimit -n 1048576
fi

# TEST BRANCH ONLY: issue-style performance runs (batch size 2, cold-start latency), repeated sequentially
# with fresh Triton/Inductor caches; both the compiled (Triton) and eager times are logged per repetition.
if [[ $SCENARIO == "performance" && -n "$MODEL_ONLY" ]]; then
    case $DT in
        amp_bf16) dt_flags="--amp --amp-dtype=bfloat16" ;;
        amp_fp16) dt_flags="--amp --amp-dtype=float16" ;;
        *) dt_flags="--${DT}" ;;
    esac
    BS2_REPS=${BS2_REPS:-6}
    BS2_ITERS=${BS2_ITERS:-30}
    SUMMARY=${LOG_DIR}/bs2_summary.csv
    echo "model,dtype,mode,rep,host,compiled_ms,eager_ms,speedup" > ${SUMMARY}
    for rep in $(seq 1 ${BS2_REPS}); do
        export TRITON_CACHE_DIR=${WORKSPACE}/.bs2_cache/triton_${rep}
        export TORCHINDUCTOR_CACHE_DIR=${WORKSPACE}/.bs2_cache/inductor_${rep}
        rm -rf ${TRITON_CACHE_DIR} ${TORCHINDUCTOR_CACHE_DIR}
        REP_CSV=${LOG_DIR}/${LOG_NAME}_rep${rep}.csv
        ZE_AFFINITY_MASK=${CARD} python benchmarks/dynamo/${SUITE}.py --${SCENARIO} ${dt_flags} -d${DEVICE} -n${BS2_ITERS} ${Mode_extra} ${Shape_extra} ${partition_flags} ${Model_only_extra} --cold-start-latency --batch-size=2 --backend=inductor --timeout=4800 --output=${REP_CSV} 2>&1 | tee ${LOG_DIR}/${LOG_NAME}_rep${rep}.log
        awk -F, -v m="${MODEL_ONLY}" -v d="${DT}" -v mode="${MODE}" -v r="${rep}" -v h="$(hostname)" \
            'NR>1 && $2!="geomean" && $4+0>0 {sp=$4; lat=$5} END{printf "%s,%s,%s,%s,%s,%.4f,%.4f,%.4f\n", m, d, mode, r, h, lat, lat*sp, sp}' \
            ${REP_CSV} >> ${SUMMARY}
    done
    # keep the canonical names so the later CI steps find a result
    cp ${LOG_DIR}/${LOG_NAME}_rep${BS2_REPS}.csv ${LOG_DIR}/${LOG_NAME}.csv
    cp ${LOG_DIR}/${LOG_NAME}_rep${BS2_REPS}.log ${LOG_DIR}/${LOG_NAME}.log
    cat ${SUMMARY}
    exit 0
fi

if [[ $DT == "amp_bf16" ]]; then
    ZE_AFFINITY_MASK=${CARD} python benchmarks/dynamo/${SUITE}.py --${SCENARIO} --amp --disable-cudagraphs -d${DEVICE} -n10 --dashboard ${Mode_extra} ${Shape_extra} ${partition_flags} ${Model_only_extra} --backend=inductor --timeout=4800 --output=${LOG_DIR}/${LOG_NAME}.csv 2>&1 | tee ${LOG_DIR}/${LOG_NAME}.log
elif [[ $DT == "amp_fp16" ]]; then
    export INDUCTOR_AMP_DT=float16
    ZE_AFFINITY_MASK=${CARD} python benchmarks/dynamo/${SUITE}.py --${SCENARIO} --amp --disable-cudagraphs -d${DEVICE} -n10 --dashboard ${Mode_extra} ${Shape_extra} ${partition_flags} ${Model_only_extra}  --backend=inductor --timeout=4800 --output=${LOG_DIR}/${LOG_NAME}.csv 2>&1 | tee ${LOG_DIR}/${LOG_NAME}.log
else
    ZE_AFFINITY_MASK=${CARD} python benchmarks/dynamo/${SUITE}.py --${SCENARIO} --${DT} --disable-cudagraphs -d${DEVICE} -n10 --dashboard ${Mode_extra} ${Shape_extra} ${partition_flags} ${Model_only_extra}  --backend=inductor --timeout=4800 --output=${LOG_DIR}/${LOG_NAME}.csv 2>&1 | tee ${LOG_DIR}/${LOG_NAME}.log
fi
