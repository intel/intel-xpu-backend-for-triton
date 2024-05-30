#!/bin/bash
set -e
M=${1:-4096}
K=${2:-4096}
N=${3:-4096}

echo -e "================M: $M==============K: $K==============N: $N==============="

# basekit
source /opt/intel/oneapi/setvars.sh --force

# store result
rm -rf result.csv result.txt

if [ $M  -le 8 ]
then
    export TRITON_INTEL_SKIP_PREFETCH_A=1
fi

# update shape size in driver.py and 09-experimental-block-pointer.py
sed -i "s/x_vals=.*/x_vals=[[$M, $K, $N]],/g" 09-experimental-block-pointer.py
sed -i "s/float M = .*/float M = $M, K = $K, N = $N;/g" ../../third_party/intel/backend/driver.py

# default
BLOCK_SIZE_M=256
BLOCK_SIZE_N=256
BLOCK_SIZE_K=32
GROUP_SIZE_M=4
num_stages=4
num_warps=32

# Small M
if [ $M  -le 8 ]
then
    BLOCK_SIZE_M=8
    BLOCK_SIZE_N=512
    BLOCK_SIZE_K=64
    GROUP_SIZE_M=1
    num_stages=4
    num_warps=32
fi

if [ $M  = 4096 ] && [ $K = 4096 ]	&& [ $N = 128 ]
then
    BLOCK_SIZE_M=64
    BLOCK_SIZE_N=128
    BLOCK_SIZE_K=32
    GROUP_SIZE_M=4
    num_stages=4
    num_warps=32
fi

echo "===Using: BLOCK_SIZE_M: $BLOCK_SIZE_M, BLOCK_SIZE_N: $BLOCK_SIZE_N, BLOCK_SIZE_K: $BLOCK_SIZE_K, GROUP_SIZE_M: $GROUP_SIZE_M, num_stages: $num_stages, num_warps: $num_warps====="
sed -i "s/triton.Config({'BLOCK_SIZE_M'.*/triton.Config({'BLOCK_SIZE_M': $BLOCK_SIZE_M, 'BLOCK_SIZE_N': $BLOCK_SIZE_N, 'BLOCK_SIZE_K': $BLOCK_SIZE_K, 'GROUP_SIZE_M': $GROUP_SIZE_M}, num_stages=$num_stages, num_warps=$num_warps),/g" 09-experimental-block-pointer.py

# clean Triton cache
rm -rf ./tt_cache
export TRITON_CACHE_DIR=./tt_cache
# clean IGC cache
export NEO_CACHE_PERSISTENT=0

TRITON_INTEL_ENABLE_BLOCK_PTR=1 \
TRITON_INTEL_PREFETCH_DISTANCE=2 \
TRITON_INTEL_SPLIT_BARRIER=1 \
IGC_VISAOptions=" -TotalGRFNum 256 -enableBCR -nolocalra -printregusage -DPASTokenReduction -enableHalfLSC -abiver 2" \
IGC_ForcePrefetchToL1Cache=1 \
IGC_VATemp=1 \
UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0 \
IGC_DisableLoopUnroll=1 \
python 09-experimental-block-pointer.py 2>&1 | tee result.txt

if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    exit 1
fi

Triton_tflops_max=`grep "Triton Peak TFlops" result.txt | awk '{print $NF}' |  tail -n10 | awk 'BEGIN{max=0} {if ($1>max) max=$1} END{print max}'`
Triton_tflops_min=`grep "Triton Peak TFlops" result.txt | awk '{print $NF}'  | tail -n10 | awk 'BEGIN{min=9999} {if ($1<min) min=$1} END{print min}'`
Triton_tflops_avg=$(grep "Triton Peak TFlops" result.txt | awk '{print $NF}'  | tail -n10 | awk -v max="$Triton_tflops_max" -v min="$Triton_tflops_min" '{sum+=$1} END{print (sum-max-min)/(NR-2)}')

Triton_gbs_max=`grep "Triton Peak HBM" result.txt | awk '{print $NF}'  | tail -n10 | awk 'BEGIN{max=0} {if ($1>max) max=$1} END{print max}'`
Triton_gbs_min=`grep "Triton Peak HBM" result.txt | awk '{print $NF}'  | tail -n10 | awk 'BEGIN{min=9999} {if ($1<min) min=$1} END{print min}'`
Triton_gbs_avg=$(grep "Triton Peak HBM" result.txt | awk '{print $NF}'  | tail -n10 | awk -v max="$Triton_gbs_max" -v min="$Triton_gbs_min" '{sum+=$1} END{print (sum-max-min)/(NR-2)}')

echo -e "=================================== Result ========================================"
echo "M, K, N, avg_tflops, avg_gbs, max_tflops, max_gbs, min_tflops, min_gbs" | tee result.csv
echo $M, $K, $N, $Triton_tflops_avg, $Triton_gbs_avg, $Triton_tflops_max, $Triton_gbs_max, $Triton_tflops_min, $Triton_gbs_min | tee -a result.csv    
