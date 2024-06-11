#!/bin/bash
set -e
Z=${1:-4}
H=${2:-48}
N_CTX=${3:-1024}
D_HEAD=${4:-64}

echo -e "================Z: $Z==============H: $H==============N_CTX: $N_CTX=====D_HEAD: $D_HEAD=========="

# basekit
source /opt/intel/oneapi/setvars.sh --force

# store result
rm -rf result.csv result.txt

# update shape size in driver.py and 06-fused-attention.forward.py
sed -i "s/x_vals=.*/x_vals=[[$Z, $H, $N_CTX, $D_HEAD]],/g" 06-fused-attention.forward.py
sed -i "s/float Z = .*/float Z = $Z, H = $H, N_CTX = $N_CTX, D_HEAD = $D_HEAD;/g" ../../third_party/intel/backend/driver.py

# clean Triton cache
rm -rf ./tt_cache
export TRITON_CACHE_DIR=./tt_cache
# clean IGC cache
export NEO_CACHE_PERSISTENT=0

TRITON_INTEL_ENABLE_BLOCK_PTR=1 \
TRITON_DISABLE_LINE_INFO=1 \
IGC_VISAOptions=" -TotalGRFNum 256 -enableBCR -nolocalra -printregusage -DPASTokenReduction -enableHalfLSC" \
IGC_ForcePrefetchToL1Cache=1 \
IGC_VATemp=1 \
UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0 \
IGC_DisableLoopUnroll=1 \
NEO_CACHE_PERSISTENT=0 \
python 06-fused-attention.forward.py 2>&1 | tee result.txt

if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    exit 1
fi

Triton_tflops_max=`grep "Triton Peak TFlops" result.txt | awk '{print $NF}' |  tail -n10 | awk 'BEGIN{max=0} {if ($1>max) max=$1} END{print max}'`
Triton_tflops_min=`grep "Triton Peak TFlops" result.txt | awk '{print $NF}'  | tail -n10 | awk 'BEGIN{min=9999} {if ($1<min) min=$1} END{print min}'`
Triton_tflops_avg=$(grep "Triton Peak TFlops" result.txt | awk '{print $NF}'  | tail -n10 | awk -v max="$Triton_tflops_max" -v min="$Triton_tflops_min" '{sum+=$1} END{print (sum-max-min)/(NR-2)}')

echo -e "=================================== Result ========================================"
echo "Z, H, N_CTX, D_HEAD, avg_tflops, max_tflops, min_tflops" | tee result.csv
echo $Z, $H, $N_CTX, $D_HEAD, $Triton_tflops_avg, $Triton_tflops_max, $Triton_tflops_min | tee -a result.csv
