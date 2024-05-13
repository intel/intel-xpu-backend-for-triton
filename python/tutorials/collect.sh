#!/bin/bash

M=${1:-4096}
K=${2:-4096}
N=${3:-4096}

echo -e "================M: $M==============K: $K==============N: $N==============="

# basekit
source /opt/intel/oneapi/setvars.sh --force

# store result
rm -rf result.csv result.txt


# update shape size in driver.py and 09-experimental-block-pointer.py
sed -i "s/x_vals=.*/x_vals=[[$M, $K, $N]],/g" 09-experimental-block-pointer.py
sed -i "s/float M = .*/float M = $M, K = $K, N = $N;/g" ../../third_party/intel/backend/driver.py

TRITON_INTEL_ENABLE_BLOCK_PTR=1 \
IGC_VISAOptions=" -TotalGRFNum 256 -enableBCR -nolocalra -printregusage -DPASTokenReduction -enableHalfLSC" \
IGC_ForcePrefetchToL1Cache=1 \
IGC_VATemp=1 \
UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0 \
IGC_DisableLoopUnroll=1 \
IGC_EnableVISANoSchedule=1 \
python 09-experimental-block-pointer.py 2>&1 | tee result.txt

Triton_tflops_max=`grep "Triton Peak TFlops" result.txt | awk '{print $NF}' | awk 'BEGIN{max=0} {if ($1>max) max=$1} END{print max}'`
Triton_tflops_min=`grep "Triton Peak TFlops" result.txt | awk '{print $NF}' | awk 'BEGIN{min=9999} {if ($1<min) min=$1} END{print min}'`
Triton_tflops_avg=$(grep "Triton Peak TFlops" result.txt | awk '{print $NF}' | awk -v max="$Triton_tflops_max" -v min="$Triton_tflops_min" '{sum+=$1} END{print (sum-max-min)/NR}')

Triton_gbs_max=`grep "Triton Peak HBM" result.txt | awk '{print $NF}' | awk 'BEGIN{max=0} {if ($1>max) max=$1} END{print max}'`
Triton_gbs_min=`grep "Triton Peak HBM" result.txt | awk '{print $NF}' | awk 'BEGIN{min=9999} {if ($1<min) min=$1} END{print min}'`
Triton_gbs_avg=$(grep "Triton Peak HBM" result.txt | awk '{print $NF}' | awk -v max="$Triton_gbs_max" -v min="$Triton_gbs_min" '{sum+=$1} END{print (sum-max-min)/NR}')    

echo -e "=================================== Result ========================================"
echo "M,K,N,avg_tflops,avg_gbs,max_tflops,max_gbs,min_tflops,min_gbs" | tee result.csv
echo $M,$K,$N,$Triton_tflops_avg,$Triton_gbs_avg,$Triton_tflops_max,$Triton_gbs_max,$Triton_tflops_min,$Triton_gbs_min | tee -a result.csv    
