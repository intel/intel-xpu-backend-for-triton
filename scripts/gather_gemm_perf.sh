#!/bin/bash
cd python/tutorials

# Define the output file
OUT_FILE="gemm_results.csv"
rm $OUT_FILE

# Define the grid as an array of strings
grid=(
    "512 8192 8192"
    "512 8192 32768"
    "512 32768 8192"
    "16384 8192 1024"
    "16384 1024 8192"
    "16384 8192 4096"
    "16384 4096 8192"
    "4096 16384 8192"
    "8192 16384 4096"
    "1024 16384 8192"
    "8192 16384 1024"
)

# Generate header
echo "M,K,N,avg_tflops,avg_gbs,max_tflops,max_gbs,min_tflops,min_gbs" | tee $OUT_FILE

# Iterate over the grid
for i in "${grid[@]}"; do
    # Split the string into m, n, k
    read -r m n k <<< "$i"

    # Execute the collect.sh script with m, n, k as arguments
    for a in $(seq 1 3); do
        if (bash collect.sh "$m" "$n" "$k"); then
            break
        else
            echo "Filed with command, retrying $a"

        # if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        #     echo "Error in the script"
        #     continue
        # fi
        fi
    done

    Triton_tflops_max=`grep "Triton Peak TFlops" result.txt | awk '{print $NF}' |  tail -n10 | awk 'BEGIN{max=0} {if ($1>max) max=$1} END{print max}'`
    Triton_tflops_min=`grep "Triton Peak TFlops" result.txt | awk '{print $NF}'  | tail -n10 | awk 'BEGIN{min=9999} {if ($1<min) min=$1} END{print min}'`
    Triton_tflops_avg=$(grep "Triton Peak TFlops" result.txt | awk '{print $NF}'  | tail -n10 | awk -v max="$Triton_tflops_max" -v min="$Triton_tflops_min" '{sum+=$1} END{print (sum-max-min)/(NR-2)}')

    Triton_gbs_max=`grep "Triton Peak HBM" result.txt | awk '{print $NF}'  | tail -n10 | awk 'BEGIN{max=0} {if ($1>max) max=$1} END{print max}'`
    Triton_gbs_min=`grep "Triton Peak HBM" result.txt | awk '{print $NF}'  | tail -n10 | awk 'BEGIN{min=9999} {if ($1<min) min=$1} END{print min}'`
    Triton_gbs_avg=$(grep "Triton Peak HBM" result.txt | awk '{print $NF}'  | tail -n10 | awk -v max="$Triton_gbs_max" -v min="$Triton_gbs_min" '{sum+=$1} END{print (sum-max-min)/(NR-2)}')

    echo -e "=================================== Result ========================================"
    echo $m,$k,$n,$Triton_tflops_avg,$Triton_gbs_avg,$Triton_tflops_max,$Triton_gbs_max,$Triton_tflops_min,$Triton_gbs_min >> $OUT_FILE

    done

# Display the contents of the output file
cat "$OUT_FILE"
