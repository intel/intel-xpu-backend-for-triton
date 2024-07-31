##!/bin/bash

RES_FILE=${1:-summary.csv}

rm -rf "$RES_FILE"
# header
echo "Z, H, N_CTX, D_HEAD, avg_tflops, max_tflops, min_tflops" > "$RES_FILE"

run_benchmark() {
    local success=false
    local max_attempts=20
    local attempt=0

    while [ "$success" = false ] && [ "$attempt" -lt "$max_attempts" ]; do
        ((attempt++))
        echo "Attempt $attempt: Z:$1 H:$2 N_CTX:$3 D_HEAD:$4"
        
        bash collect.sh "$1" "$2" "$3" "$4" | tail -n 1 > temp.txt
        
        local exit_status=${PIPESTATUS[0]}
        echo "Exit status: $exit_status"
        
        if [ "$exit_status" -eq 0 ]; then
            cat temp.txt | tee -a "$RES_FILE"
            success=true
        else
            echo "Retry..."
            sleep 1
        fi
    done

    rm -f temp.txt

    if [ "$success" = false ]; then
        echo "Benchmark failed after $max_attempts attempts."
        echo "$1" "$2" "$3" "$4" | tee -a "$RES_FILE"
    fi
}

run_benchmark 4 48 1024 64
run_benchmark 32 32 512 64
run_benchmark 16 32 1024 64
run_benchmark 8 32 2048 64
run_benchmark 4 32 4096 64
run_benchmark 2 32 8192 64
run_benchmark 1 32 16384 64
#run_benchmark 32 16 512 128
#run_benchmark 16 16 1024 128
#run_benchmark 8 16 2048 128
#run_benchmark 4 16 4096 128
#run_benchmark 2 16 8192 128
#run_benchmark 1 16 16384 128
