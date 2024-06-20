#! /usr/bin/bash

RES_FILE=${1:-summary.csv}

rm -rf "$RES_FILE"
# header
echo "M, K, N, avg_tflops, avg_gbs, max_tflops, max_gbs, min_tflops, min_gbs" > "$RES_FILE"

run_benchmark() {
    local success=false
    local max_attempts=20
    local attempt=0

    while [ "$success" = false ] && [ "$attempt" -lt "$max_attempts" ]; do
        ((attempt++))
        echo "Attempt $attempt: M:$1 K:$2 N:$3"
        
        bash collect.sh "$1" "$2" "$3" | tail -n 1 > temp.txt
        
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
        echo "$1" "$2" "$3" | tee -a "$RES_FILE"
    fi
}

run_benchmark 4096 4096 4096
exit 1
run_benchmark 8192 8192 8192
run_benchmark 1 5120 13824
run_benchmark 1024 28672 8192
run_benchmark 3072 4096 3072
run_benchmark 4 4096 12288
# # # habana shapes
run_benchmark 512 8192 8192
run_benchmark 512 8192 32768
run_benchmark 512 32768 8192
run_benchmark 16384 8192 1024
run_benchmark 16384 1024 8192
run_benchmark 16384 8192 4096
run_benchmark 16384 4096 8192
run_benchmark 4096 16384 8192
run_benchmark 8192 16384 4096
run_benchmark 1024 16384 8192
run_benchmark 8192 16384 1024
run_benchmark 8 128 16384
run_benchmark 8 16384 128
run_benchmark 32768 128 4096
run_benchmark 32768 4096 128
run_benchmark 4096 4096 128
