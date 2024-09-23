#!/usr/bin/env bash

set -uo pipefail

# $1 is the latest commit id from SPIRV-LLVM-Translator
# $2 is the commit id from Triton's spirv-llvm-translator.conf

if [ "$#" -ne 3 ]; then
    echo "Please provide commit id from Translator and from spirv-llvm-translator.conf and the summary log. Usage: $0 cid1 cid2 log"
    exit 1
fi

if [ "$1" == "$2" ]; then
    echo "No updates needed"
    exit 0
fi

BASE=$(cd $(dirname "$0")/../.. && pwd)
TRITON_PROJ=${BASE}/intel-xpu-backend-for-triton

# get all commit ids
COMMIT_IDS=$(git -C $TRITON_PROJ/external/SPIRV-LLVM-Translator log --format="%H" "$2".."$1")

# check every commit ids
cd $TRITON_PROJ
FOUND=false
for cid in $COMMIT_IDS; do
    echo "$cid" > ./lib/Target/SPIRV/spirv-llvm-translator.conf
    if ! ./scripts/compile-triton.sh --clean; then
        echo "Triton compile failed for translator commit $cid"
        continue
    fi

    # execute default tests
    ./scripts/test-triton.sh --skip-pytorch-install --core 2>&1 | tee tmp.log
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "Tests passed for translator commit $cid"
        echo "A newer commit found: $cid"
        FOUND=true
        break
    else
        echo -e "\nTests failed for translator commit $cid:" | tee -a "$3"
        awk '/=+ FAILURES =+/, /=+ short test summary info =+/' tmp.log >> "$3"
    fi
done

if [ "$FOUND" = false ]; then
    git restore ./lib/Target/SPIRV/spirv-llvm-translator.conf
fi
