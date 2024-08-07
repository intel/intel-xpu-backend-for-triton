#!/usr/bin/env bash

set -euo pipefail

# $1 is the latest commit id from SPIRV-LLVM-Translator
# $2 is the commit id from Triton's spirv-llvm-translator.conf

if [ "$#" -ne 2 ]; then
    echo "Please provide commit id from Translator and from spirv-llvm-translator.conf. Usage: $0 cid1 cid2"
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
        echo "Compile failed for translator commit $cid"
        continue
	fi

    # build pytorch and IPEX outside of 'test-triton.sh'
    if ./scripts/test-triton.sh --skip-deps; then
        echo "Tests passed for translator commit $cid"
        echo "A newer commit found: $cid"
        FOUND=true
        break
    else
        echo "Tests failed for translator commit $cid"
    fi
done

if [ "$FOUND" = false ]; then
    git restore ./lib/Target/SPIRV/spirv-llvm-translator.conf
fi
