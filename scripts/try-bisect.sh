#!/usr/bin/env bash

set -euo pipefail

TRITON_PROJ="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )"
SCRIPTS_DIR="$TRITON_PROJ/scripts"

bash "$SCRIPTS_DIR/compile-triton.sh"
if [ $? -ne 0 ]; then
    exit 125  # skip commit if build failed
fi

bash "$SCRIPTS_DIR/test-pytorch.sh" "$@"
if [ $? -ne 0 ]; then
    exit 1  # mark bad commit
else
    exit 0  # mark good commit
fi
