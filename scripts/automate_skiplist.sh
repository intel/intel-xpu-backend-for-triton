#!/usr/bin/env bash
set -euo pipefail

# The target platform
PLATFORM=$1

if [ "$#" -ne 1 ]; then
   echo "Please provide the platform name. Usage: $0 arg"
   exit 1
fi

BASE=$(cd $(dirname "$0")/../.. && pwd)
TRITON_PROJ=${BASE}/intel-xpu-backend-for-triton

# Run core test, regression test and interpreter test in mode unskip
. ${TRITON_PROJ}/scripts/test-triton.sh --core --interpreter --unskip --reports

# Parse logs and get all failed cases for all categories
TXT_DIR=${TRITON_PROJ}/scripts/skiplist/${PLATFORM}
mkdir -p "${TXT_DIR}"

for xml_file in ${TRITON_TEST_REPORTS_DIR}/*.xml; do
    file_name=$(basename ${xml_file} .xml)
    OUT_FILE=${TXT_DIR}/$file_name.txt

    python ${TRITON_PROJ}/scripts/get_failed_cases.py ${xml_file} ${OUT_FILE}
done
