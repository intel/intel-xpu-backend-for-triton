#!/bin/bash

# Get the path of the directory this script is in
WORK_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Find all .py files in the same directory
echo "Modifying test cases under $WORK_DIR from xpu to cuda version..."
for file in "$WORK_DIR"/*.py; do

  # Replace xpu with cuda in each file
  sed -i 's/intel_extension_for_pytorch._inductor.xpu/torch._inductor/g' "$file"
  sed -i 's/XPUAsyncCompile/AsyncCompile/g' "$file"
  sed -i 's/intel_extension_for_pytorch._C import _getCurrentRawStream/torch._C import _cuda_getCurrentRawStream/' "$file"
  sed -i 's/get_xpu_stream/get_cuda_stream/g' "$file"
  sed -i 's/xpu/cuda/g' "$file"

  echo "Modified $file"

done

