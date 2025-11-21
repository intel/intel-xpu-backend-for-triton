#!/usr/bin/env bash

set -euo pipefail

gh auth status

echo "**** Download nightly builds. ****"
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
RUN_ID=$(gh run list -w "Triton wheels" -R intel/intel-xpu-backend-for-triton --json databaseId,conclusion | jq -r '[.[] | select(.conclusion=="success")][0].databaseId')
TEMP_DIR=$(mktemp -d)
WHEEL_PATTERN="wheels-pytorch-py${PYTHON_VERSION}*"
gh run download $RUN_ID \
--repo intel/intel-xpu-backend-for-triton \
--pattern "$WHEEL_PATTERN" \
--dir $TEMP_DIR

if [ -d "wheels" ]; then
  rm -rf "wheels"
fi

mkdir wheels
cp $TEMP_DIR/$WHEEL_PATTERN/torch-*.whl wheels/
cp $TEMP_DIR/$WHEEL_PATTERN/triton-*.whl wheels/

pip install wheels/*

# Sample version - 3.0.0+git5297206
TRITON_COMMIT=$(python -c "import importlib.metadata; import packaging.version; print(packaging.version.Version(importlib.metadata.version('triton')).local[3:])")
TRITON_REPO_DIR="intel-xpu-backend-for-triton"

rm -rf intel-xpu-backend-for-triton
git clone --single-branch -b main --recurse-submodules https://github.com/intel/intel-xpu-backend-for-triton.git
cd $TRITON_REPO_DIR
git checkout $TRITON_COMMIT
cd ..

# Workaround for test-triton.sh scripts which always require local build
mkdir $TRITON_REPO_DIR/python/build

# PT dev dependencies
# pip install astunparse numpy ninja pyyaml cmake typing-extensions requests
# https://github.com/intel/intel-xpu-backend-for-triton/blob/1111a28c162f1f0fb48bda93b8ab441be4c2280a/scripts/test-triton.sh#L114
# python3 -m pip install lit pytest pytest-xdist pytest-rerunfailures pytest-select pytest-timeout setuptools==69.5.1 defusedxml
# https://github.com/intel/intel-xpu-backend-for-triton/blob/1111a28c162f1f0fb48bda93b8ab441be4c2280a/scripts/test-triton.sh#L223
# python3 -m pip install matplotlib pandas tabulate -q

pip install -r $TRITON_REPO_DIR/scripts/requirements-test.txt
pip install defusedxml
pip uninstall -y torch triton
pip freeze >requirements-offline.txt
mkdir -p third-party-wheels
pip download -r requirements-offline.txt -d wheels

tar -czvf wheels.tar.gz wheels
tar -czvf intel-xpu-backend-for-triton.tar.gz $TRITON_REPO_DIR

wget -O l_intel-for-pytorch-gpu-dev_p_offline.sh \
https://registrationcenter-download.intel.com/akdlm/IRC_NAS/d12ef2ba-7efd-4866-8f85-78eaf40b2fe2/intel-deep-learning-essentials-2025.0.1.27_offline.sh

wget -O l_intel-pti-dev_p_offline.sh \
https://registrationcenter-download.intel.com/akdlm/IRC_NAS/884eaa22-d56f-45dc-9a65-901f1c625f9e/l_intel-pti-dev_p_0.9.0.38_offline.sh

wget -O vulkan-sdk.tar.xz \
https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.xz
