#!/bin/bash

set -xeuo pipefail

PROJECT_ROOT="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd .. && pwd )"
LLVM_REPO="${LLVM_REPO:-intel-innersource/drivers.gpu.compiler.llvm-pisa}"
if [[ ! -v LLVM_COMMIT ]]; then
    LLVM_COMMIT="$(<$PROJECT_ROOT/cmake/llvm-hash.txt)"
fi
LLVM_PREFIX="llvm-${LLVM_COMMIT:0:8}"
WORKSPACE="${GITHUB_WORKSPACE:-/tmp}"

if command -v pigz &> /dev/null; then
    GZIP=pigz
else
    GZIP=gzip
fi

checkout_llvm() {
    cd "$WORKSPACE"
        if [[ -d llvm-project ]]; then
            cd llvm-project
            git clean -xffd
            git reset --hard
            git fetch
        else
            git clone --no-checkout "https://github.com/$LLVM_REPO" llvm-project
            cd llvm-project
    fi
    git checkout "$LLVM_COMMIT"
}

build_almalinux() {
    cd "$WORKSPACE"
    mkdir -p scripts "$LLVM_PREFIX-almalinux-x64"

    docker run --rm --name almalinux --detach \
        --volume "$WORKSPACE/scripts:/scripts" \
        --volume "$WORKSPACE/llvm-project:/llvm-project" \
        --volume "$WORKSPACE/$LLVM_PREFIX-almalinux-x64:/llvm" \
        quay.io/pypa/manylinux_2_28_x86_64:latest sleep inf

    cat <<EOF | tee scripts/build-almalinux.sh
    dnf install clang lld -y
    python3.9 -m venv /venv
    source /venv/bin/activate

    pip install cmake ninja -r /llvm-project/mlir/python/requirements.txt

    cmake -GNinja -Bllvm-project/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_ASM_COMPILER=clang \
        -DCMAKE_CXX_FLAGS="-Wno-everything" \
        -DCMAKE_LINKER=lld \
        -DCMAKE_INSTALL_PREFIX=/llvm \
        -DLLVM_BUILD_UTILS=ON \
        -DLLVM_BUILD_TOOLS=ON \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DLLVM_ENABLE_PROJECTS="mlir;lld" \
        -DENABLE_PISA_3D=true \
        -DLLVM_INSTALL_UTILS=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU;SPIRV" \
        -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="pISA;Xe" \
        -DLLVM_ENABLE_TERMINFO=OFF \
        llvm-project/llvm

    ninja -C /llvm-project/build check-mlir install
EOF

    docker exec almalinux bash -x /scripts/build-almalinux.sh
    docker rm -f almalinux
}

upload_almalinux() {
    cd "$WORKSPACE"
    tar -I "$GZIP -9" -cf "$LLVM_PREFIX-almalinux-x64.tar.gz" "$LLVM_PREFIX-almalinux-x64"
    ls -lh "$LLVM_PREFIX-almalinux-x64.tar.gz"
    aws --endpoint-url=http://s3.icx.x1infra.com s3 cp "$LLVM_PREFIX-almalinux-x64.tar.gz" "s3://ceph-bkt-f9889ed3-723e-4693-99c3-56622282640e/llvm/"
}

build_centos() {
    cd "$WORKSPACE"
    mkdir -p scripts "$LLVM_PREFIX-centos-x64"

    docker run --rm --name centos7 --detach \
        --volume "$WORKSPACE/scripts:/scripts" \
        --volume "$WORKSPACE/llvm-project:/llvm-project" \
        --volume "$WORKSPACE/$LLVM_PREFIX-centos-x64:/llvm" \
        quay.io/pypa/manylinux2014_x86_64 sleep inf

    cat <<EOF | tee scripts/build-centos.sh
    python3.9 -m venv /venv
    source /venv/bin/activate

    pip install cmake ninja -r /llvm-project/mlir/python/requirements.txt

    cmake -GNinja -Bllvm-project/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/llvm \
        -DLLVM_BUILD_UTILS=ON \
        -DLLVM_BUILD_TOOLS=ON \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DLLVM_ENABLE_PROJECTS="mlir;lld" \
        -DENABLE_PISA_3D=true \
        -DLLVM_INSTALL_UTILS=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU;SPIRV" \
        -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="pISA;Xe" \
        -DLLVM_ENABLE_TERMINFO=OFF \
        -DLLVM_ABI_BREAKING_CHECKS=FORCE_OFF \
        llvm-project/llvm

    ninja -C /llvm-project/build check-mlir install
EOF

    docker exec centos7 bash -x /scripts/build-centos.sh
    docker rm -f centos7
}

upload_centos() {
    cd "$WORKSPACE"
    tar -I "$GZIP -9" -cf "$LLVM_PREFIX-centos-x64.tar.gz" "$LLVM_PREFIX-centos-x64"
    ls -lh "$LLVM_PREFIX-centos-x64.tar.gz"
    aws --endpoint-url=http://s3.icx.x1infra.com s3 cp "$LLVM_PREFIX-centos-x64.tar.gz" "s3://ceph-bkt-f9889ed3-723e-4693-99c3-56622282640e/llvm/"
}

build_ubuntu() {
    cd "$WORKSPACE"
    mkdir -p scripts "$LLVM_PREFIX-ubuntu-x64"

    docker run --rm --name ubuntu --detach \
        --volume "$WORKSPACE/scripts:/scripts" \
        --volume "$WORKSPACE/llvm-project:/llvm-project" \
        --volume "$WORKSPACE/$LLVM_PREFIX-ubuntu-x64:/llvm" \
        ubuntu:22.04 sleep inf

    cat <<EOF | tee scripts/build-ubuntu.sh
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y --no-install-recommends --fix-missing \
        python3 python3-dev python3-pip python-is-python3 pkg-config build-essential zlib1g-dev ninja-build git

    pip install cmake ninja -r llvm-project/mlir/python/requirements.txt

    cmake -GNinja -Bllvm-project/build \
        -DLLVM_ENABLE_DUMP=1 \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS="mlir;lld" \
        -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU;SPIRV" \
        -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="pISA;Xe" \
        -DENABLE_PISA_3D=true \
        -DLLVM_INSTALL_UTILS=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_INSTALL_PREFIX=/llvm \
        llvm-project/llvm

    ninja -C llvm-project/build check-mlir install
EOF

    docker exec ubuntu bash -x /scripts/build-ubuntu.sh
    docker rm -f ubuntu
}

upload_ubuntu() {
    cd "$WORKSPACE"
    tar -I "$GZIP -9" -cf "$LLVM_PREFIX-ubuntu-x64.tar.gz" "$LLVM_PREFIX-ubuntu-x64"
    ls -lh "$LLVM_PREFIX-ubuntu-x64.tar.gz"
    aws --endpoint-url=http://s3.icx.x1infra.com s3 cp "$LLVM_PREFIX-ubuntu-x64.tar.gz" "s3://ceph-bkt-f9889ed3-723e-4693-99c3-56622282640e/llvm/"
}
