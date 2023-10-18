pip install cibuildwheel
export CIBW_MANYLINUX_X86_64_IMAGE="quay.io/pypa/manylinux2014_x86_64:latest"
export CIBW_BEFORE_BUILD="pip install cmake;"
export CIBW_SKIP="{cp,pp}{35,36}-*"
export CIBW_BUILD="{cp,pp}3*-manylinux_x86_64"
export CIBW_ENVIRONMENT="TRITON_CODEGEN_INTEL_XPU_BACKEND=1 http_proxy=${http_proxy} https_proxy=${https_proxy}"
python3 -m cibuildwheel python --platform linux --output-dir wheelhouse
