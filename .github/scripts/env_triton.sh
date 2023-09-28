# compiler
source ${HOME}/intel/oneapi/compiler/latest/env/vars.sh
# oneMKL
source ${HOME}/intel/oneapi/mkl/latest/env/vars.sh
export MKL_DPCPP_ROOT=${MKLROOT}
export LD_LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:$LIBRARY_PATH

# tbb
source ${HOME}/intel/oneapi/tbb/latest/env/vars.sh

# dnnl
source ${HOME}/intel/oneapi/dnnl/latest/env/vars.sh

# IPEX
export USE_AOT_DEVLIST='12.55.8'

# IMM ENV option
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=0
export ExperimentalCopyThroughLock=1
