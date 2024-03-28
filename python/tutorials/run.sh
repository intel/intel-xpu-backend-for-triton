source /opt/intel/oneapi/2024.0/oneapi-vars.sh

export INTEL_ENABLE_BLOCK_PTR=1
export IGC_ShaderDumpEnable=1 IGC_DumpToCurrentDir=1
export MLIR_ENABLE_DUMP=1
export TRITON_DISABLE_LINE_INFO=1
export IGC_VISAOptions=" -TotalGRFNum 256 -enableBCR -nolocalra -printregusage -DPASTokenReduction -enableHalfLSC"
export IGC_ForcePrefetchToL1Cache=1
export IGC_VATemp=1
export UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0
export IGC_DisableLoopUnroll=1

export IGC_EnableVISANoSchedule=1
export TRITON_CACHE_DIR=/home/gta/deweiwang/xpu2/cache
#export DISABLE_LLVM_OPT=1
#export FROM_LLVM=1
