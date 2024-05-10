export TRITON_INTEL_ENABLE_BLOCK_PTR=1
export IGC_VISAOptions=" -TotalGRFNum 256 -enableBCR -nolocalra -printregusage -DPASTokenReduction -enableHalfLSC"
export IGC_ForcePrefetchToL1Cache=1
export IGC_VATemp=1
export UR_L0_IN_ORDER_BARRIER_BY_SIGNAL=0
export IGC_DisableLoopUnroll=1
export IGC_EnableVISANoSchedule=1
