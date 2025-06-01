add_triton_ut(
  NAME DPAStoLinearLayout
  SRCS DPAStoLinearLayoutTest.cpp
  LIBS
    TritonIntelGPUIR
    TritonGPUTransforms
    TritonIntelAnalysis
    TritonIntelGPUTransforms
    TritonNvidiaGPUTransforms
)
