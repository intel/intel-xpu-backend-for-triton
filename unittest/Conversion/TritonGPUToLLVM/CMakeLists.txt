add_triton_ut(
  NAME TritonLLVMOpBuilderTest
  SRCS TritonLLVMOpBuilderTest.cpp
  LIBS
    TritonGPUIR
    TritonGPUTransforms
    TritonIntelAnalysis
    TritonIntelGPUTransforms
    TritonNvidiaGPUTransforms
)
