add_triton_ut(
  NAME PrefetchCostModelTest
  SRCS PrefetchCostModelTest.cpp
  LIBS
    TritonIntelGPUTransforms
    TritonNvidiaGPUTransforms
)
