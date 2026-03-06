add_library(TritonTestAnalysis
  intel/TestAxisInfo.cpp
  intel/TestRangeAnalysis.cpp
  intel/TestStrideInfo.cpp
  TestAlias.cpp
  TestAxisInfo.cpp
  TestAllocation.cpp
  TestBufferRegion.cpp
  TestLivenessAnalysis.cpp
  TestMembar.cpp
)
target_link_libraries(TritonTestAnalysis PUBLIC MLIRPass TritonAnalysis TritonIntelAnalysis TritonNvidiaGPUTransforms)
target_compile_options(TritonTestAnalysis PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
