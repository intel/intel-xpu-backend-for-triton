add_library(TritonTestAnalysis
  intel/TestAxisInfo.cpp
  intel/TestRangeAnalysis.cpp
  TestAlias.cpp
  TestAxisInfo.cpp
  TestAllocation.cpp
  TestBufferRegion.cpp
  TestLivenessAnalysis.cpp
  TestMembar.cpp
)
target_link_libraries(TritonTestAnalysis PUBLIC MLIRPass TritonAnalysis)
target_compile_options(TritonTestAnalysis PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
