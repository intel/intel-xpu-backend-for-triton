add_mlir_library(TritonTestAnalysis
  intel/TestAxisInfo.cpp
  intel/TestRangeAnalysis.cpp
  TestAlias.cpp
  TestAxisInfo.cpp
  TestAllocation.cpp
  TestLivenessAnalysis.cpp
  TestMembar.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  TritonAnalysis
)
