add_mlir_library(TritonTestAnalysis
  intel/TestAxisInfo.cpp
  TestAlias.cpp
  TestAxisInfo.cpp
  TestAllocation.cpp
  TestLivenessAnalysis.cpp
  TestMembar.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  TritonAnalysis
)
