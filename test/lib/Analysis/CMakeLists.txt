add_mlir_library(TritonTestAnalysis
  TestAlias.cpp
  TestAxisInfo.cpp
  TestAllocation.cpp
  TestLivenessAnalysis.cpp
  TestMembar.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  ${triton_libs}
)
