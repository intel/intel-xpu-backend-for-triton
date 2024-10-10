add_triton_library(TritonAnalysis
  intel/TestAxisInfo.cpp
  AxisInfo.cpp
  Allocation.cpp
  Membar.cpp
  Alias.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
)
