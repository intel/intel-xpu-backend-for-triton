add_subdirectory(Dialect)

add_triton_library(TritonIntelLLVMIR
  PostProcess.cpp
  LLVMIRFreezeMaskedDivRem.cpp
  LLVMIROptimizeFloorDiv.cpp

  DEPENDS
  LLVMIRIncGen
)
