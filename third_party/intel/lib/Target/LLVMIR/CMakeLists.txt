add_subdirectory(Dialect)

add_triton_library(TritonIntelLLVMIR
  PostProcess.cpp
  LLVMIRGuardMaskedDivRem.cpp

  DEPENDS
  LLVMIRIncGen
)
