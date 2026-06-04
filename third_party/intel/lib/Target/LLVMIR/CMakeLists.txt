add_subdirectory(Dialect)

add_triton_library(TritonIntelLLVMIR
  PostProcess.cpp
  LLVMIRGuardMaskedDivRem.cpp
  LLVMIRScalarizePtrVectors.cpp

  DEPENDS
  LLVMIRIncGen
)
