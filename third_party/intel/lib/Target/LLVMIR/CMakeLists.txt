add_subdirectory(Dialect)

add_triton_library(TritonIntelLLVMIR
  PostProcess.cpp
  LLVMIRFreezeMaskedDivRem.cpp
  LLVMIRScalarizePtrVectors.cpp

  DEPENDS
  LLVMIRIncGen
)
