add_subdirectory(Dialect)

add_triton_library(TritonIntelLLVMIR
  PostProcess.cpp
  SLPVectorizer.cpp
  LLVMIRFreezeMaskedDivRem.cpp

  DEPENDS
  LLVMIRIncGen
)
