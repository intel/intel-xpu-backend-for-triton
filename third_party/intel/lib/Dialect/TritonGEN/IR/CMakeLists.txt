add_triton_library(TritonGENIR
  TritonGENAttrs.cpp
  TritonGENDialect.cpp
  TritonGENOps.cpp

  DEPENDS
  TritonGENTableGen
  TritonGENAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
)
