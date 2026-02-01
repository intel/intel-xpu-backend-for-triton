add_triton_library(TritonGENToLLVM
  Attributes.cpp
  TritonGENToLLVMPass.cpp

  DEPENDS
  TritonGENToLLVMConversionPassIncGen
  TritonGENTableGen
  TritonGENAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  MLIRSPIRVDialect
  TritonIntelUtils
)
