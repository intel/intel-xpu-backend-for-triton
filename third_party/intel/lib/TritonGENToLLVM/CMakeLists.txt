add_triton_library(TritonGENToLLVM
  TritonGENToLLVMPass.cpp

  DEPENDS
  TritonGENToLLVMConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  MLIRSPIRVDialect
  TritonIntelUtils
)
