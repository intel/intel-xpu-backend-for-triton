add_triton_library(TritonGENToSPIRV
  TritonGENToSPIRVPass.cpp

  DEPENDS
  TritonGENToSPIRVConversionPassIncGen
  TritonGENTableGen
  TritonGENAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRSPIRVDialect
)
