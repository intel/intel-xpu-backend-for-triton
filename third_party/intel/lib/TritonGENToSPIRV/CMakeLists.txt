add_triton_library(TritonGENToSPIRV
  TritonGENToSPIRVPass.cpp

  DEPENDS
  TritonGENToSPIRVConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRSPIRVDialect
)
