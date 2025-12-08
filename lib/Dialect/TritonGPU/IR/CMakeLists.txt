add_triton_library(TritonGPUIR
  Dialect.cpp
  LinearLayoutConversions.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  TritonGPUCTAAttrIncGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonIntelGPUAttrDefsIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  TritonIR
  TritonTools
)
