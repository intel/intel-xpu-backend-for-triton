add_triton_library(TritonGPUIR
  Dialect.cpp
  LinearLayoutConversions.cpp
  Ops.cpp
  Types.cpp
  Traits.cpp

  DEPENDS
  TritonGPUCGAAttrIncGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonIntelGPUAttrDefsIncGen
  TritonGPUOpInterfacesIncGen
  TritonIntelGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  TritonIR
  TritonTools
)
