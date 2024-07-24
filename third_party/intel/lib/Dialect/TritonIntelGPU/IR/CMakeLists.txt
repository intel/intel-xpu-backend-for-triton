add_triton_library(TritonIntelGPUIR
  Dialect.cpp
  Ops.cpp
  LinearLayoutConversions.cpp
  DEPENDS
  TritonIntelGPUTableGen
  TritonIntelGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRSideEffectInterfaces
)
