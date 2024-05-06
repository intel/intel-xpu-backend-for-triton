add_triton_library(TritonIntelGPUIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TritonIntelGPUTableGen
  TritonIntelGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRSideEffectInterfaces
)
