add_triton_library(GENIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  GENTableGen
  GENAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
)
