add_triton_library(TritonIntelTritonIR
  TritonIntelDialect.cpp

  DEPENDS
  TritonIntelTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  TritonIR
)
