add_triton_library(TritonIntelUtils
  LLVMIntr.cpp
  Mangling.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)
