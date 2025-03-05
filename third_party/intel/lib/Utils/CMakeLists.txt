add_triton_library(TritonIntelUtils
  LLVMIntr.cpp
  Mangling.cpp
  Utility.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)
