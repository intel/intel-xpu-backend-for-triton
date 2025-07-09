add_triton_library(TritonIntelUtils
  DefUseChain.cpp
  LLVMIntr.cpp
  Mangling.cpp
  Utility.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)
