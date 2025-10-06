add_triton_library(TritonIntelUtils
  DefUseChain.cpp
  LibCallEmitter.cpp
  LLVMIntr.cpp
  Mangling.cpp
  Utility.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)
