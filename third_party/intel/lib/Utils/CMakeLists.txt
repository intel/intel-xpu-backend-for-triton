add_triton_library(TritonIntelUtils
  DefUseChain.cpp
  LibCallEmitter.cpp
  LLVMIntr.cpp
  Mangling.cpp
  Utility.cpp

  DEPENDS
  TritonGENTableGen
  TritonGENAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
