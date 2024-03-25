add_triton_library(TritonIntelGPUTransforms
  AccelerateMatmul.cpp
  MatchTargetSize.cpp
  PrefetchBlock.cpp
  Utility.cpp

  DEPENDS
  TritonIntelGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonIntelGPUIR
)
