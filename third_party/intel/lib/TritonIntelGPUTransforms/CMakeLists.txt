add_triton_library(TritonIntelGPUTransforms
  AccelerateMatmul.cpp
  DistributeToWarps.cpp
  MatchTargetSize.cpp
  Pipeliner/MatmulLoopPipeline.cpp
  Pipeliner/SoftwarePipeliner.cpp
  PrefetchBlock.cpp
  RemoveLayoutConversions.cpp
  RewriteTensorPointer.cpp
  Utility.cpp

  DEPENDS
  TritonIntelGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRSCFTransforms  
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonIntelGPUIR
)
