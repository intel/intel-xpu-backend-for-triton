add_triton_library(TritonIntelGPUTransforms
  AccelerateMatmul.cpp
  DistributeToWarps.cpp
  MatchTargetSize.cpp
  Pipeliner/MatmulLoopPipeline.cpp
  Pipeliner/SoftwarePipeliner.cpp
  PrefetchBlock.cpp
  ReduceDataDuplication.cpp
  RemoveLayoutConversions.cpp
  RewriteTensorPointer.cpp
  ScheduleLoad.cpp
  Utility.cpp

  DEPENDS
  TritonIntelGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRSCFTransforms
  MLIRTransforms
  MLIRTransformUtils
  TritonIntelAnalysis
  TritonIR
  TritonGENIR
  TritonGPUIR
  TritonIntelGPUIR
)
