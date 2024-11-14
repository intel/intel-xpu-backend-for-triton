add_triton_library(TritonIntelGPUTransforms
  AccelerateMatmul.cpp
  Coalesce.cpp
  DistributeToWarps.cpp
  MatchTargetSize.cpp
  MaterializeBlockPointer.cpp
  OptimizeElementwiseParallelism.cpp
  OptimizeReductionLocality.cpp
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
