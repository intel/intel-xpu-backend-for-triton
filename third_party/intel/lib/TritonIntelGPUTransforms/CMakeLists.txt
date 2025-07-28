add_triton_library(TritonIntelGPUTransforms
  AccelerateMatmul.cpp
  Coalesce.cpp
  DecomposeScaledBlocked.cpp
  DistributeToWarps.cpp
  MatchTargetSize.cpp
  MaterializeBlockPointer.cpp
  OptimizeDotOperands.cpp
  OptimizeReductionLocality.cpp
  Pipeliner/MatmulLoopPipeline.cpp
  Pipeliner/SoftwarePipeliner.cpp
  PrefetchBlock.cpp
  ReduceDataDuplication.cpp
  ReduceVariableLiveness.cpp
  RemoveLayoutConversions.cpp
  RewriteStackPtr.cpp
  ScheduleLoad.cpp
  Utility.cpp

  DEPENDS
  TritonGPUTransforms
  TritonIntelGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRSCFTransforms
  MLIRSPIRVDialect
  MLIRTransforms
  MLIRTransformUtils
  TritonIntelAnalysis
  TritonIR
  TritonGENIR
  TritonGPUIR
  TritonIntelGPUIR
  TritonIntelUtils
)
