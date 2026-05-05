add_triton_library(TritonIntelGPUTransforms
  AccelerateMatmul.cpp
  AnnotateCacheControl.cpp
  BlockIOUtils.cpp
  DecomposeScaledBlocked.cpp
  FixupDescriptorEncoding.cpp
  HoistLayoutConversions.cpp
  LowerTo2DBlockLoad.cpp
  MaterializeBlockPointer.cpp
  OptimizeDotOperands.cpp
  OptimizeReductionLocality.cpp
  Pipeliner/MatmulLoopPipeline.cpp
  Pipeliner/SoftwarePipeliner.cpp
  ReduceDataDuplication.cpp
  ReduceVariableLiveness.cpp
  RemoveLayoutConversions.cpp
  RewriteStackPtr.cpp
  Utility.cpp
  WidenLoadStoreEncoding.cpp

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
