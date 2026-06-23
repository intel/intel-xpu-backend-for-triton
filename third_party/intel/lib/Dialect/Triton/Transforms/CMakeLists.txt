add_triton_library(TritonIntelTransforms
  DescriptorVersioning.cpp
  FoldTrueCmpIOp.cpp
  FuseReshape.cpp
  PrepareIfCombining.cpp
  OptimizeLoadMasks.cpp
  ReassociateDotScale.cpp
  RemoveMasks.cpp
  RewriteDescriptorGatherScatter.cpp
  RewriteTensorDescriptorToPointer.cpp
  SimplifySignedArithmetic.cpp
  StrideVersioning.cpp

  DEPENDS
  TritonIntelTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  TritonIntelAnalysis
  TritonIR
)
