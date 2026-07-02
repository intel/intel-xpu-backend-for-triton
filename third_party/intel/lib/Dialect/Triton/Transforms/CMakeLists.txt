add_triton_library(TritonIntelTransforms
  DescriptorVersioning.cpp
  FoldTrueCmpIOp.cpp
  FuseReshape.cpp
  PrepareIfCombining.cpp
  ReassociateDotScale.cpp
  RemoveMasks.cpp
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
